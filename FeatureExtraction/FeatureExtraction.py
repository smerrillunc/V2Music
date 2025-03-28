#!/usr/bin/env python

import yt_dlp
import os
import gc

import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
import numpy as np
from sklearn.decomposition import PCA

import torchaudio
from torchaudio.prototype.pipelines import VGGISH

#import ffmpeg
from moviepy import AudioFileClip

import tqdm
import argparse
from transformers import EncodecModel, AutoProcessor

import librosa

def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
        print(f"The file {filename} has been deleted.")
    else:
        print(f"The file {filename} does not exist.")


def extract_video_features(video_path, max_frames=360):

  # Define preprocessing transformations
  preprocess = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize((299, 299)),
      transforms.ToTensor(),
      # normalize based on mean and standard deviation of imagenet dataset
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  # Open video file
  cap = cv2.VideoCapture(video_path)
  fps = cap.get(cv2.CAP_PROP_FPS)

  features = []
  frame_count = 0

  with torch.no_grad():  # No gradients needed for inference
      while cap.isOpened() and frame_count < max_frames*fps:
          ret, frame = cap.read()
          if not ret:
              break

          # Process one frame per second
          if frame_count % int(fps) == 0:
              # Preprocess the frame
              frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
              input_tensor = preprocess(frame).unsqueeze(0)  # Add batch dimension

              # Extract features using the InceptionV3 model
              feature_vector = inception(input_tensor)
              features.append(feature_vector.squeeze(0).cpu().numpy())  # Convert to numpy array

          frame_count += 1

  cap.release()
  return np.array(features)


def extract_audio_features(audio_path):   
    # Load and preprocess audio
    audio_sample, sample_rate = librosa.load(audio_path, sr=processor.sampling_rate, duration=10)    
    
    # pre-process the audio inputs
    inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")

    # explicitly encode then decode the audio inputs
    encoder_outputs = encodec.encode(inputs["input_values"], inputs["padding_mask"])
    
    return encoder_outputs.audio_codes.flatten().numpy()



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Read file content.')

  parser.add_argument("-s", "--start_index", type=int, default=0, help='YoutubeID Index to start on in YoutubeID File')
  parser.add_argument("-e", "--end_index", type=int, default=100, help='YoutubeID Index to end on in YoutubeID File')
  parser.add_argument("-p", "--path", type=str, default='/Users/scottmerrill/Desktop', help='Path to YoutubeID file.  This will also be where output featuers are saved')
  args = vars(parser.parse_args())


  inception = inception_v3(pretrained=True, transform_input=False)
  inception.fc = torch.nn.Identity()  # Remove the classification layer (we only need features)
  inception.eval()  # Set the model to evaluation mode
  print('Loaded InceptionV3')

  os.makedirs(args['path'] + '/video', exist_ok=True)
  os.makedirs(args['path'] + '/audio', exist_ok=True)

  downloaded_vids = os.listdir(args['path'] + '/audio')
  downloaded_vids = [x.split('.')[0] for x in downloaded_vids]


  # load the model + processor (for pre-processing the audio)
  encodec = EncodecModel.from_pretrained("facebook/encodec_24khz")
  codebook_size = encodec.config.codebook_size

  processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

  # Here are the youtube ids used by original VM-NET
  with open(args['path'] + '/Youtube_ID.txt', 'r') as file:
      content = file.read()  # Read the entire content of the file
  youtube_urls = content.split('\n')

  youtube_urls = youtube_urls[args['start_index']:args['end_index']]

  for video_url in tqdm.tqdm(youtube_urls):
      # video id      
      vid = video_url.split('=')[1]

      if vid in downloaded_vids:
        print(f"VID: {vid} alread processed, skipping")
        continue


      print(f"processing VID: {vid}")

      try:
          ydl_opts = {
              'quiet': True,  # Suppresses verbose output
              'format': 'mp4',  # Directly download the best MP4 format available
              'outtmpl': args['path'] + f'/{vid}.%(ext)s',  # Customize output filename
              'noplaylist': True,  # Ensure only the video itself is downloaded, not a playlist
              'postprocessor_args': [
                  '-ss', '00:00:00',  # Start from the beginning of the video
                  '-t', '360',  # Limit to 360 seconds (6 minutes)
              ],
              'cookiefile': args['path'] + '/cookies.txt',  # Use cookies from the file

          }

          # download youtube video
          with yt_dlp.YoutubeDL(ydl_opts) as ydl:
              ydl.download([video_url])


          # once we download all training features, we have to do pca whitening
          video_feats = extract_video_features(args['path'] + f'/{vid}.mp4')
          
          # convert mp4 to mp3 and get audio features
          
          # Load MP4 file and extract audio
          #audio = AudioFileClip(args['path'] + f'/{vid}.mp4')
            
          # Write audio to MP3 file
          #audio.write_audiofile(args['path'] + f'/{vid}.mp3')

          #_ = ffmpeg.input('tmp.mp4').output(args['path']  '/tmp.mp3').global_args('-loglevel', 'quiet', '-y').run()
          audio_feats = extract_audio_features(args['path'] + f'/{vid}.mp4')
      
          np.save(args['path'] + f'/video/{vid}.npy', video_feats)
          np.save(args['path'] + f'/audio/{vid}.npy', audio_feats)        
          del video_feats, audio_feats
          gc.collect()

      except Exception as e:
          print(video_url, e)  

      # delete files if they exist
      delete_file(args['path'] + f'/{vid}.mp4')
      delete_file(args['path'] + f'/{vid}.mp3')

      
  print("Download Complete")
