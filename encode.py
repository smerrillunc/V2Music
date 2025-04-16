#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import EncodecModel, AutoProcessor
from torch.amp import autocast

def main():
    parser = argparse.ArgumentParser(
        description="Process and encode WAV files with Encodec."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/work/users/t/i/tis/V2Music/preprocessing/bg_audio/mdx_extra",
        help="Path to the directory containing the WAV files."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Batch size for processing WAV files."
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index for slicing the sorted list of WAV files."
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index for slicing the sorted list of WAV files (non-inclusive). If not set, process until the end."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the encoded tensors. Defaults to a folder named 'encoded_tensors' inside data_dir."
    )
    parser.add_argument(
        "--out_format",
        type=str,
        default="npz",
        choices=["npz", "pt"],
        help="Output file format: 'npz' for compressed numpy archives or 'pt' for PyTorch tensors. Default is npz."
    )
    parser.add_argument(
        "ids", nargs="*", default=None,
        help="Optional list of IDs to filter the wav files. Only files whose filenames contain one of these IDs will be processed."
    )
    args = parser.parse_args()
    
    data_dir = args.data_dir
    batch_size = args.batch_size
    start_index = args.start
    end_index = args.end
    out_format = args.out_format

    # Set output directory. If not provided, use 'encoded_tensors' inside data_dir.
    if args.output_dir is None:
        output_enc_dir = os.path.join(data_dir, "encoded_tensors")
    else:
        output_enc_dir = args.output_dir
    os.makedirs(output_enc_dir, exist_ok=True)
    
    print("Loading model and processor...")
    # Load the model and processor.
    model = EncodecModel.from_pretrained("facebook/encodec_48khz")
    processor = AutoProcessor.from_pretrained("facebook/encodec_48khz")
    target_sr = processor.sampling_rate  # Typically 48000 Hz

    # Find WAV files in the specified directory.
    wav_files = sorted(glob.glob(os.path.join(data_dir, "*.wav")))

    #Allow to only encode specific files
    if args.ids:
        wav_files = [f for f in wav_files if any(_id in os.path.basename(f) for _id in args.ids)]

    if end_index is None or end_index > len(wav_files):
        args.end = len(wav_files)

    if not wav_files:
        print("No WAV files found in the directory:", data_dir)
        exit(1)
    
    # Slice the list according to start and end indices.
    wav_files = wav_files[start_index:end_index]
    total_files = len(wav_files)
    print(f"Found {total_files} WAV files to process.")

    # Move model to GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    err_count = 0
    
    # Process files in batches.
    for i in range(0, total_files, batch_size):
        batch_files = wav_files[i:i+batch_size]
        audio_list = []
        # build this list to skip files with errors
        successful_files = []
        
        for wav_file in batch_files:
            print("Reading file:", wav_file)
            try:
                audio, sr = sf.read(wav_file)
                
                # Ensure audio is stereo with shape (channels, num_samples)
                if audio.ndim == 1:
                    audio = np.vstack((audio, audio))
                elif audio.ndim == 2:
                    if audio.shape[1] == 2:
                        audio = audio.T
                    elif audio.shape[1] > 2:
                        audio = audio[:, :2].T
                    else:
                        audio = np.vstack((audio[:, 0], audio[:, 0]))
                else:
                    raise ValueError("Unsupported audio dimensions for file: " + wav_file)
                
                # Resample if needed.
                if sr != target_sr:
                    resampled_channels = []
                    for ch in range(audio.shape[0]):
                        resampled = librosa.resample(audio[ch], orig_sr=sr, target_sr=target_sr)
                        resampled_channels.append(resampled)
                    audio = np.vstack(resampled_channels)
                
                audio_list.append(audio)
                successful_files.append(wav_file) 
            except Exception as e:
                err_count += 1
                print(f"{err_count} Error reading file {wav_file}: {e}")
                continue
        
        # Process batch with the processor.
        inputs = processor(
            raw_audio=audio_list,
            sampling_rate=target_sr,
            padding=True,
            return_tensors="pt",
        )
        # Move inputs to the same device as the model.
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # Encode the audio batch.
            encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"], bandwidth=24.0)
        
        # Process the model outputs.
        audio_codes = encoder_outputs.audio_codes.transpose(0, 1).cpu()
        audio_scales = torch.stack(encoder_outputs.audio_scales, dim=1).cpu()
        
        # Save each file's encoded output.
        for j, wav_file in enumerate(successful_files):
            base_name = os.path.splitext(os.path.basename(wav_file))[0]
            if out_format == "pt":
                output_file = os.path.join(output_enc_dir, f"{base_name}.pt")
                encoded_dict = {
                    "audio_codes": audio_codes[j],
                    "audio_scales": audio_scales[j],
                    "padding_mask": inputs["padding_mask"][j]
                }
                torch.save(encoded_dict, output_file)
            elif out_format == "npz":
                output_file = os.path.join(output_enc_dir, f"{base_name}.npz")
                np.savez_compressed(
                    output_file,
                    audio_codes=audio_codes[j].numpy(),
                    audio_scales=audio_scales[j].numpy(),
                    padding_mask=inputs["padding_mask"][j].cpu().numpy()
                )
            else:
                raise ValueError("Unsupported file format: " + out_format)
            print(f"Saved encoded tensor to {output_file}")

if __name__ == "__main__":
    main()