import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim

import itertools
import random

import argparse
import tqdm

from DataLoader import VideoAudioDataset
from torch.utils.data import DataLoader, Dataset

from models import AutoregressiveModelSegmented
import utils

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)

    parser = argparse.ArgumentParser(description='Read file content.')

    # TRANSFORMER PARAMS
    parser.add_argument("-ms", "--max_seq_len", type=int, default=200, help='Max sequence laength for Transformer Encoders')
    parser.add_argument("-nh", "--num_heads", type=int, default=1, help='Number of Heads for Transformer Encoders')
    parser.add_argument("-nl", "--num_layers", type=int, default=1, help='Number of Layers for Transformer Encoders')
    parser.add_argument("-ed", "--embed_dim", type=int, default=50, help='Embedding dimension')
    parser.add_argument("-ns", "--num_segments", type=int, default=10, help='Number of segments')

    parser.add_argument("-vd", "--vid_input_dim", type=int, default=1024, help='Video input dimension')
    parser.add_argument("-msl", "--music_seq_len", type=int, default=9002, help='Video input dimension')


    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help='Learning Rate for Encoders')

    # Learning Params
    parser.add_argument("-bs", "--batch_size", type=int, default=5, help='Train Batch Size')
    parser.add_argument("-tbs", "--test_batch_size", type=int, default=10, help='Test Batch Size')
    parser.add_argument("-e", "--epochs", type=int, default=1000, help='Epochs')
    parser.add_argument("-cs", "--codebook_size", type=int, default=1024, help='Codebook size.  IMPORTANT THIS SHOULD MATCH WHAT IS USED FOR TRAIN DATA')

    # Admin params

    # Longleaf
    #parser.add_argument("-sp", "--save_path", type=str, default='/nas/longleaf/home/smerrill/PD/data', help='save path')
    #parser.add_argument("-dp", "--data_path", type=str, default='/nas/longleaf/home/smerrill/PD/data', help='dataset path')

    # Local
    parser.add_argument("-sp", "--save_path", type=str, default='/Users/scottmerrill/Documents/UNC/Vision Transformers/V2Music/checkpoints', help='save path')
    parser.add_argument("-dp", "--data_path", type=str, default='/Users/scottmerrill/Documents/UNC/Vision Transformers/V2Music/V2M', help='dataset path')

    args = vars(parser.parse_args())

    # Detect device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # make checkpoint dir
    os.makedirs(args['save_path'], exist_ok=True)

    # x should be passed in shape: (batch_size, video_seq_length, video_input_dim)
    model = AutoregressiveModelSegmented(vid_input_dim=args['vid_input_dim'], 
                                        music_seq_len=args['music_seq_len'],
                                         codebook_vocab=args['codebook_size'],
                                         hidden_dim=args['embed_dim'],
                                         num_layers=args['num_layers'],
                                         num_heads=args['num_heads'],
                                         num_segments=args['num_segments'])
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])


    train_filenames = pd.read_csv(args['data_path']+'/train.csv')['filename'].values[:10]
    test_filenames = pd.read_csv(args['data_path']+'/test.csv')['filename'].values[:10]

    train_dataset = VideoAudioDataset(args['data_path'], train_filenames)
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, )

    # we'll compute this after
    #test_dataset = VideoAudioDataset(args['data_path'], test_filenames)
    #test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=True, )


    # to store outputs
    df = pd.DataFrame()

    criterion = nn.CrossEntropyLoss()

    # Batch iterator
    for epoch in tqdm.tqdm(range(args['epochs'])):
        i = 0
        losses = []
        for video_feats, flow_feats, audio_targets in train_loader:
            video_feats = video_feats.to(device)
            flow_feats = flow_feats.to(device)
            audio_targets = audio_targets.to(device)

            optimizer.zero_grad()
            predictions = model(video_feats, flow_feats)

            #print(predictions.view(-1, args['codebook_size']).shape) #[batch_size*music_seq_len, codebooksize]
            #print(audio_targets.shape)#[batch_size*music_seq_len]

            # predictions are logits for audio codes with CELoss Function
            loss = criterion(predictions.view(-1, args['codebook_size']), (audio_targets.long()).view(-1)) 
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().item())
            print(f'Batch Train Loss {loss}')

        tmp = pd.DataFrame({'Epoch':epoch,
                            'Loss':np.mean(losses)}, index=[0])
        
        df = pd.concat([df, pd.DataFrame(tmp, index=[0])])
        df.to_csv(args['save_path'] + f'/losses.csv', index=False)

        if epoch % 10 == 0:
            # save checkpoint and evaluate after every 10 epochs
            #model.eval()
            utils.save_checkpoint(model, optimizer, epoch, args['save_path'] + f'/{epoch}.pth')

            # path here is path to test set data
            # This test data is super slow, so let's just do it one time after
            #tmp = utils.compute_evaluations(model, test_loader, epoch)
            #df = pd.concat([df, tmp])
            #df.to_csv(args['save_path'] + f'/eval.csv', index=False)
            #model.train()
