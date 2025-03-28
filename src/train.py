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

from models import AutoregressiveModel

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)

    parser = argparse.ArgumentParser(description='Read file content.')

    # TRANSFORMER PARAMS
    parser.add_argument("-ms", "--max_seq_len", type=int, default=200, help='Max sequence laength for Transformer Encoders')
    parser.add_argument("-nh", "--num_heads", type=int, default=1, help='Number of Heads for Transformer Encoders')
    parser.add_argument("-nl", "--num_layers", type=int, default=1, help='Number of Layers for Transformer Encoders')

    parser.add_argument("-vd", "--vid_input_dim", type=int, default=128, help='Audio input dimension')
    parser.add_argument("-msl", "--music_seq_len", type=int, default=1024, help='Video input dimension')
    parser.add_argument("-ed", "--embed_dim", type=int, default=256, help='Embedding dimension')


    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help='Learning Rate for Encoders')

    # Learning Params
    parser.add_argument("-bs", "--batch_size", type=int, default=10, help='Train Batch Size')
    parser.add_argument("-tbs", "--test_batch_size", type=int, default=10, help='Test Batch Size')
    parser.add_argument("-e", "--epochs", type=int, default=1000, help='Epochs')
    parser.add_argument("-sw", "--short_window", type=int, default=2, help='Num frames for short window to consider')
    parser.add_argument("-lw", "--long_window", type=int, default=5, help='Num frames for long window to consider')
    parser.add_argument("-cs", "--codebook_size", type=int, default=1024, help='Codebook size.  IMPORTANT THIS SHOULD MATCH WHAT IS USED FOR TRAIN DATA')

    # Admin params

    # Longleaf
    #parser.add_argument("-sp", "--save_path", type=str, default='/nas/longleaf/home/smerrill/PD/data', help='save path')
    #parser.add_argument("-dp", "--data_path", type=str, default='/nas/longleaf/home/smerrill/PD/data', help='dataset path')

    # Local
    parser.add_argument("-sp", "--save_path", type=str, default='/Users/scottmerrill/Documents/UNC/MultiModal/VMR/checkpoints', help='save path')
    parser.add_argument("-dp", "--data_path", type=str, default='/Users/scottmerrill/Documents/UNC/MultiModal/VMR/Youtube8m', help='dataset path')

    args = vars(parser.parse_args())

    # make checkpoint dir
    os.makedirs(args['save_path'], exist_ok=True)

    # x should be passed in shape: (batch_size, video_seq_length, video_input_dim)
    model = AutoregressiveModel(vid_input_dim=args['vid_input_dim'], music_seq_len=args['music_seq_len'], codebook_vocab=args['codebook_size'])
    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])


    train_filenames = pd.read_csv(args['data_path']+'/train.csv')['filename'].values
    test_filenames = pd.read_csv(args['data_path']+'/test.csv')['filename'].values

    train_dataset = VideoAudioDataset(args['data_path'], train_filenames)
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, )

    test_dataset = VideoAudioDataset(args['data_path'], test_filenames)
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=True, )


    # to store outputs
    df = pd.DataFrame()

    criterion = nn.CrossEntropyLoss()

    # Batch iterator
    for epoch in tqdm.tqdm(range(args['epochs'])):
        for video_feats, audio_targets in train_loader:
            try:
                optimizer.zero_grad()
                predictions = model(x)

                # predictions are logits for audio codes with CELoss Function
                loss = criterion(predictions, audio_targets)
                optimizer.step()

                print("HERE")
                break
                
            except Exception as e:
                # adding a wrapper just in case
                print(e)

        if epoch % 10 == 0:
            # save checkpoint and evaluate after every 10 epochs
            model.eval()
            #utils.save_checkpoint(model, optimizer, epoch, args['save_path'] + f'/{epoch}.pth')
            print(f'Train Loss {loss}')

            # path here is path to test set data
            #tmp = utils.compute_evaluations(video_model, audio_model, args['test_batch_size'], args['max_seq_len'], args['window_size'],\
                                            #args['segments'],args['min_frames'], args['data_path'], test_filenames ,epoch, ks=[1, 5])
            #df = pd.concat([df, tmp])
            #df.to_csv(args['save_path'] + f'/eval.csv', index=False)
            #audio_model.train()
            model.train()
