import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
# Set the default dtype to float32
torch.set_default_dtype(torch.float32)


class VideoAudioDataset(Dataset):
    def __init__(self, path, filenames):
        self.path = path
        self.filenames = filenames
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # Load data from files and ensure all are float32
        video_data = np.load(os.path.join(self.path, 'video', filename)).astype(np.float32)
        flow_data = np.load(os.path.join(self.path, 'optical_flow', filename)).astype(np.float32)
        audio_data = np.load(os.path.join(self.path, 'audio', filename)).astype(np.float32)
        
        # Process video data: Ensure it's in float32 and slice
        video_data = video_data[:, :1024].astype(np.float32)  # Fix slicing and ensure float32
        
        # Process flow data: Prepend 0, ensure it's in float32
        flow_data = np.concatenate(([0], flow_data)).astype(np.float32)  # Dirty fix for first flow feature
        
        # Convert to torch tensors
        video_data = torch.tensor(video_data)
        flow_data = torch.tensor(flow_data)
        audio_data = torch.tensor(audio_data)
        
        return video_data, flow_data, audio_data