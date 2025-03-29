import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
# Set the default dtype to float32
torch.set_default_dtype(torch.float32)


class VideoAudioDataset(Dataset):
    def __init__(self, path, filenames):
        """
        Args:
            path (str): Path to the directory containing the video, optical_flow, and audio subdirectories.
            filenames (list): List of filenames to be loaded.
        """
        self.path = path
        self.filenames = filenames
    
    def __len__(self):
        """Returns the total number of samples"""
        return len(self.filenames)
    
    def __getitem__(self, idx):
        """
        Lazily loads the data for a single sample when requested.
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            video_data (tensor): Video data tensor for the sample.
            flow_data (tensor): Flow data tensor for the sample.
            audio_data (tensor): Audio data tensor for the sample.
        """
        filename = self.filenames[idx]
        
        # Construct the paths to the video, optical flow, and audio files
        video_path = os.path.join(self.path, 'video', filename)
        flow_path = os.path.join(self.path, 'optical_flow', filename)
        audio_path = os.path.join(self.path, 'audio', filename)
        
        # Load data from disk one sample at a time (Lazy loading)
        video_data = np.load(video_path).astype(np.float32)
        flow_data = np.load(flow_path).astype(np.float32)
        audio_data = np.load(audio_path).astype(np.float32)
        
        # Process video data (for example, slice to a fixed length)
        video_data = video_data[:, :1024].astype(np.float32)  # Ensure proper data type and slicing
        
        # Process flow data (for example, prepend 0 as a fix)
        flow_data = np.concatenate(([0], flow_data)).astype(np.float32)  # Dirty fix for first flow feature
        
        # Convert to torch tensors
        video_data = torch.tensor(video_data)
        flow_data = torch.tensor(flow_data)
        audio_data = torch.tensor(audio_data)
        
        return video_data, flow_data, audio_data
