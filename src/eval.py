import torch
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import peak_prominences

class Eval:
    def __init__(self):
        pass

    def compute_mean_and_covariance(self, embeddings):
        """
        Computes the mean and covariance matrix for a batch of embeddings.
        
        Args:
            embeddings (torch.Tensor): Tensor of shape (batch_size, feature_dim)
        
        Returns:
            mean (torch.Tensor): Mean vector of shape (batch_size, feature_dim)
            covariance (torch.Tensor): Covariance matrices of shape (batch_size, feature_dim, feature_dim)
        """
        batch_size, feature_dim = embeddings.shape
        mean = embeddings.mean(dim=1, keepdim=True)  # (batch_size, 1, feature_dim)
        centered = embeddings - mean  # (batch_size, feature_dim)
        
        covariance = torch.matmul(centered.unsqueeze(2), centered.unsqueeze(1)).mean(dim=0)  # (feature_dim, feature_dim)
        
        return mean.squeeze(1), covariance

    def calculate_frechet_audio_distance(self, embeddings_ground_truth, embeddings_retrieved, epsilon=1e-6):
        """
        Computes the Fréchet Audio Distance (FAD) for a batch of examples.
        
        Args:
            embeddings_ground_truth (torch.Tensor): Ground truth embeddings (batch_size, feature_dim)
            embeddings_retrieved (torch.Tensor): Retrieved embeddings (batch_size, feature_dim)
            epsilon (float): Small constant to ensure numerical stability.
        
        Returns:
            torch.Tensor: FAD scores for each batch element (batch_size,)
        """
        mu_x, sigma_x = self.compute_mean_and_covariance(embeddings_ground_truth)
        mu_y, sigma_y = self.compute_mean_and_covariance(embeddings_retrieved)
        
        # Regularize covariance matrices by adding epsilon to the diagonal
        identity = torch.eye(sigma_x.shape[-1], device=sigma_x.device) * epsilon
        sigma_x += identity
        sigma_y += identity
        
        # Compute squared difference of means
        mean_diff = (mu_x - mu_y).pow(2).sum(dim=-1)  # (batch_size,)

        # Eigen decomposition for covariance matrices
        try:
            eigvals_x, eigvecs_x = torch.linalg.eigh(sigma_x)
            eigvals_y, eigvecs_y = torch.linalg.eigh(sigma_y)
        except RuntimeError as e:
            print(f"Error in eigen decomposition: {e}")
            return torch.tensor([float('nan')] * embeddings_ground_truth.shape[0], device=sigma_x.device)

        # Ensure eigenvalues are positive (by clamping small negative values)
        eigvals_x = torch.clamp(eigvals_x, min=epsilon)
        eigvals_y = torch.clamp(eigvals_y, min=epsilon)

        # Compute square roots of covariance matrices using eigenvalue decomposition
        sigma_x_sqrt = eigvecs_x @ torch.diag_embed(torch.sqrt(eigvals_x)) @ eigvecs_x.transpose(-2, -1)
        sigma_y_sqrt = eigvecs_y @ torch.diag_embed(torch.sqrt(eigvals_y)) @ eigvecs_y.transpose(-2, -1)

        # Compute sqrt of product of covariance matrices
        sqrt_product = sigma_x_sqrt @ sigma_y_sqrt
        
        # Compute trace term for FAD
        trace_term = torch.trace(sigma_x + sigma_y - 2 * sqrt_product)
        
        # Compute final FAD score
        fad_scores = mean_diff + trace_term  # (batch_size,)
        return fad_scores
    
    # Function to detect local maxima in embeddings using scipy's find_peaks
    def find_local_maxima_in_embeddings(self, embeddings, prominence_threshold=0.1):
        """
        Detect local maxima (peaks) in the embeddings.

        Args:
            embeddings (torch.Tensor): Tensor of shape (batch_size, feature_dim).
            prominence_threshold (float): Minimum prominence required to consider a peak.

        Returns:
            peaks_list (list): List of indices where the local maxima (peaks) occur in the embeddings.
        """
        embeddings = embeddings.cpu().detach().numpy()  # Convert to numpy for peak detection
        peaks_list = []
        
        for i in range(embeddings.shape[0]):  # Iterate through each embedding (e.g., audio or video)
            # Find local maxima (peaks) in the embedding
            peaks, _ = find_peaks(embeddings[i])
            prominences = peak_prominences(embeddings[i], peaks)[0]

            # Filter peaks based on prominence
            significant_peaks = peaks[prominences >= prominence_threshold]
            peaks_list.append(significant_peaks)
        
        return peaks_list

    # Function to calculate Intersection over Union (IoU) between audio and video peaks
    def calc_intersection_over_union(self, audio_peaks, video_peaks):
        """
        Calculate Intersection over Union (IoU) between the audio and video peaks.

        Args:
            audio_peaks (list): Indices of audio peaks.
            video_peaks (list): Indices of video peaks.

        Returns:
            float: IoU score between audio and video peaks.
        """
        intersection = len(set(audio_peaks).intersection(set(video_peaks)))
        union = len(set(audio_peaks).union(set(video_peaks)))
        iou_score = intersection / union
        return iou_score

    # Function to compute the AV-Align score from audio and video embeddings
    def compute_av_align_score(self, audio_embeddings, video_embeddings, prominence_threshold=0.1):
        """
        Compute the AV-Align score between the audio and video embeddings.

        Args:
            audio_embeddings (torch.Tensor): Audio embeddings of shape (batch_size, feature_dim).
            video_embeddings (torch.Tensor): Video embeddings of shape (batch_size, feature_dim).
            prominence_threshold (float): Minimum prominence to filter significant peaks.

        Returns:
            float: AV-Align score (IoU between audio and video peaks).
        """
        # Detect peaks in the audio and video embeddings
        audio_peaks = self.find_local_maxima_in_embeddings(audio_embeddings, prominence_threshold)
        video_peaks = self.find_local_maxima_in_embeddings(video_embeddings, prominence_threshold)

        # Calculate the Intersection over Union (IoU) for the audio and video peaks
        iou_scores = []
        for audio, video in zip(audio_peaks, video_peaks):
            iou_score = self.calc_intersection_over_union(audio, video)
            iou_scores.append(iou_score)

        # Compute the mean IoU score across the batch
        mean_iou_score = np.mean(iou_scores)
        return mean_iou_score
