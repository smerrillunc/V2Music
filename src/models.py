import torch
import torch.nn as nn
import math

    

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(TransformerEncoder, self).__init__()
        
        # Define the transformer encoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # Linear projection to map input features to transformer input size (hidden_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x):
        """
        x: Input shape [batch_size, window_size, input_dim]
        The shape needs to be (batch_size, seq_len, feature_dim) for transformer encoder.
        """
        # Project input features to hidden_dim
        x = self.input_projection(x)  # Shape: [batch_size, window_size, hidden_dim]
        
        # Pass through the transformer encoder (transposed to match shape requirements)
        x = x.permute(1, 0, 2)  # Shape: [window_size, batch_size, hidden_dim]
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Revert to [batch_size, window_size, hidden_dim]
        
        return x

class SegmentedFeatureExtractor(nn.Module):
    def __init__(self, input_dim, num_segments, hidden_dim, num_heads, num_layers, music_seq_len):
        super(SegmentedFeatureExtractor, self).__init__()

        self.num_segments = num_segments
        self.music_seq_len = music_seq_len

        # Initialize the transformer encoder to process each segment
        self.segment_transformer = TransformerEncoder(input_dim=input_dim, hidden_dim=hidden_dim,
                                                     num_heads=num_heads, num_layers=num_layers)

        # Projection layer to match the desired output shape for the decoder
        # This layer projects the concatenated segment embeddings to a suitable hidden_dim
        self.projection = nn.Linear(hidden_dim * num_segments, hidden_dim * music_seq_len)

    def forward(self, features):
        """
        features: Shape = [batch_size, seq_len, input_dim]
        """
        batch_size, seq_len, input_dim = features.shape
        segment_size = seq_len // self.num_segments

        segment_embeddings = []

        for s in range(self.num_segments):
            # Step 1: Extract the features for the current segment
            start_idx = s * segment_size
            end_idx = (s + 1) * segment_size if s < self.num_segments - 1 else seq_len
            segment_features = features[:, start_idx:end_idx, :]

            # Step 2: Pass the segment through the transformer encoder
            segment_output = self.segment_transformer(segment_features)  # Shape: [batch_size, segment_size, hidden_dim]

            # Step 3: Aggregate features in the segment (mean pooling)
            segment_representation = segment_output.mean(dim=1)  # Shape: [batch_size, hidden_dim]
            segment_embeddings.append(segment_representation)

        # Step 4: Concatenate all segment embeddings along the sequence axis (dim=1)
        segment_embeddings = torch.cat(segment_embeddings, dim=1)  # Shape: [batch_size, num_segments * hidden_dim]

        # Step 5: Apply projection to match the target sequence length
        # This projection will make the final shape [batch_size, music_seq_len * hidden_dim]
        segment_embeddings = self.projection(segment_embeddings)  # Shape: [batch_size, music_seq_len * hidden_dim]

        # Step 6: Reshape to match [batch_size, music_seq_len, hidden_dim]
        segment_embeddings = segment_embeddings.view(batch_size, self.music_seq_len, -1)  # Shape: [batch_size, music_seq_len, hidden_dim]

        return segment_embeddings
    

class MusicDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2, num_heads=8, hidden_dim=512):
        super(MusicDecoder, self).__init__()
        
        # Initialize the transformer decoder
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        
        # Linear layer to map the decoder output to codebook_vocab size
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x, memory):
        """
        x: Shape = [batch_size, music_seq_len, input_dim] (target sequence for the decoder)
        memory: Shape = [batch_size, vid_seq_len, input_dim] (encoder output, for attention)
        
        x - the sequence of tokens for the decoder to generate predictions from.
        memory - the sequence of video features that the decoder attends to.

        Output: Shape = [batch_size, music_seq_len, codebook_vocab]
        """
        
        # Permute the inputs to match the shape expected by the transformer decoder
        x = x.permute(1, 0, 2)  # Shape = [music_seq_len, batch_size, input_dim] (target sequence)
        memory = memory.permute(1, 0, 2)  # Shape = [vid_seq_len, batch_size, input_dim] (encoder output)
        
        # Apply the transformer decoder
        decoded_output = self.transformer_decoder(x, memory)  # Shape: [music_seq_len, batch_size, input_dim]
        
        # Project the output of the transformer decoder to the codebook_vocab space
        output_tokens = self.fc(decoded_output.permute(1, 0, 2))  # Shape: [batch_size, music_seq_len, output_dim]
        
        return output_tokens
    

class AutoregressiveModelSegmented(nn.Module):
    def __init__(self, vid_input_dim=1024,  music_seq_len=1500, codebook_vocab=512, hidden_dim=512, num_layers=2, num_segments=10, num_heads=8):
        super(AutoregressiveModelSegmented, self).__init__()
        self.video_feature_branch = SegmentedFeatureExtractor(vid_input_dim, num_segments, hidden_dim, num_heads, num_layers, music_seq_len)
        self.music_decoder = MusicDecoder(hidden_dim, codebook_vocab, num_layers, num_heads, hidden_dim)
        self.music_seq_len = music_seq_len
    
    def forward(self, vid_feats, flow_feats):
        # dirty fix.  First flow feature should be zero
        x = (flow_feats.unsqueeze(-1) + vid_feats)
        
        # vid_feats: Shape = (batch_size, seq_len, vid_input_dim)
        video_feature_out = self.video_feature_branch(x)  # Shape = [batch_size, seq_len, hidden_dim]
        music_output = self.music_decoder(video_feature_out, video_feature_out)  # music_output: Shape = (batch_size, music_seq_len, codebook_vocab)
        return music_output
