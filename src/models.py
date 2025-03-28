import torch
import torch.nn as nn
import math
import IPython.display as ipd


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # pe: Shape = (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # position: Shape = (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # div_term: Shape = (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)  # Shape = (max_len, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term)  # Shape = (max_len, d_model/2)
        self.register_buffer('pe', pe.unsqueeze(0))  # pe: Shape = (1, max_len, d_model)
    
    def forward(self, x):
        # x: Shape = (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]  # Output: Shape = (batch_size, seq_len, d_model)

    

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

    
class SlidingWindowMusicPredictor(nn.Module):
    def __init__(self,
                 vid_input_dim,
                 window_size,         # Length of the sliding window Ls
                 overlap,             # Overlap between windows O
                 hidden_dim,          # Hidden dimension for the transformer
                 num_heads,           # Number of attention heads for the Transformer
                 num_layers):         # Number of layers in the Transformer
        super(SlidingWindowMusicPredictor, self).__init__()
        
        self.window_size = window_size
        self.overlap = overlap
        
        # Transformer Encoder for long-term dependencies
        self.long_term_model = TransformerEncoder(input_dim=1024, hidden_dim=hidden_dim, 
                                                  num_heads=num_heads, num_layers=num_layers)
                
    def forward(self, visual_features):
        """
        visual_features shape: [batch_size, seq_len, feature_dim]
        - batch_size: Number of videos in a batch.
        - seq_len: Number of frames in the video (sequence length).
        - feature_dim: Number of features for each frame (e.g., output dimension of the visual encoder).
        """
        batch_size, seq_len, feature_dim = visual_features.shape
        
        # Initialize t (window start position)
        t = 0
        predictions = []
        
        while t + self.window_size <= seq_len:
            # Step 2: Extract features within the current sliding window [t, t + Ls]
            window_features = visual_features[:, t:t + self.window_size, :]
            
            # Step 3: Capture long-term dependencies using the Transformer Encoder
            # Pass the window features through the long-term Transformer encoder model
            transformer_output = self.long_term_model(window_features)  # Shape: [batch_size, window_size, hidden_dim]
            
            # Optionally, use the last output of the transformer (or the average)
            # To summarize the window output into a single vector, use the last token or pooling
            predictions.append(transformer_output[:, -1, :])  # Using the last token in the window
        
            # Move the window forward by setting t = t + Ls - O
            t = t + self.window_size - self.overlap
        
        # Concatenate all predictions and take the mean or last token
        # The output will have shape [batch_size, hidden_dim]
        all_predictions = torch.stack(predictions, dim=1)  # Shape: [batch_size, num_windows, hidden_dim]
        
        # Optionally, aggregate across all windows (mean, max, or just the last one)
        # Here, we will take the mean across all windows as an example
        final_representation = all_predictions.mean(dim=1)  # Shape: [batch_size, hidden_dim]
        
        return final_representation


class CrossAttentionFusion(nn.Module):
    def __init__(self, vid_input_dim, hidden_dim, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, short_term_output, long_term_output):
        # short_term_output: Shape = (batch_size, hidden_dim)
        # long_term_output: Shape = (batch_size, hidden_dim)
        attn_output, _ = self.attention(short_term_output.unsqueeze(1), long_term_output.unsqueeze(1), long_term_output.unsqueeze(1))
        # attn_output: Shape = (batch_size, 1, hidden_dim)
        fused_output = self.fc(attn_output.squeeze(1))  # fused_output: Shape = (batch_size, hidden_dim)
        return fused_output


class MusicDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2, num_heads=8, hidden_dim=512):
        super(MusicDecoder, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        # x: Shape = (batch_size, music_seq_len, input_dim)
        x = x.permute(1, 0, 2)  # Shape = (music_seq_len, batch_size, input_dim) [for transformer]
        decoded_output = self.transformer_decoder(x, x)  # decoded_output: Shape = (music_seq_len, batch_size, input_dim)
        output_tokens = self.fc(decoded_output.permute(1, 0, 2))  # output_tokens: Shape = (batch_size, music_seq_len, output_dim)
        return output_tokens


class AutoregressiveModel(nn.Module):
    def __init__(self, vid_input_dim=1024, music_seq_len=1500, codebook_vocab=512, hidden_dim=512, num_layers=2, short_window=1, long_window=5, num_heads=8):
        super(AutoregressiveModel, self).__init__()
        self.short_term_branch = SlidingWindowMusicPredictor(vid_input_dim, short_window, overlap=0, \
                 hidden_dim=hidden_dim, num_heads=num_heads,num_layers=num_layers)
        
        self.long_term_branch = SlidingWindowMusicPredictor(vid_input_dim, long_window, overlap=0, hidden_dim=hidden_dim, num_heads=num_heads,num_layers=num_layers)
        self.fusion = CrossAttentionFusion(hidden_dim, hidden_dim, num_heads)
        self.music_decoder = MusicDecoder(hidden_dim, codebook_vocab, num_layers, num_heads, hidden_dim)
        self.music_seq_len = music_seq_len
    
    def forward(self, x):
        # x: Shape = (batch_size, seq_len, vid_input_dim)
        short_term_out = self.short_term_branch(x)  # short_term_out: Shape = (batch_size, hidden_dim)
        long_term_out = self.long_term_branch(x)  # long_term_out: Shape = (batch_size, hidden_dim)
        fused_output = self.fusion(short_term_out, long_term_out)  # fused_output: Shape = (batch_size, hidden_dim)
        fused_output = fused_output.unsqueeze(1).repeat(1, self.music_seq_len, 1)  # Shape = (batch_size, music_seq_len, hidden_dim)
        music_output = self.music_decoder(fused_output)  # music_output: Shape = (batch_size, music_seq_len, codebook_vocab)
        return music_output
