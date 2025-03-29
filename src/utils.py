import pandas as pd
import numpy as np
import torch
from eval import Eval



def save_checkpoint(model, optimizer, epoch, filename):
    """Saves model and optimizer state dict."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch} to {filename}")


def compute_evaluations(model, test_loader, epoch):

    metrics = Eval()
    model.eval()

    fads = []
    av_aligns = []

    
    for video_feats, flow_feats, audio_targets in test_loader:
        predictions = model(video_feats, flow_feats)

        #[batch_size, music_seq_len, codebooksize] to convert to codebook values
        audio_embeddings_retrieved = predictions.argmax(dim=-1)


        fad = metrics.calculate_frechet_audio_distance(audio_targets.float(), audio_embeddings_retrieved.float())
        
        # this won't work since video embeddings are of different shape than audio embeddings
        #av_align = metrics.compute_av_align_score(video_feats, audio_embeddings_retrieved, prominence_threshold=0.1)

        fads.append(fad)
        av_aligns.append(av_align)

    print(f'Mean FADs: {np.mean(fads)}')
    
    tmp = pd.DataFrame({'epoch':epoch,
       'fad':np.mean(av_aligns),
       #'av_align':np.mean(fads)
      }, index=[0])

        
    return tmp
