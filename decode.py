#!/usr/bin/env python3

# Also works for single ids like this uv run decode.py -- "-1vflapLfr8" "-2AitZq1AkA"

import os
import glob
import torch
import soundfile as sf
import numpy as np
import argparse
from transformers import EncodecModel, AutoProcessor

def load_file(file_path, device):
    """Load audio codes from a .npz or .pt file."""
    if file_path.endswith('.npz'):
        npz_data = np.load(file_path)
        return {
            "audio_codes": torch.from_numpy(npz_data["audio_codes"]).to(device),
            "audio_scales": torch.from_numpy(npz_data["audio_scales"]).to(device),
            "padding_mask": torch.from_numpy(npz_data["padding_mask"]).to(device)
        }
    elif file_path.endswith('.pt'):
        return torch.load(file_path, map_location=device)
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Supported formats are .npz and .pt.")

def load_audio_features(file_paths, device='cuda'):
    """Load and batch audio features from a list of file paths.

    The batch dimension size is determined by the number of file paths provided.
    """
    if not file_paths:
        raise ValueError("The file_paths list is empty.")
    
    # Renaming the variable 'B' to 'batch_size'
    batch_size = len(file_paths)
    
    # Load the first file to determine the shapes.
    first_data = load_file(file_paths[0], device)
    codes_tensor = torch.empty((batch_size,) + first_data["audio_codes"].shape,
                               dtype=first_data["audio_codes"].dtype, device=device)
    scales_tensor = torch.empty((batch_size,) + first_data["audio_scales"].shape,
                                dtype=first_data["audio_scales"].dtype, device=device)
    mask_tensor = torch.empty((batch_size,) + first_data["padding_mask"].shape,
                              dtype=first_data["padding_mask"].dtype, device=device)

    # Populate the first slot.
    codes_tensor[0] = first_data["audio_codes"]
    scales_tensor[0] = first_data["audio_scales"]
    mask_tensor[0]  = first_data["padding_mask"]

    print("First file shapes:")
    print("Codes:", codes_tensor[0].shape)
    print("Scales:", scales_tensor[0].shape)
    print("Padding mask:", mask_tensor[0].shape)

    # Load the remaining files.
    for i in range(1, batch_size):
        data = load_file(file_paths[i], device)
        current_codes = data["audio_codes"]
        current_scales = data["audio_scales"]
        current_mask   = data["padding_mask"]
         
        # Ensure consistency with the first loaded file.
        if (current_codes.shape != codes_tensor[0].shape or
            current_scales.shape != scales_tensor[0].shape or
            current_mask.shape  != mask_tensor[0].shape):
            raise ValueError(
                f"Dimension mismatch in file {file_paths[i]}: Expected shapes "
                f"{(codes_tensor[0].shape, scales_tensor[0].shape, mask_tensor[0].shape)} but got "
                f"{(current_codes.shape, current_scales.shape, current_mask.shape)}"
            )
        
        codes_tensor[i] = current_codes
        scales_tensor[i] = current_scales
        mask_tensor[i]   = current_mask

    # Transpose so that the first dimension represents the channels.
    return codes_tensor.transpose(0, 1), scales_tensor.transpose(0, 1), mask_tensor

def main():
    parser = argparse.ArgumentParser(
        description="Decode audio codes from NPZ or PT files using EncodecModel."
    )
    parser.add_argument(
        "--file_format", type=str, default="npz",
        help="Input file format: 'npz' or 'pt'. (default: npz)"
    )
    parser.add_argument(
        "--input_dir", type=str,
        default="/work/users/t/i/tis/V2Music/preprocessing/bg_audio/mdx_extra/encoded_tensors",
        help="Directory containing encoded tensor files. (default: /nas/longleaf/home/tis/wd/V2Music/preprocessing/bg_audio/mdx_extra/encoded_tensors)"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/work/users/t/i/tis/V2Music/preprocessing/bg_audio/mdx_extra/decoded_wav",
        help="Directory to save decoded wav files. (default: /nas/longleaf/home/tis/wd/V2Music/preprocessing/bg_audio/mdx_extra/decoded_wav)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=20,
        help="Batch size for processing files. (default: 20)"
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Start index for processing the sorted list of files. (default: 0)"
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="End index for processing the sorted list of files. (default: process all files after start index)"
    )
    parser.add_argument(
        "ids", nargs="*", default=None,
        help="Optional list of IDs to filter the encoded files. Only files whose filenames contain one of these IDs will be processed."
    )

    args = parser.parse_args()

    # Load the model and processor.
    model = EncodecModel.from_pretrained("facebook/encodec_48khz")
    processor = AutoProcessor.from_pretrained("facebook/encodec_48khz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    target_sr = processor.sampling_rate  

    # Ensure the output directory exists.
    os.makedirs(args.output_dir, exist_ok=True)

    # List and sort encoded files in the input directory.
    encoded_files = sorted(glob.glob(os.path.join(args.input_dir, f"*.{args.file_format}")))
    if not encoded_files:
        print("No encoded tensor files found in:", args.input_dir)
        exit(1)
   
    if args.ids:
        encoded_files = [f for f in encoded_files if any(_id in os.path.basename(f) for _id in args.ids)]
    
    if args.end is None or args.end > len(encoded_files):
        args.end = len(encoded_files)
        
    encoded_files = encoded_files[args.start:args.end]

    print(f"Processing {len(encoded_files)} files (from index {args.start} to {args.end}).")

    # Process the files in batches.
    for start_idx in range(0, len(encoded_files), args.batch_size):
        batch_file_paths = encoded_files[start_idx:start_idx + args.batch_size]
        
        # Load batched features.
        batch_audio_codes, batch_audio_scales, batch_padding_mask = load_audio_features(batch_file_paths, device=device)
        
        with torch.no_grad():
            # Decode the batch of audio codes.
            decoded_batch = model.decode(batch_audio_codes, batch_audio_scales, batch_padding_mask)[0]

        # Move the decoded audio to CPU and convert to numpy.
        decoded_batch = decoded_batch.cpu().numpy()
        # Compute valid lengths for each file in the batch using the padding masks.
        valid_lengths = batch_padding_mask.sum(dim=1).cpu()

        # Process each sample in the current batch.
        for i, file in enumerate(batch_file_paths):
            valid_len = valid_lengths[i]
            # Assume decoded audio shape is (channels, time).
            sample_decoded = decoded_batch[i]
            # Crop the decoded audio to include only valid time steps.
            cropped_audio = sample_decoded[:, :valid_len]
            # Transpose from (channels, time) to (time, channels) for saving.
            output_audio = cropped_audio.T
            output_file = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(file))[0]}.wav")
            sf.write(output_file, output_audio, target_sr)
            print(f"Saved decoded audio to: {output_file}")

if __name__ == '__main__':
    main()