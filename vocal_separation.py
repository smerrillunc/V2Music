#!/usr/bin/env python

#IMPORTANT: needs to be placed into demucs repo to work 

import os
import sys
import argparse
from pathlib import Path
import demucs.separate

def main():
    parser = argparse.ArgumentParser(
        description="Process a subset of audio files using Demucs separation."
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Starting index (inclusive) of the audio files to process."
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="Ending index (exclusive) of the audio files to process. Defaults to the total number of files."
    )
    parser.add_argument(
        "--audio_dir", type=str,
        default="/work/users/t/i/tis/V2Music/preprocessing/data/audio",
        help="Path to the directory containing audio files (m4a and webm)."
    )
    parser.add_argument(
        "--out_path", type=str,
        default="/work/users/t/i/tis/V2Music/preprocessing/bg_audio",
        help="Output directory for the separated stems."
    )
    parser.add_argument(
        "--model", "-n", type=str,
        default="mdx_extra",
        help="Model name to use for separation (e.g., mdx_extra)."
    )
    parser.add_argument(
        "--filename_template", type=str,
        default="{track}_{stem}.{ext}",
        help="Template for output file names. Available placeholders: {track}, {stem}, {ext}."
    )

    args = parser.parse_args()

    # Create a Path object for the audio directory
    audio_dir = Path(args.audio_dir)
    out_path = Path(args.out_path)
    # Create output directory if it does not exist
    out_path.mkdir(parents=True, exist_ok=True)

    # Get all .m4a and .webm files in the directory and sort them for consistency
    audio_files = sorted(list(audio_dir.glob("*.m4a")) + list(audio_dir.glob("*.webm")))
    total_files = len(audio_files)
    
    if total_files == 0:
        print(f"No audio files found in {audio_dir}")
        sys.exit(1)
    
    # Set default end value if not provided
    if args.end is None or args.end > total_files:
        args.end = total_files
    
    # Slice out the subset based on start and end indices
    subset_files = audio_files[args.start:args.end]
    
    if not subset_files:
        print(f"No files in the range from {args.start} to {args.end}.")
        sys.exit(1)
    
    print(f"Processing files {args.start} to {args.end} (total {total_files}).")
    
    # Build the Demucs command line arguments
    demucs_args = [
        "--two-stems", "vocals",
        # Uncomment the following line if you want to use the minus method for creating no_vocals
        # "--other-method", "minus",
        "-n", args.model,
        "--filename", args.filename_template,
        "-o", args.out_path,
    ]
    # Append the paths (converted to strings) for the subset of audio files.
    demucs_args.extend(map(str, subset_files))
    
    # Call the Demucs separation function with our arguments.
    demucs.separate.main(demucs_args)

if __name__ == "__main__":
    main()