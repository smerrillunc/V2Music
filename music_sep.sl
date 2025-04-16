#!/bin/bash

#SBATCH --job-name=music_sep_job           # Job name
#SBATCH --array=0-5                     # Array job indices (0 to 4); 5 tasks in total
#SBATCH -N 1                            # Number of nodes
#SBATCH -n 4                            # Number of tasks (CPU cores)
#SBATCH --mem=32G                       # Memory per node
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --time=10:00:00                 # Total runtime (hh:mm:ss)
#SBATCH -p a100-gpu,l40-gpu             # List of possible GPUs
#SBATCH --qos=gpu_access                # Required directive
#SBATCH --mail-type=ALL                 # Email on job events
#SBATCH --mail-user=tis@cs.unc.edu
#SBATCH --output=/work/users/t/i/tis/V2Music/out/music_sep_job_%A_%a.out   # STDOUT file
#SBATCH --error=/work/users/t/i/tis/V2Music/err/music_sep_job_%A_%a.err    # STDERR file

# Source bash configuration (if any aliases or other settings are needed)
source ~/.bashrc

# Debug: display ffmpeg path and version info
echo "FFMPEG located at: $(which ffmpeg)"
echo "$(ffmpeg -version)"

# Change to the project directory and activate the virtual environment
cd /work/users/t/i/tis/demucs
source .venv/bin/activate

# Total number of items to process
total=19169

# Determine the total number of tasks; if SLURM_ARRAY_TASK_COUNT is not set, default to 5
ntasks=${SLURM_ARRAY_TASK_COUNT:-5}

# Calculate the uniform block size using integer division
block_size=$(( total / ntasks ))
remainder=$(( total % ntasks ))

# Calculate the start index for this job in the array
start=$(( SLURM_ARRAY_TASK_ID * block_size ))

# For the last task, add any remaining items; for others, just add the block size
if [ "$SLURM_ARRAY_TASK_ID" -eq $(( ntasks - 1 )) ]; then
    end=$total
else
    end=$(( start + block_size ))
fi

echo "Processing indices from ${start} to ${end}"

# Run the vocal separation command with the calculated indices
uv run vocal_separation.py --start "${start}" --end "${end}"