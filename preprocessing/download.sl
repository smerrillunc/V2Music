#!/bin/bash
#SBATCH --job-name=dataset_builder    # Job name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # One task per array job (ensures one CPU per job)
#SBATCH --cpus-per-task=1             # Request one CPU per array job
#SBATCH -p datamover                  # Partition name
##SBATCH --nodelist=rc-dm1
#SBATCH --mem=32G                     # Memory per node
#SBATCH --time=10:00:00               # Total runtime (hh:mm:ss)
#SBATCH --mail-type=ALL               # Email notifications for job events
#SBATCH --mail-user=tis@cs.unc.edu
#SBATCH --array=0-9%10                   # Change 0-9 as needed
#SBATCH --output=/work/users/t/i/tis/V2Music/out/download_job_%A_%a.out   # STDOUT file
#SBATCH --error=/work/users/t/i/tis/V2Music/err/download_job_%A_%a.err    # STDERR file

# Create the output directories if they do not exist
mkdir -p /work/users/t/i/tis/V2Music/out
mkdir -p /work/users/t/i/tis/V2Music/err

 
# Source the bash configuration to load custom aliases (if any)
source ~/.bashrc

# Display the path of ffmpeg (for debugging purposes)
echo "FFMPEG located at: $(which ffmpeg)"
echo "$(ffmpeg -version)"

# Change to the project folder
cd /work/users/t/i/tis/V2Music

# Activate your virtual environment (or source your environment)
source .venv/bin/activate

# Move into the preprocessing directory
cd preprocessing

# Define parameters
START=0                   # Starting index for the first task
BLOCK_SIZE=500                 # Block size for start/end indices

# Calculate the starting and ending indices for this array task
CURRENT_START=$(( START + SLURM_ARRAY_TASK_ID * BLOCK_SIZE ))
CURRENT_END=$(( CURRENT_START + BLOCK_SIZE ))

# Log the calculated indices (output will be written to the specified STDOUT file)
echo "Running download.py with --start ${CURRENT_START} --end ${CURRENT_END}"

#sleep 3600
# Run the download command with the computed indices
uv run download.py --start "${CURRENT_START}" --end "${CURRENT_END}"
