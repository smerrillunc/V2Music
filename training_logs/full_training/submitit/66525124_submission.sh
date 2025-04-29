#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH --job-name=audiocraft_6420c4e9
#SBATCH --mem=512GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --output=/work/users/t/i/tis/VidMuse/output/VidMuse/xps/6420c4e9/submitit/%j_0_log.out
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --signal=USR2@90
#SBATCH --time=10080
#SBATCH --wckey=submitit

# setup
source ~/.bashrc
echo "FFMPEG located at $(which ffmpeg)"
echo "$(ffmpeg -version)"
cd /work/users/t/i/tis/VidMuse
source .venv/bin/activate

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /work/users/t/i/tis/VidMuse/output/VidMuse/xps/6420c4e9/submitit/%j_%t_log.out /work/users/t/i/tis/VidMuse/.venv/bin/python3 -u -m submitit.core._submit /work/users/t/i/tis/VidMuse/output/VidMuse/xps/6420c4e9/submitit
