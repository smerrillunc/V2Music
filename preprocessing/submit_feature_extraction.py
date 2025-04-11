import os
import numpy as np
import itertools


# SLURM job template
slurm_template = """#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 16g
#SBATCH -n 1
#SBATCH -t 02-23:59:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=smerrill@unc.edu
#SBATCH --output=/proj/mcavoy_lab/logs/job_%j.out 
#SBATCH --error=/proj/mcavoy_lab/logs/job_%j.err

module purge
module load python/3.11.9
source /proj/mcavoy_lab/V2M/venv/bin/activate
python FeatureExtraction.py --start_index {start_index} --end_index {end_index} --path /proj/mcavoy_lab/V2M
"""

parameters_list = []


for start_index in np.arange(0, 200000, 1000):
    end_index = start_index + 1000
    tmp = {'start_index':start_index,
           'end_index':end_index}
    parameters_list.append(tmp)


# Submit SLURM jobs
for idx, parameters in enumerate(parameters_list):
    job_script = f"{idx}.sh"
    with open(job_script, "w") as file:
        file.write(slurm_template.format(**parameters))
    os.system(f"sbatch {job_script}")

print("All jobs submitted.")