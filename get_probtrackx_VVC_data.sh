#!/bin/bash
# SBATCH --job-name=get_voxels
# SBATCH --output=/home/claraconyngham/clara_fyp/slurm_logs/get_voxels.out

source /foundcog/pyenv3.8/bin/activate
python /home/claraconyngham/clara_fyp/get_probtrackx_VVC_data.py