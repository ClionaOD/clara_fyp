#!/bin/bash
#SBATCH --job-name=ml_regression_%j
#SBATCH --output=ml_regression_%j.log

BASEDIR=/home/claraconyngham/clara_fyp

source /foundcog/pyenv3.8/bin/activate
python ${BASEDIR}/regression.py \
    --base_dir ${BASEDIR} \
    --category cat \
