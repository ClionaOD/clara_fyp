#!/bin/bash

SUBJ=$1
SES=$2

source /foundcog/pyenv3.8/bin/activate
python /home/claraconyngham/clara_fyp/get_probtrackx_VVC_data.py \
    --sub $SUBJ \
    --ses $SES