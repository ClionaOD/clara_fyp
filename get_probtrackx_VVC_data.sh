#!/bin/bash

SUBJ=$1
SES=$2
BASEDIR=$3

source /foundcog/pyenv3.8/bin/activate
python ${BASEDIR}/get_probtrackx_VVC_data.py \
    --sub $SUBJ \
    --ses $SES \
    --base_dir $BASEDIR