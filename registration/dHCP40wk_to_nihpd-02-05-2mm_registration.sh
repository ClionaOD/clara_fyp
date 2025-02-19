#!/bin/bash

## This file calculates the warp to transform from twomonth nihpd-asym space, to the dHCP 40 week template space

## Output files in ${BASEDIR}/registration/:
## nihpd-02-05_to_dHCP_1Warp.nii.gz, nihpd-02-05_to_dHCP_0GenericAffine.mat, nihpd-02-05_to_dHCP_1InverseWarp.nii.gz. nihpd-02-05_to_dHCP_InverseWarped.nii.gz are the resulting transforms.

BASEDIR=/home/clionaodoherty/clara_fyp

# twomonth template without cerebellum
TWOMONTHPTH=${BASEDIR}/templates/nihpd-02-05_t1w_fcgmasked_2mm_nocerebellum.nii.gz

# infant dhcp 40 week without cerebellum and dura
NEONATEPTH=${BASEDIR}/templates/dHCP40wk_template_t1_nodura_nocerebellum_rc.nii.gz

antsRegistrationSyN.sh \
  -d 3 \
  -f $TWOMONTHPTH \
  -m $NEONATEPTH \
  -o ${BASEDIR}/registration/dHCP_to_nihpd-02-05-2mm_