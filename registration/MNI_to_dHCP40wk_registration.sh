#!/bin/bash

## This file calculates the warp to transform from adult MNI152 space, to the dHCP 40 week template space
## The warp is then applied to the Schaefer 400 atlas to transform it into the dHCP 40 week template space
## The warp is then applied to the A424 atlas to transform it into the dHCP 40 week template space

## Output files in ${BASEDIR}/registration/:
## MNI_to_dHCP_1Warp.nii.gz, MNI_to_dHCP_0GenericAffine.mat, MNI_to_dHCP_1InverseWarp.nii.gz. MNI_to_dHCP_InverseWarped.nii.gz are the resulting transforms.

BASEDIR=/home/clionaodoherty/clara_fyp

# adult MNI without cerebellum
ADULTPTH=${BASEDIR}/templates/MNI152_T1_1mm_brain_nocerebellum.nii.gz

# infant dhcp 40 week without cerebellum and dura
INFANTPTH=${BASEDIR}/templates/dHCP40wk_template_t1_nodura_nocerebellum_rc.nii.gz

antsRegistrationSyN.sh \
  -d 3 \
  -f $INFANTPTH \
  -m $ADULTPTH \
  -o ${BASEDIR}/registration/MNI_to_dHCP_

# Schaefer 400 atlas
SCHAEFERATLAS=${BASEDIR}/atlases/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz

antsApplyTransforms \
  -d 3 \
  -i $SCHAEFERATLAS \
  -r $INFANTPTH \
  -o ${BASEDIR}/atlases/Schaefer_in_dHCP40wk_commonspace.nii.gz \
  -t ${BASEDIR}/registration/MNI_to_dHCP_1Warp.nii.gz \
  -t ${BASEDIR}/registration/MNI_to_dHCP_0GenericAffine.mat \
  --interpolation NearestNeighbor

# Schaefer 400 atlas
A424ATLAS=${BASEDIR}/atlases/A424_MNI_nocerebellum.nii.gz

antsApplyTransforms \
  -d 3 \
  -i $A424ATLAS \
  -r $INFANTPTH \
  -o ${BASEDIR}/atlases/A424_in_dHCP40wk_commonspace.nii.gz \
  -t ${BASEDIR}/registration/MNI_to_dHCP_1Warp.nii.gz \
  -t ${BASEDIR}/registration/MNI_to_dHCP_0GenericAffine.mat \
  --interpolation NearestNeighbor