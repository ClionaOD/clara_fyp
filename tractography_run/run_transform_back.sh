#!/bin/bash
DWIPTH=/dhcp/dhcp_dmri_pipeline
SUBJ=$1
SES=$2

# Name done flag and remove it
DONEFILE=${DWIPTH}/${SUBJ}/${SES}/probtrackx2_clara/done_transform_back
rm $DONEFILE

# Get rid of any previously transformed files
rm ${DWIPTH}/${SUBJ}/${SES}/probtrackx2_clara/seeds_to_*40wk.nii.gz

# Transform s2t to 40wk space
# TODO: change wildcard to be with our filename
for file in ${DWIPTH}/${SUBJ}/${SES}/probtrackx2_clara/seeds_to_*
do
   S2T=$(basename -- "$file")
   echo $S2T
   # TODO: change -r to be in ./templates/template_t1.nii.gz
   applywarp -i ${file} -o ${file}_40wk -r /dhcp/rhodri_registration/atlases/dhcp_volume_40weeks/template_t1 -w ${DWIPTH}/${SUBJ}/${SES}/xfm/${SUBJ}_${SES}_from-dwi_to-template40wk_mode-image.nii.gz --interp=trilinear
done


# Run find the biggest
# find_the_biggest ${DWIPTH}/${SUBJ}/${SES}/probtrackx2/seeds_to_*40wk.nii.gz ${DWIPTH}/${SUBJ}/${SES}/probtrackx2/biggest_40wks.nii.gz
# This was killed, presumably by SLURM as using too much memory on c5.9xlarge worker nodes 

touch $DONEFILE
