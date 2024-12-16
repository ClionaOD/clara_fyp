#!/bin/bash
DWIPTH=/dhcp/dhcp_dmri_pipeline
SUBJ=$1
SES=$2
ROIPTH=/dhcp/rhodri_registration/rois

filename=$(basename -- "$0")
extension="${filename##*.}"
filename="${filename%.*}"
DONEFILE=${DWIPTH}/${SUBJ}/${SES}/probtrackx2/done_${filename}

rm $DONEFILE

mkdir ${DWIPTH}/${SUBJ}/${SES}/probtrackx2

# Seed and target from 40wks to dwi space
applywarp -i ${ROIPTH}/hcp_regions_v4_seed_40wks.nii.gz -o ${DWIPTH}/${SUBJ}/${SES}/probtrackx2/${SUBJ}_${SES}_hcp_regions_v4_seed_dwi.nii.gz -r ${DWIPTH}/${SUBJ}/${SES}/dwi/${SUBJ}_${SES}_desc-preproc_space-dwi_brainmask.nii.gz -w ${DWIPTH}/${SUBJ}/${SES}/xfm/${SUBJ}_${SES}_from-template40wk_to-dwi_mode-image.nii.gz --interp=nn
applywarp -i ${ROIPTH}/hcp_regions_v4_target_40wks.nii.gz -o ${DWIPTH}/${SUBJ}/${SES}/probtrackx2/${SUBJ}_${SES}_hcp_regions_v4_target_dwi.nii.gz -r ${DWIPTH}/${SUBJ}/${SES}/dwi/${SUBJ}_${SES}_desc-preproc_space-dwi_brainmask.nii.gz -w ${DWIPTH}/${SUBJ}/${SES}/xfm/${SUBJ}_${SES}_from-template40wk_to-dwi_mode-image.nii.gz --interp=nn

# Split target
TOSPLIT=${DWIPTH}/${SUBJ}/${SES}/probtrackx2/${SUBJ}_${SES}_hcp_regions_v4_target_dwi
TEXTOUT=${TOSPLIT}_list.txt
rm $TEXTOUT
CWD=`pwd`
COUNTS=`fslstats $TOSPLIT -H 370 0 370` # COUNT VOXELS IN EACH REGION
IND=0
for X in $COUNTS
do
   if [ $X != "0.000000" ]; then	# ONLY INCLUDE REGIONS WITH VOXELS
	echo "Found ROI $IND with $X voxels"
	fslmaths $TOSPLIT -thr $IND -uthr $IND ${TOSPLIT}_${IND}	
	echo ${TOSPLIT}_${IND} >> ${TEXTOUT}
   fi
   IND=$(($IND+1))
done

touch $DONEFILE
