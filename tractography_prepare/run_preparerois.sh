#!/bin/bash

# This file runs one subject's seed and target preparation for probtrackx2_clara
# Change below two lines to set test subject

#sub-CC00124XX09
#ses-42302

DWIPTH=/dhcp/dhcp_dmri_pipeline
SUBJ=$1
SES=$2

BASEDIR=/home/clionaodoherty/clara_fyp

filename=$(basename -- "$0")
extension="${filename##*.}"
filename="${filename%.*}"
DONEFILE=${DWIPTH}/${SUBJ}/${SES}/probtrackx2_clara/done_${filename}

rm $DONEFILE

mkdir ${DWIPTH}/${SUBJ}/${SES}/probtrackx2_clara

# this should be where seeds and targets are
ROIPTH=${BASEDIR}/atlases

# Seed and target from 40wks to dwi space
## TODO: Make sure it's probtrackx2_clara
## Keep -r and -w the same
## TODO: Change input (-i) for seed to be your Schaefer VVC regions and output (-o) to be an appropriate filename
applywarp -i ${ROIPTH}/dhcp40w_cat_vvc_seeds_nodura.nii.gz -o ${DWIPTH}/${SUBJ}/${SES}/probtrackx2_clara/${SUBJ}_${SES}_fcg_category_seed_regions_hcp40w_dwi.nii.gz -r ${DWIPTH}/${SUBJ}/${SES}/dwi/${SUBJ}_${SES}_desc-preproc_space-dwi_brainmask.nii.gz -w ${DWIPTH}/${SUBJ}/${SES}/xfm/${SUBJ}_${SES}_from-template40wk_to-dwi_mode-image.nii.gz --interp=nn
## TODO: Change input for seed to be your Schaefer VVC regions and output to be an appropriate filename
applywarp -i ${ROIPTH}/hcp_regions_v4_target_40wks.nii.gz -o ${DWIPTH}/${SUBJ}/${SES}/probtrackx2_clara/${SUBJ}_${SES}_hcp_regions_v4_target_dwi.nii.gz -r ${DWIPTH}/${SUBJ}/${SES}/dwi/${SUBJ}_${SES}_desc-preproc_space-dwi_brainmask.nii.gz -w ${DWIPTH}/${SUBJ}/${SES}/xfm/${SUBJ}_${SES}_from-template40wk_to-dwi_mode-image.nii.gz --interp=nn

# Split target
## TODO: Change filename, match to above structure that you used for the target
TOSPLIT=${DWIPTH}/${SUBJ}/${SES}/probtrackx2_clara/${SUBJ}_${SES}_hcp_regions_v4_target_dwi
TEXTOUT=${TOSPLIT}_list.txt
rm $TEXTOUT
CWD=`pwd`
## TODO: change from 370, 0, 370 to the number of regions in our (aka how many unique indices in the A424 atlas)
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
