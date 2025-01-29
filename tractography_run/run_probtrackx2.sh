#!/bin/bash

SUBJ=$1
SES=$2
DWIPTH=/dhcp/dhcp_dmri_pipeline
## TODO: change to paths from preparerois.sh
TOSPLIT=${DWIPTH}/${SUBJ}/${SES}/probtrackx2_clara/${SUBJ}_${SES}_A424_target_regions_dwi
TEXTOUT=${TOSPLIT}_list.txt

DONEFILE=${DWIPTH}/${SUBJ}/${SES}/probtrackx2_clara/done_probtrackx2

rm $DONEFILE
	
# TODO: -x change to where seeds are stored
# TODO: change -o
# probtrackx2_gpu --onewaycondition -P 5000 --forcedir --opd --os2t \
# 	--rseed=1234 -s ${DWIPTH}/${SUBJ}/${SES}/dwi.bedpostX/merged \
# 	--dir=${DWIPTH}/${SUBJ}/${SES}/probtrackx2_clara \
# 	-m ${DWIPTH}/${SUBJ}/${SES}/dwi/${SUBJ}_${SES}_desc-preproc_space-dwi_brainmask.nii.gz  \
# 	--targetmasks=$TEXTOUT  \
# 	-x ${DWIPTH}/${SUBJ}/${SES}/probtrackx2_clara/${SUBJ}_${SES}_Schaefer_VVC_seed_regions_dwi.nii.gz \
# 	-o fdt_paths_hcp_rhodri

probtrackx2 --onewaycondition -P 5000 --forcedir --opd --os2t \
	--rseed=1234 -s ${DWIPTH}/${SUBJ}/${SES}/dwi.bedpostX/merged \
	--dir=${DWIPTH}/${SUBJ}/${SES}/probtrackx2_clara \
	-m ${DWIPTH}/${SUBJ}/${SES}/dwi/${SUBJ}_${SES}_desc-preproc_space-dwi_brainmask.nii.gz  \
	--targetmasks=$TEXTOUT  \
	-x ${DWIPTH}/${SUBJ}/${SES}/probtrackx2_clara/${SUBJ}_${SES}_Schaefer_VVC_seed_regions_dwi.nii.gz \
	-o fdt_paths_hcp_rhodri

touch $DONEFILE
