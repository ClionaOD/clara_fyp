#!/bin/bash

DWIPTH=/dhcp/dhcp_dmri_pipeline

# All subjects and sessions
for SUBJDIR in ${DWIPTH}/sub-CC*; do
   SUBJ=$(basename -- "$SUBJDIR")
   for SESSDIR in ${SUBJDIR}/ses-*/; do
	if [[ -d "$SESSDIR" ]]
	then
		MYSESS=$(basename -- "$SESSDIR")
		# Check for bedpostX outputs
		if [[ -f "${DWIPTH}/${SUBJ}/${MYSESS}/dwi.bedpostX/merged_f1samples.nii.gz" ]]; then
  			echo "With bedpostX output ${SUBJ}_${MYSESS}"
			if [[ -f "${DWIPTH}/${SUBJ}/${MYSESS}/probtrackx2/done_run_preparerois" ]]; then
				echo "Already run prepare ROIS"
			else
				sbatch /dhcp/rhodri_registration/scripts/run_preparerois.sh $SUBJ $MYSESS
			fi		
		fi
			
	fi 	    
   done

done
