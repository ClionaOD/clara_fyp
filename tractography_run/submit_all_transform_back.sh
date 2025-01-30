#!/bin/bash

DWIPTH=/dhcp/dhcp_dmri_pipeline
BASEDIR=/home/claraconyngham/clara_fyp

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
			if [[ -f "${DWIPTH}/${SUBJ}/${MYSESS}/probtrackx2_clara/done_transform_back" ]]; then
				echo "Already ran transform back"
			else
				sbatch --output=${BASEDIR}/slurm_logs/transforms_%j.out ${BASEDIR}/tractography_run/run_transform_back.sh $SUBJ $MYSESS
			fi		
		fi
			
	fi 	    
   done

done
