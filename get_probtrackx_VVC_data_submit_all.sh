#!/bin/bash

DWIPATH="/dhcp/dhcp_dmri_pipeline"
BASEDIR="/home/claraconyngham/clara_fyp"

# Loop through all subjects
for SUBJDIR in "${DWIPATH}"/sub-CC*; do
   if [[ -d "$SUBJDIR" ]]; then
       SUBJ=$(basename -- "$SUBJDIR")
       
       # Loop through all sessions
       for SESSDIR in "${SUBJDIR}"/ses-*; do
           if [[ -d "$SESSDIR" ]]; then
               MYSESS=$(basename -- "$SESSDIR")
               
               sbatch --output="${BASEDIR}/slurm_logs/${SUBJ}_${MYSESS}_get_voxels_%j.out" \
                      "${BASEDIR}/get_probtrackx_VVC_data.sh" "$SUBJ" "$MYSESS" "$BASEDIR"
           fi
       done
   fi
done
