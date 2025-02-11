from nilearn import plotting
import nibabel as nib
import os
import numpy as np
import pickle

dwi_dir = "/dhcp/dhcp_dmri_pipeline"

base_dir = '/home/claraconyngham/clara_fyp'
template_path = f"{base_dir}/templates/dHCP40wk_template_t1.nii.gz"
output_dir = f"{base_dir}/probtrackx_seed_arrays"
os.makedirs(output_dir, exist_ok=True)

# Function to generate the results path
def get_results_path(subject, session, ROI):
    return f"{dwi_dir}/{subject}/{session}/probtrackx2_clara/seeds_to_{subject}_{session}_A424_target_regions_dwi_{ROI}_40wk.nii.gz"

# Define VVC values
VVC_values = np.array([1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 16, 201, 202, 203, 204, 205, 206, 207, 209, 211, 212, 214])
schaefer_vvc = nib.load('atlases/schaefer_40weeks_VVC.nii.gz')
VVC_mask = np.isin(schaefer_vvc.get_fdata(), VVC_values)
n_voxels = np.sum(VVC_mask)

subject_arrays = []

for idx, subject in enumerate(os.listdir(dwi_dir)):
    if idx >= 1:
        break
    subject_path = os.path.join(dwi_dir, subject)
    for session in os.listdir(subject_path):
        session_path = os.path.join(subject_path, session)

        if not os.path.isdir(session_path):
            continue

        rois = range(397)
        
        subject_array = np.zeros((n_voxels, len(rois)))
        for roi in rois:
            results_path = get_results_path(subject, session, roi)
            if os.path.exists(results_path):
                results_img = nib.load(results_path)
                results_data = results_img.get_fdata()
                
                assert results_data.shape == schaefer_vvc.shape, f"Shape mismatch: {results_data.shape} vs {schaefer_vvc.shape}"
                
                vvc_vector = results_data[VVC_mask]

                assert vvc_vector.shape == n_voxels, f"VVC vector shape mismatch: {vvc_vector.shape} vs {n_voxels}"

                subject_array[:,roi] = vvc_vector 
        # Save the subject_array for each session to a separate pickle file
        session_output_file = os.path.join(output_dir, f"subject_{subject}_session_{session}_arrays.pkl")
        with open(session_output_file, 'wb') as f:
            pickle.dump(subject_array, f)
        print(f"Saved {session_output_file}")

