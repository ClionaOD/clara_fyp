from nilearn import plotting
import nibabel as nib
import os
import numpy as np
import pickle

import argparse


# Function to generate the results path
def get_results_path(subject, session, ROI):
    return f"{dwi_dir}/{subject}/{session}/probtrackx2_clara/seeds_to_{subject}_{session}_A424_target_regions_dwi_{ROI}_40wk.nii.gz"


def get_sub_ses_arr(subject, session, dwi_dir, mask, n_rois=396):
    rois = range(n_rois)

    n_voxels = np.sum(mask)

    subject_array = np.zeros((n_voxels, n_rois))
    for roi in rois:
        results_path = get_results_path(subject, session, roi)

        if os.path.exists(results_path):
            results_img = nib.load(results_path)
            results_data = results_img.get_fdata()

            assert (
                results_data.shape == mask.shape
            ), f"Shape mismatch: {results_data.shape} vs {mask.shape}"

            roi_vector = results_data[mask]

            assert (
                roi_vector.shape == n_voxels
            ), f"ROI vector shape mismatch: {roi_vector.shape} vs {n_voxels}"

            subject_array[:, roi] = roi_vector

    if np.all(subject_array == 0):
        print(f"No data found for {subject} {session}")
        return None
    else:
        return subject_array


if __name__ == "__main__":
    ## Load args from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", type=str, default=None)
    parser.add_argument("--ses", type=str, default=None)
    args = parser.parse_args()

    ## Define paths
    dwi_dir = "/dhcp/dhcp_dmri_pipeline"

    base_dir = "/home/claraconyngham/clara_fyp"

    template_path = f"{base_dir}/templates/dHCP40wk_template_t1.nii.gz"
    output_dir = f"{base_dir}/probtrackx_seed_arrays"
    os.makedirs(output_dir, exist_ok=True)

    # Define VVC values
    schaefer_vvc = nib.load(f"{base_dir}/atlases/schaefer_40weeks_VVC.nii.gz")
    VVC_values = np.array(
        [
            1,
            2,
            3,
            4,
            5,
            7,
            8,
            10,
            11,
            13,
            16,
            201,
            202,
            203,
            204,
            205,
            206,
            207,
            209,
            211,
            212,
            214,
        ]
    )
    VVC_mask = np.isin(schaefer_vvc.get_fdata(), VVC_values)

    ## Get subject array
    subject = args.sub
    session = args.ses

    subject_array = get_sub_ses_arr(subject, session, dwi_dir, VVC_mask)

    if subject_array is not None:
        # Save the subject_array for each session to a separate pickle file
        session_output_file = os.path.join(
            output_dir, f"{subject}_{session}_arrays.pkl"
        )
        with open(session_output_file, "wb") as f:
            pickle.dump(subject_array, f)
        print(f"Saved {session_output_file}")
