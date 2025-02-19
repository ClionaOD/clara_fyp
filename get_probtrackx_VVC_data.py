from nilearn import plotting
import nibabel as nib
import os
import numpy as np
import pickle

import argparse
import subprocess


class ProbtrackxVVC:
    def __init__(self, subject, session, base_dir, dwi_dir, n_rois) -> None:
        self.subject = subject
        self.session = session
        self.base_dir = base_dir
        self.dwi_dir = dwi_dir
        self.n_rois = n_rois

    # Function to generate the results path
    def get_results_path(self, ROI):
        return f"{self.dwi_dir}/{self.subject}/{self.session}/probtrackx2_clara/seeds_to_{self.subject}_{self.session}_A424_target_regions_dwi_{ROI}_40wk.nii.gz"

    def applyAntsWarp(self, input_image):
        # reference_image: image to be transformed to "${BASEDIR}/templates/nihpd-02-05_t1w_fcgmasked_2mm_nocerebellum.nii.gz"

        reference_image = f"{self.base_dir}/templates/nihpd-02-05_t1w_fcgmasked_2mm_nocerebellum.nii.gz"
        transform_1 = (
            f"{self.base_dir}/registration/dHCP_to_nihpd-02-05-2mm_1Warp.nii.gz"
        )
        transform_2 = (
            f"{self.base_dir}/registration/dHCP_to_nihpd-02-05-2mm_0GenericAffine.mat"
        )
        output_image = input_image.replace(".nii.gz", "_to_nihpd-02-05-2mm.nii.gz")

        cmd = f"antsApplyTransforms -d 3 -i {input_image} -r {reference_image} -o {output_image} -t {transform_1} -t {transform_2} -n NearestNeighbor"

        subprocess.run(cmd, shell=True)
        return output_image

    def get_sub_ses_arr(self, mask):
        rois = range(self.n_rois)

        n_voxels = np.sum(mask)

        subject_array = np.zeros((n_voxels, self.n_rois))
        for roi in rois:

            results_path = self.get_results_path(roi)
            if os.path.exists(results_path):

                transformed_path = self.applyAntsWarp(results_path)
                results_img = nib.load(transformed_path)
                results_data = results_img.get_fdata()

                assert (
                    results_data.shape == mask.shape
                ), f"Shape mismatch: {results_data.shape} vs {mask.shape}"

                roi_vector = results_data[mask]

                assert (
                    roi_vector.shape == n_voxels
                ), f"ROI vector shape mismatch: {roi_vector.shape} vs {n_voxels}"

                subject_array[:, roi] = roi_vector
            else:
                print(f"File not found: {results_path}")
                # ADD A COLUMN OF NANs
                subject_array[:, roi] = np.nan

        if np.all(subject_array == 0):
            print(f"No data found for {self.subject} {self.session}")
            return None
        else:
            return subject_array


if __name__ == "__main__":
    ## Load args from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", type=str, default=None)
    parser.add_argument("--ses", type=str, default=None)
    parser.add_argument(
        "--base_dir", type=str, default="/home/clionaodoherty/clara_fyp"
    )
    args = parser.parse_args()

    args.sub = "sub-CC00124XX09"
    args.ses = "ses-42302"
    args.base_dir = "/home/claraconyngham/clara_fyp"

    ## Define paths
    dwi_dir = "/dhcp/dhcp_dmri_pipeline"

    output_dir = "/dhcp/clara_analysis/probtrackx_seed_arrays"
    os.makedirs(output_dir, exist_ok=True)

    # Define VVC values
    schaefer_vvc = nib.load(
        f"{args.base_dir}/atlases/schaefer_nihpd-02-05_fcg_VVC.nii.gz"
    )
    VVC_mask = schaefer_vvc.get_fdata().astype(bool)

    print(f"WARNING: will overwrite current 40wk to 2month files in {dwi_dir}")
    get_vvc = ProbtrackxVVC(args.sub, args.ses, args.base_dir, dwi_dir, n_rois=396)

    subject_array = get_vvc.get_sub_ses_arr(VVC_mask)

    if subject_array is not None:
        # Save the subject_array for each session to a separate pickle file
        session_output_file = os.path.join(
            output_dir, f"{args.sub}_{args.ses}_nrois-{get_vvc.n_rois}_arrays.pkl"
        )
        with open(session_output_file, "wb") as f:
            pickle.dump(subject_array, f)
        print(f"Saved {session_output_file}")
