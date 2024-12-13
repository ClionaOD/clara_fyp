import pandas as pd
import numpy as np

import nibabel as nib

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import shapiro

ROI = "both_vvc"

data = pd.read_csv(f"results/{ROI}_twomonth_vcovthreshold10.csv")

brain_mask = "/foundcog/templates/mask/nihpd_asym_02-05_fcgmask_2mm.nii.gz"

# find the 95th percentile of each column
for condition in data.columns:
    print(f"{condition} 95th percentile: {data[condition].quantile(0.95)}")
    # find the indices that are above the 95th percentile
    print(
        f"{condition} number of voxels above 95th percentile: {len(data[data[condition] > data[condition].quantile(0.95)])}"
    )

# for each column, test if normally distributed across voxels
for condition in data.columns:
    stat, p = shapiro(data[condition])
    if p > 0.05:
        print(f"{condition} is normally distributed")

sns.set(style="whitegrid")
for condition in data.columns:
    plt.hist(data[condition], bins=20, alpha=0.5, label=condition)
    plt.tight_layout()
    plt.savefig(f"{condition}_both_vvc_twomonth_betadist_acrossvoxels.png")
    plt.close()


### YOUR CODE HERE ###
# find the indices of the 4346 array that are above/below the 95th percentile
# e.g cat_inds = [1,5,2,6,...]

condition_inds = {}
for condition in data.columns:
    # Calculate the 97.5th and 2.5th percentiles
    upper_threshold = data[condition].quantile(0.975)
    lower_threshold = data[condition].quantile(0.025)

    # Get indices of voxels above 97.5th percentile or below 2.5th percentile
    combined_indices = data.index[
        (data[condition] > upper_threshold) | (data[condition] < lower_threshold)
    ].tolist()

    # Create the named variable for the indices
    condition_inds[f"{condition}_inds"] = combined_indices

    # Print summary
    print(f"{condition} 97.5th percentile: {upper_threshold}")
    print(f"{condition} 2.5th percentile: {lower_threshold}")
    print(
        f"{condition} total number of voxels outside thresholds: {len(combined_indices)}"
    )


## Load the atlas and the results
masker, labels, vertices = pd.read_pickle("./data/twomonth_atlas.pickle")

## This loop will save out brain images for just the top/bottom voxels
## after you do your thing
## this will save one image per category
for condition in data.columns:

    # Get the voxel indices for the current category
    voxel_indices = condition_inds[f"{condition}_inds"]

    ## Put the data back into the brain image
    brain_vec = np.zeros_like(
        labels, dtype=float
    )  # this will be a vector with one value per voxel in the brain
    brain_vec[brain_vec == 0] = (
        np.nan
    )  # set all values to "not a number" so that we have an empty brain

    brain_vec[voxel_indices] = data.loc[voxel_indices, condition]

    # this is some practical stuff to get the 1d vector back into a 3d brain image
    brain_img = masker.inverse_transform(brain_vec)
    _img = brain_img.get_fdata()
    _img[_img == 0] = np.nan

    # Save the image
    brain_img = nib.Nifti1Image(_img, brain_img.affine)
    nib.save(
        brain_img,
        f"./img_results/{condition}_{ROI}_twomonth_vcovthreshold10_percentilethresholded.nii.gz",
    )
