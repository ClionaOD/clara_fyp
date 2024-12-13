import pandas as pd
import numpy as np

import nibabel as nib

test = 1

## Set constants for loading our results
PATH = "/foundcog/foundcog_results/pictures_roi_betas_bysubj_twomonth_vcovthreshold10_eg_newsubs.pickle"
BETAREJECT = "vcovthreshold"
VCOVTHRESHOLD = 10
ROI = "both_vvc"

## Load the atlas and the results
masker, labels, vertices = pd.read_pickle("./data/twomonth_atlas.pickle")

res = pd.read_pickle(PATH)

## Extract the conditions and the indices of the conditions
conditions = res["condnames"]
condition_set = set(
    [cond[:-1] for cond in conditions]
)  # this will remove the number from the end of the condition name
category_inds = {
    category: [i for i, cond in enumerate(conditions) if category in cond]
    for category in condition_set
}  # this will give us a dictionary with the category name as the key, and the indices of the conditions in that category as the value


## Extract the betas for our chosen ROI
roivalues = res["roivalues"][ROI]

# Initiate empty lists to store the data
all_runreps = [
    {"sub": sub, "runrep": runrep}
    for sub in roivalues.keys()
    for runrep in roivalues[sub].keys()
]

sublist = []
betalist = []
subcounts = {}

## Loop through the data and store the betas
for session in all_runreps:
    thissub = session["sub"]
    sublist.append(thissub)

    # we need to keep track of how many times we've seen this subject
    if thissub not in subcounts:
        subcounts[thissub] = 0
    subcounts[thissub] += 1

    run_betas = roivalues[thissub][session["runrep"]][
        "betas"
    ]  # the estimated response in a voxel
    run_vcov = roivalues[thissub][session["runrep"]][
        "vcov"
    ]  # the variance-covariance matrix of the estimated response

    # because some infants are noisy, we must threshold using the variance-covariance matrix
    # this gets rid of voxels which have poor quality beta estimates
    if BETAREJECT == "vcovthreshold":
        run_betas[run_vcov > VCOVTHRESHOLD] = np.nan  # set them to "not a number"

    # we want to remove the mean of the betas for each voxel, to normalise the data
    run_betas -= np.nanmean(run_betas, axis=1, keepdims=True)

    betalist.append(run_betas)

## Do the weighted averaging, using the number of times we've seen each subject as the weight
subweights = [subcounts[sub] for sub in sublist]
allbetas = np.stack(betalist, axis=0)
ma = np.ma.MaskedArray(allbetas, mask=np.isnan(allbetas))
ma_result = np.ma.average(ma, weights=subweights, axis=0)
weighted_average = ma_result.data

## Put the data back into the brain image
brain_vec = np.zeros_like(
    labels, dtype=float
)  # this will be a vector with one value per voxel in the brain
brain_vec[brain_vec == 0] = (
    np.nan
)  # set all values to "not a number" so that we have an empty brain

assert len(vertices[ROI]) == len(
    weighted_average
)  # make sure we have the same number of voxels in the brain as we have betas, this is a sanity check


category_res_df = pd.DataFrame(columns=list(condition_set), index=vertices[ROI])
for category, inds in category_inds.items():
    # Find the average beta for the category, eg. np.mean('cat1','cat2','cat3')
    # We're using nanmean to ignore nans, as we have thresholded some voxels
    category_beta_avg = np.nanmean(weighted_average[:, inds], axis=1)

    ## store the results in a dataframe
    category_res_df.loc[vertices[ROI], category] = category_beta_avg

    # Because we were extracting betas from an ROI, we need to use the ROI mask to put the betas back into the brain
    # vertices[ROI] is a list of the indices of the voxels in the ROI
    # so we can get the indices that match this region using vertices[ROI]
    # then we set the values from np.nan to the average beta for the category
    brain_vec[vertices[ROI]] = category_beta_avg

    # this is some practical stuff to get the 1d vector back into a 3d brain image
    brain_img = masker.inverse_transform(brain_vec.reshape((1, -1)))
    _img = brain_img.get_fdata()
    _img[_img == 0] = np.nan

    # Save the image
    brain_img = nib.Nifti1Image(_img, brain_img.affine)
    nib.save(
        brain_img, f"./img_results/{category}_{ROI}_twomonth_vcovthreshold10.nii.gz"
    )

category_res_df.to_csv(f"./results/{ROI}_twomonth_vcovthreshold10.csv")
