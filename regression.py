import glob
import pickle
import os

import pandas as pd
import numpy as np
import nibabel as nib

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error


def load_X_sample(
    base_dir,
    sublist,
    nsubs=20,
    nvoxels=4355,  # this is how many are in our 2 mm category models from twomonth space
):
    ## It takes a while, so let's save it out for easier loading later
    if os.path.exists(f"{base_dir}/regression_subsample.pkl"):
        subsample = pickle.load(open(f"{base_dir}/regression_subsample.pkl", "rb"))
    else:
        print("Sampling data...")

        subsample = []
        for sub in sublist[:nsubs]:
            with open(sub, "rb") as f:
                subdata = pickle.load(f)
                subdata[np.isnan(subdata)] = 0
                subsample.append(subdata[:nvoxels, :])
        subsample = np.array(subsample)  # (nsubs, nvoxels, nrois)
        pickle.dump(subsample, open(f"{base_dir}/regression_subsample.pkl", "wb"))
    return subsample


def load_X_data(base_dir, sublist):
    if os.path.exists(f"{base_dir}/regression_subarrays.pkl"):
        sub_arrays = pickle.load(open(f"{base_dir}/regression_subarrays.pkl", "rb"))
    else:
        sub_arrays = []
        for sub in sublist:
            with open(sub, "rb") as f:
                subdata = pickle.load(f)
                # because the values in X are counts out of number of streamlines
                # we can replace the nans with 0s
                subdata[np.isnan(subdata)] = 0
                sub_arrays.append(subdata)
        sub_arrays = np.array(sub_arrays)  # (nsubs, nvoxels, nrois)
        pickle.dump(sub_arrays, open(f"{base_dir}/regression_subarrays.pkl", "wb"))
    return sub_arrays


def load_y_data(base_dir, category, nsubs=20, masker=None):
    # Load the functional data
    func_dir = f"{base_dir}/img_results"
    funcfile = f"{func_dir}/{category}_both_vvc_twomonth_vcovthreshold10.nii.gz"  # TODO: change this to be the correct file if necessary. Should be in nihpd-02-05 2mm space

    func_img = nib.load(funcfile)
    # we can get the ROI voxels using non nan values, because of way data are saved
    func_data = func_img.get_fdata()
    if masker is None:
        func_data = func_data[~np.isnan(func_data)]  # (nvoxels,)
    else:
        func_data = func_data[masker]
    return np.concatenate([func_data for i in range(nsubs)])  # (nvoxels*nsubs,)


if __name__ == "__main__":
    base_dir = "/home/claraconyngham/clara_fyp"
    dhcp_dir = "/dhcp/clara_analysis/probtrackx_seed_arrays"
    allsubs = glob.glob(f"{dhcp_dir}/sub-*")

    # Load the data
    # For testing purposes, we're just going to use 5 subjects and sample their data along the n_voxel axis
    # X = load_X_sample(base_dir, allsubs)
    # X_reshaped = X.reshape(-1, X.shape[-1])  # (nvoxels*nsubs, nrois)

    schaefer_vvc = nib.load(f"{base_dir}/atlases/schaefer_nihpd-02-05_fcg_VVC.nii.gz")
    VVC_mask = schaefer_vvc.get_fdata().astype(bool)

    ## TODO: Load the actual data this time.
    ## Uncomment the following lines and comment out the above lines
    ## Change all following references to X_sample to X
    X = load_X_data(base_dir, allsubs)
    X_reshaped = X.reshape(-1, X.shape[-1])  # (nvoxels*nsubs, nrois)

    y = load_y_data(base_dir, "cat", nsubs=X.shape[0], masker=VVC_mask)
    # replace nans with the mean of the data
    y[np.isnan(y)] = np.nanmean(y)

    # check that the data is the same shape
    assert (
        X_reshaped.shape[0] == y.shape[0]
    ), "X and y n_voxel shapes don't match along 1st dimension"

    # Get group (subject) labels for doing GroupKFold cross validation
    # These should be the same shape as the y data
    # and should point to values in the 2nd dimension of the X_sample_reshaped data that correspond to the same subject
    group_labels = np.repeat(np.arange(X.shape[0]), X.shape[1])

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    split = next(splitter.split(X_reshaped, y, group_labels))
    X_train, X_test = X_reshaped[split[0]], X_reshaped[split[1]]
    y_train, y_test = y[split[0]], y[split[1]]
    groups_train, groups_test = (
        group_labels[split[0]],
        group_labels[split[1]],
    )

    group_kfold = GroupKFold(n_splits=5)

    for i, (train_index, test_index) in enumerate(
        group_kfold.split(X_train, y_train, groups_train)
    ):
        print(f"Fold {i}:")
        print(
            f"  Train: index={train_index}, group={np.unique(groups_train[train_index])}"
        )
        print(
            f"  Test:  index={test_index}, group={np.unique(groups_train[test_index])}"
        )

    if os.path.exists(f"{base_dir}/best_alpha_l1.pkl"):
        # load the best alpha and l1_ratio
        best_alpha, l1_ratio = pickle.load(open(f"{base_dir}/best_alpha_l1.pkl", "rb"))
    #   load the stuff
    #   best_alpha = load (same for best l1)
    else:
    #   do the stuff in lines 129 - 143

        model_cv = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
            n_alphas=100,
            cv=group_kfold.split(X_train, y_train, groups_train),
            random_state=42,
            fit_intercept=True,  # if False, data is expected to be already centered
            n_jobs=None,  # if -1, use all available CPUs
        )
        model_cv.fit(X_train, y_train)

        ## Check fit params
        best_alpha = model_cv.alpha_
        print(f"Best alpha: {best_alpha}")
        l1_ratio = model_cv.l1_ratio_
        print(f"Best l1_ratio: {l1_ratio}")
        # save the best alpha and l1_ratio into a pickle file
        pickle.dump((best_alpha, l1_ratio), open(f"{base_dir}/best_alpha_l1.pkl", "wb"))
    

    # To get out the coefficients, we need to refit the model with the best alpha
    model = ElasticNet(
        alpha=best_alpha,
        l1_ratio=l1_ratio,
        random_state=42,
        fit_intercept=True,
        normalize=False,
    )
    model.fit(X_train, y_train)
    coefficients = model.coef_

    ## Now we can evaluate the model on the test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE: {mse}")

    ## Summary
    # mse tells us how well the model is doing. If it's close to 0, the model is doing well.
    # coefficients tells us which rois are important for the model. This should be of shape (nrois,)
