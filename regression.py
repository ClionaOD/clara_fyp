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
    nvoxels=4346,  # this is how many are in our 2 mm category models from twomonth space
):
    ## It takes a while, so let's save it out for easier loading later
    if os.path.exists(f"{base_dir}/regression_subsample.pkl"):
        subsample = pickle.load(open(f"{base_dir}/regression_subsample.pkl", "rb"))
    else:
        print("Sampling data...")

        for sub in sublist[:nsubs]:
            with open(sub, "rb") as f:
                subdata = pickle.load(f)
                subsample.append(subdata[:nvoxels, :])
        pickle.dump(subsample, open(f"{base_dir}/regression_subsample.pkl", "wb"))

    subsample = np.array(subsample)  # (nsubs, nvoxels, nrois)
    return subsample


def load_X_data(base_dir, sublist):
    sub_arrays = []
    for sub in sublist:
        with open(sub, "rb") as f:
            subdata = pickle.load(f)
            sub_arrays.append(subdata)
    sub_arrays = np.array(sub_arrays)  # (nsubs, nvoxels, nrois)
    return sub_arrays


def load_y_data(base_dir, category, nsubs=20):
    # Load the functional data
    func_dir = f"{base_dir}/img_results"
    funcfile = f"{func_dir}/{category}_both_vvc_twomonth_vcovthreshold10.nii.gz"  # TODO: change this to be the correct file if necessary. Should be in nihpd-02-05 2mm space

    func_img = nib.load(funcfile)
    # we can get the ROI voxels using non nan values, because of way data are saved
    func_data = func_img.get_fdata()
    func_data = func_data[~np.isnan(func_data)]  # (nvoxels,)
    return np.concatenate([func_data for i in range(nsubs)])  # (nvoxels*nsubs,)


if __name__ == "__main__":
    base_dir = "/home/clionaodoherty/clara_fyp"
    dhcp_dir = "/dhcp/clara_analysis/probtrackx_seed_arrays"
    allsubs = glob.glob(f"{dhcp_dir}/sub-*")

    # Load the data
    # For testing purposes, we're just going to use 5 subjects and sample their data along the n_voxel axis
    X_sample = load_X_sample(base_dir, allsubs)
    X_sample_reshaped = X_sample.reshape(
        -1, X_sample.shape[-1]
    )  # (nvoxels*nsubs, nrois)

    ## TODO: Load the actual data this time.
    ## Uncomment the following lines and comment out the above lines
    ## Change all following references to X_sample to X
    # X = load_X_data(base_dir, allsubs)
    # X_reshaped = X.reshape(-1, X.shape[-1])  # (nvoxels*nsubs, nrois)

    y = load_y_data(base_dir, "cat")

    # check that the data is the same shape
    assert (
        X_sample_reshaped.shape[0] == y.shape[0]
    ), "X and y n_voxel shapes don't match along 1st dimension"

    # Get group (subject) labels for doing GroupKFold cross validation
    # These should be the same shape as the y data
    # and should point to values in the 2nd dimension of the X_sample_reshaped data that correspond to the same subject
    group_labels = np.repeat(np.arange(X_sample.shape[0]), X_sample.shape[1])

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    split = next(splitter.split(X_sample_reshaped, y, group_labels))
    X_train, X_test = X_sample_reshaped[split[0]], X_sample_reshaped[split[1]]
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

    ## Summary
    # mse tells us how well the model is doing. If it's close to 0, the model is doing well.
    # coefficients tells us which rois are important for the model. This should be of shape (nrois,)
