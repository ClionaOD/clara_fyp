import argparse
import glob
import os
import pickle

import nibabel as nib
import numpy as np
import pandas as pd

from scipy.stats import zscore
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.dummy import DummyRegressor


def load_X_sample(
    base_dir,
    sublist,
    nsubs=20,
    nvoxels=4355,  # this is how many are in our 2 mm category models from twomonth space
):
    ## It takes a while, so let's save it out for easier loading later
    if os.path.exists(f"{base_dir}/regression_subsample2.pkl"):
        subsample = pickle.load(open(f"{base_dir}/regression_subsample2.pkl", "rb"))
    else:
        print("Sampling data...")

        subsample = []
        for sub in sublist[:nsubs]:
            with open(sub, "rb") as f:
                subdata = pickle.load(f)
                subdata[np.isnan(subdata)] = 0
                subsample.append(subdata[:nvoxels, :])
        subsample = np.array(subsample)  # (nsubs, nvoxels, nrois)
        pickle.dump(subsample, open(f"{base_dir}/regression_subsample2.pkl", "wb"))
    return subsample


def load_X_data(base_dir, sublist):
    if os.path.exists(f"{base_dir}/regression_subarrays2.npy"):
        sub_arrays = np.load(f"{base_dir}/regression_subarrays2.npy")
    else:
        sub_arrays = []
        for sub in sublist:
            with open(sub, "rb") as f:
                subdata = pickle.load(f)
                # drop 120th column because it's all nans
                #subdata = np.delete(subdata, 120, axis=1)
                # because the values in X are counts out of number of streamlines
                # we can replace the nans with 0s
                if np.any(np.isnan(subdata)):
                    print(f"Found nans in {sub}")

                    if np.unique(np.where(np.isnan(subdata))[1]).shape[0] > 1:
                        print(f"More than one column with nans, removing {sub}")
                        continue

                    print(np.unique(np.where(np.isnan(subdata))[1]))
                    subdata[np.isnan(subdata)] = 0
                sub_arrays.append(subdata)
        sub_arrays = np.array(sub_arrays)  # (nsubs, nvoxels, nrois)
        print(f"Sub array shape: {sub_arrays.shape}")
        np.save(f"{base_dir}/regression_subarrays2.npy", sub_arrays)
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
    # replace nans with the mean of the data
    func_data[np.isnan(func_data)] = np.nanmean(func_data)
    func_data = zscore(func_data, axis=0)
    func_data = func_data.reshape(-1)
    return np.concatenate([func_data for i in range(nsubs)])  # (nvoxels*nsubs,)


if __name__ == "__main__":
    dhcp_dir = "/dhcp/clara_analysis/probtrackx_seed_arrays"
    allsubs = glob.glob(f"{dhcp_dir}/sub-*")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/clionaodoherty/clara_fyp",
        help="Base directory for the data",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="cat",
        help="Category to use for the regression",
    )
    args = parser.parse_args()

    # Load the data
    # For testing purposes, we're just going to use 5 subjects and sample their data along the n_voxel axis
    #X = load_X_sample(args.base_dir, allsubs)
    # X_reshaped = X.reshape(-1, X.shape[-1])  # (nvoxels*nsubs, nrois)

    schaefer_vvc = nib.load(
        f"{args.base_dir}/atlases/schaefer_nihpd-02-05_fcg_VVC.nii.gz"
    )
    VVC_mask = schaefer_vvc.get_fdata().astype(bool)

    ## TODO: Load the actual data this time.
    ## Uncomment the following lines and comment out the above lines
    ## Change all following references to X_sample to X
    X = load_X_data(args.base_dir, allsubs) # shape (nsubs, nvoxels, nrois)
    
    # zscore the data
    for sub in range(X.shape[0]):
        for roi in range(X.shape[2]):
            X[sub, :, roi] = zscore(X[sub, :, roi], axis=0, nan_policy="omit")
    
    #Exclude vvc rois 
    roi_inds = np.arange(X.shape[2])
    rois_to_exclude = [0, 7, 18, 21, 22, 119, 120, 121, 122, 126, 127, 135, 137, 138, 153, 154, 155, 157, 160, 163, 187, 198, 201, 202, 211, 299, 300, 301, 302, 306, 307, 315, 318, 333, 334, 335, 337, 340, 343, 365, 367, 368]
    new_roi_labels = np.delete(roi_inds, rois_to_exclude)
    # new_roi_labels should be the same shape as new number of rois (396 - ones we got rid of)
    # So this will be same length as coefficients
    

    roi_mask = np.ones(X.shape[2], dtype=bool)
    roi_mask[rois_to_exclude] = False
    X = X[:, :, roi_mask]

    X_reshaped = X.reshape(-1, X.shape[-1])  # (nvoxels*nsubs, nrois)
    X_reshaped[np.isnan(X_reshaped)] = 0
    

    y = load_y_data(args.base_dir, args.category, nsubs=X.shape[0], masker=VVC_mask)

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

    if os.path.exists(f"{args.base_dir}/{args.category}_best_alpha_l1.pkl"):
        # load the best alpha and l1_ratio
        best_alpha, l1_ratio = pickle.load(
            open(f"{args.base_dir}/{args.category}_best_alpha_l1.pkl", "rb")
        )
    #   load the stuff
    #   best_alpha = load (same for best l1)
    else:
        #   do the stuff in lines 129 - 143

        baseline = DummyRegressor(strategy="mean")
        baseline.fit(X_train, y_train)
        baseline_pred = baseline.predict(X_test)
        baseline_mse = mean_squared_error(y_test, baseline_pred)
        print(f"Baseline MSE: {baseline_mse}")

        model_cv = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
            n_alphas=100,
            cv=group_kfold.split(X_train, y_train, groups_train),
            random_state=42,
            fit_intercept=True,  # if False, data is expected to be already centered
            n_jobs=-1,  # if -1, use all available CPUs
        )
        model_cv.fit(X_train, y_train)

        ## Check fit params
        best_alpha = model_cv.alpha_
        print(f"Best alpha: {best_alpha}")
        l1_ratio = model_cv.l1_ratio_
        print(f"Best l1_ratio: {l1_ratio}")
        # save the best alpha and l1_ratio into a pickle file
        pickle.dump(
            (best_alpha, l1_ratio),
            open(f"{args.base_dir}/{args.category}_best_alpha_l1.pkl", "wb"),
        )

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
    coefficients = pd.DataFrame(
        coefficients, index=new_roi_labels, columns=["Coefficient"]
    )
    # order the coefficients by value
    coefficients = coefficients.sort_values(by="Coefficient", ascending=False)


    ## Now we can evaluate the model on the test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse}")


    with open(f"{args.category}_results.pickle", "wb") as f:
        pickle.dump(
            {
                "mse": mse,
                "r2": r2,
                "coefficients": coefficients,
                "model": model,
                "nsubs": X.shape[0],
                "nvoxels": X.shape[1],
                "nrois": X.shape[2],
                "baseline_MSE": baseline_mse,
            },
            f,
        )

    ## Summary
    # mse tells us how well the model is doing. If it's close to 0, the model is doing well.
    # coefficients tells us which rois are important for the model. This should be of shape (nrois,)
