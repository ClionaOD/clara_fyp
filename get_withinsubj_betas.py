import pandas as pd
import numpy as np

AGE = "twomonth"
VCOVTHRESHOLD = 10
ROI = "both_vvc"

rdms = pd.read_pickle(
    f"/foundcog/foundcog_results/pictures_roi_rdms_{AGE}_vcovthreshold{VCOVTHRESHOLD}_reps_correlation_newsubs.pickle"
)
betas = pd.read_pickle(
    f"/foundcog/foundcog_results/pictures_roi_betas_bysubj_{AGE}_vcovthreshold{VCOVTHRESHOLD}_reps_newsubs.pickle"
)

subs = betas["sub_list_used"]
conditions = [i.split("_")[0] for i in rdms["condnames"]]
roivalues = betas["roivalues"][ROI]


withinsub_acrossrunrep_rdms = {}

all_runreps = [
    {"sub": sub, "runrep": runrep}
    for sub in roivalues.keys()
    for runrep in roivalues[sub].keys()
]

for ind1, rep1 in enumerate(all_runreps):
    for ind2 in range(ind1, len(all_runreps)):
        rep2 = all_runreps[ind2]

        if (rep1["sub"] == rep2["sub"]) & (rep1["runrep"] != rep2["runrep"]):

            rep1_betas = roivalues[rep1["sub"]][rep1["runrep"]]["betas"]
            ncond = rep1_betas.shape[1]
            rep1_vcov = roivalues[rep1["sub"]][rep1["runrep"]]["vcov"]

            rep2_betas = roivalues[rep2["sub"]][rep2["runrep"]]["betas"]
            rep2_vcov = roivalues[rep2["sub"]][rep2["runrep"]]["vcov"]

            rep1_betas[rep1_vcov > VCOVTHRESHOLD] = np.nan
            rep2_betas[rep2_vcov > VCOVTHRESHOLD] = np.nan

            rep1_betas -= np.nanmean(rep1_betas, axis=1, keepdims=True)
            rep2_betas -= np.nanmean(rep2_betas, axis=1, keepdims=True)

            rdm = (
                pd.DataFrame(np.concatenate((rep1_betas, rep2_betas), axis=1))
                .corr()
                .values
            )

            rdm = rdm[:ncond, ncond:]
            rdm = (rdm + rdm.T) / 2

            if rep1["sub"] not in withinsub_acrossrunrep_rdms.keys():
                withinsub_acrossrunrep_rdms[rep1["sub"]] = []

            withinsub_acrossrunrep_rdms[rep1["sub"]].append(rdm)

withinsub_acrossrunrep_rdms = {
    k: np.array(v) for k, v in withinsub_acrossrunrep_rdms.items()
}

EGSUB = "sub-ICC133_task-pictures"
_eg_rdm = np.nanmean(withinsub_acrossrunrep_rdms[EGSUB], axis=0)
_eg_rdm = pd.DataFrame(_eg_rdm, index=conditions, columns=conditions)

_model_rdm = np.eye(len(conditions))

from scipy.stats import zscore, spearmanr


def zscore_vectorize(rdm, k=0):
    """
    Function to turn an RDM into a vector so that we can do correlation.
    The argument k determines if we include the diagonal or not, 0 means it is included.
    We also zscore the vector so that we can compare to data from other sources.
    """
    rdm_v = rdm[np.triu_indices(rdm.shape[0], k=k)]
    rdm_v = zscore(rdm_v, nan_policy="omit")
    return rdm_v


def compare_rdms(rdm1, rdm2):
    rdm1_v = zscore_vectorize(rdm1)
    rdm2_v = zscore_vectorize(rdm2)

    return spearmanr(rdm1_v, rdm2_v)


print(compare_rdms(_eg_rdm.values, _model_rdm))

# set lower triangle to be nan
# _eg_rdm.values[np.triu_indices_from(_eg_rdm, k=1)] = np.nan
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(_eg_rdm, cmap="seismic", square=True)
plt.tight_layout()
plt.savefig(f"{EGSUB}_withinrep_acrossrunrep_rdm.png")
plt.close()

print()
