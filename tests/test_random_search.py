from sweeps.config import SweepConfig
import numpy as np
from numpy import typing as npt
from sweeps.run import next_run
from scipy.stats import chi2


def check_that_samples_are_from_the_same_distribution_according_to_chisq_two_sample_test(
    samples_1: npt.ArrayLike,
    samples_2: npt.ArrayLike,
    bins: npt.ArrayLike,
) -> bool:
    """
    https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/chi2samp.htm

    parameters
    ----------

    samples_1: (NSAMP1, NDIM) array
    samples_2: (NSAMP2, NDIM) array
    bins: (NDIM, NBIN + 1) array

    """

    samples_1 = np.asarray(samples_1)
    samples_2 = np.asarray(samples_2)
    bins = np.asarray(bins)

    n_true, _ = np.histogramdd(samples_1, bins)
    n_pred, _ = np.histogramdd(samples_2, bins)

    k1 = np.sqrt(n_true.sum() / n_pred.sum())
    k2 = np.sqrt(n_pred.sum() / n_true.sum())

    chisq = np.sum((k1 * n_true - k2 * n_pred) ** 2 / (n_true.sum() + n_pred.sum()))

    # number of non-empty bins
    k = np.argwhere((n_true != 0) | (n_pred != 0)).shape[0]

    # 0 if sample sizes are different, else 1
    c = 1 if n_pred.sum() == n_true.sum() else 0
    dof = k - c

    # check that the samples are the same at 5 sigma significance (99.99997% significance)
    return chisq < chi2.ppf(0.0000003, dof)


def plot_two_distributions(
    samples_1: npt.ArrayLike,
    samples_2: npt.ArrayLike,
    bins: npt.ArrayLike,
):
    import matplotlib.pyplot as plt

    plt.hist(samples_1, bins=bins, histtype="stepfilled", label="set1", alpha=0.2)
    plt.hist(samples_2, bins=bins, histtype="stepfilled", label="set2", alpha=0.2)
    plt.legend()
    plt.show()


def test_rand_uniform_single():
    # Calculates that the

    v1_min = 3.0
    v1_max = 5.0
    v2_min = 5.0
    v2_max = 6.0
    n_samples = 10000

    sweep_config_2params = SweepConfig(
        {
            "method": "random",
            "parameters": {
                "v1": {"min": v1_min, "max": v1_max},
                "v2": {"min": v2_min, "max": v2_max},
            },
        }
    )

    runs = []
    for i in range(n_samples):
        suggestion = next_run(sweep_config_2params, runs)
        runs.append(suggestion)

    v1_pred_samples = np.asarray([run.config["v1"]["value"] for run in runs])
    v2_pred_samples = np.asarray([run.config["v2"]["value"] for run in runs])

    v1_true_samples = np.random.uniform(v1_min, v1_max, size=n_samples)
    v2_true_samples = np.random.uniform(v2_min, v2_max, size=n_samples)

    v1_bins = np.linspace(v1_min, v1_max, 10)
    v2_bins = np.linspace(v2_min, v2_max, 10)

    pred_samples = np.transpose(np.vstack([v1_pred_samples, v2_pred_samples]))
    true_samples = np.transpose(np.vstack([v1_true_samples, v2_true_samples]))
    bins = np.vstack([v1_bins, v2_bins])

    dist_ok = check_that_samples_are_from_the_same_distribution_according_to_chisq_two_sample_test(
        pred_samples, true_samples, bins
    )
    assert dist_ok


def test_rand_normal_and_uniform():
    # Calculates that the

    v2_min = 5.0
    v2_max = 6.0
    n_samples = 10000

    sweep_config_2params = SweepConfig(
        {
            "method": "random",
            "parameters": {
                "v1": {"distribution": "normal"},
                "v2": {"min": v2_min, "max": v2_max},
            },
        }
    )

    runs = []
    for i in range(n_samples):
        suggestion = next_run(sweep_config_2params, runs)
        runs.append(suggestion)

    v1_pred_samples = np.asarray([run.config["v1"]["value"] for run in runs])
    v2_pred_samples = np.asarray([run.config["v2"]["value"] for run in runs])

    v1_true_samples = np.random.normal(0, 1, size=n_samples)
    v2_true_samples = np.random.uniform(v2_min, v2_max, size=n_samples)

    v1_bins = np.linspace(-2, 2, 10)
    v2_bins = np.linspace(v2_min, v2_max, 10)

    pred_samples = np.transpose(np.vstack([v1_pred_samples, v2_pred_samples]))
    true_samples = np.transpose(np.vstack([v1_true_samples, v2_true_samples]))
    bins = np.vstack([v1_bins, v2_bins])

    dist_ok = check_that_samples_are_from_the_same_distribution_according_to_chisq_two_sample_test(
        pred_samples, true_samples, bins
    )
    assert dist_ok


def test_rand_lognormal_and_loguniform():
    # Calculates that the

    v2_min = 5.0
    v2_max = 6.0
    n_samples = 10000

    sweep_config_2params = SweepConfig(
        {
            "method": "random",
            "parameters": {
                "v1": {"distribution": "log_normal"},
                "v2": {"min": v2_min, "max": v2_max, "distribution": "log_uniform"},
            },
        }
    )

    runs = []
    for i in range(n_samples):
        suggestion = next_run(sweep_config_2params, runs)
        runs.append(suggestion)

    v1_pred_samples = np.asarray([run.config["v1"]["value"] for run in runs])
    v2_pred_samples = np.asarray([run.config["v2"]["value"] for run in runs])

    v1_true_samples = np.random.lognormal(0, 1, size=n_samples)
    v2_true_samples = np.random.uniform(np.log(v2_min), np.log(v2_max), size=n_samples)
    v2_true_samples = np.exp(v2_true_samples)

    # the lhs needs to be >= 0 because
    v1_bins = np.linspace(0, 2, 10)
    v2_bins = np.logspace(np.log10(v2_min), np.log10(v2_max), 10)

    pred_samples = np.transpose(np.vstack([v1_pred_samples, v2_pred_samples]))
    true_samples = np.transpose(np.vstack([v1_true_samples, v2_true_samples]))
    bins = np.vstack([v1_bins, v2_bins])

    dist_ok = check_that_samples_are_from_the_same_distribution_according_to_chisq_two_sample_test(
        pred_samples, true_samples, bins
    )
    assert dist_ok
