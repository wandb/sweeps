import pytest

from sweeps.config import SweepConfig
import numpy as np
from numpy import typing as npt
from sweeps.run import next_run
import os

from pathlib import Path

test_results_dir = Path(__file__).parent.parent / "test_results"
test_results_dir.mkdir(parents=True, exist_ok=True)


def check_that_samples_are_from_the_same_distribution(
    pred_samples,
    true_samples,
    bins,
):

    n_pred, _ = np.histogram(pred_samples, bins=bins)
    n_true, _ = np.histogram(true_samples, bins=bins)

    # assert the counts are equal in each bin to 1 sigma within the poisson error
    err_pred = np.sqrt(n_pred)
    err_true = np.sqrt(n_true)

    # less than 5 sigma different
    sigma_diff = np.abs(n_pred - n_true) / np.sqrt(err_pred ** 2 + err_true ** 2)
    sigma_diff[~np.isfinite(sigma_diff)] = 0

    np.testing.assert_array_less(sigma_diff, 5)


def plot_two_distributions(
    samples_true: npt.ArrayLike,
    samples_pred: npt.ArrayLike,
    bins: npt.ArrayLike,
    xscale="linear",
):
    import matplotlib.pyplot as plt
    import inspect

    fig, ax = plt.subplots()
    ax.hist(
        samples_true,
        bins=bins,
        histtype="stepfilled",
        label="true",
        alpha=0.2,
        density=True,
    )
    ax.hist(
        samples_pred,
        bins=bins,
        histtype="stepfilled",
        label="pred",
        alpha=0.2,
        density=True,
    )
    ax.legend()
    ax.set_xscale(xscale)
    ax.tick_params(which="both", axis="both", direction="in")
    current_test = os.environ.get("PYTEST_CURRENT_TEST")
    if current_test is None:
        current_test = inspect.stack()[1].function
    else:
        current_test = current_test.split(":")[-1].split(" ")[0]
    fname = f"{current_test}.pdf"
    fig.savefig(test_results_dir / fname)


def test_rand_uniform(plot):

    v1_min = 3.0
    v1_max = 5.0
    n_samples = 10000

    sweep_config_2params = SweepConfig(
        {
            "method": "random",
            "parameters": {
                "v1": {"min": v1_min, "max": v1_max},
            },
        }
    )

    runs = []
    for i in range(n_samples):
        suggestion = next_run(sweep_config_2params, runs)
        runs.append(suggestion)

    pred_samples = np.asarray([run.config["v1"]["value"] for run in runs])
    true_samples = np.random.uniform(v1_min, v1_max, size=n_samples)
    bins = np.linspace(v1_min, v1_max, 10)

    if plot:
        plot_two_distributions(true_samples, pred_samples, bins)

    check_that_samples_are_from_the_same_distribution(pred_samples, true_samples, bins)


def test_rand_normal():
    # Calculates that the

    n_samples = 10000

    sweep_config_2params = SweepConfig(
        {
            "method": "random",
            "parameters": {
                "v1": {"distribution": "normal"},
            },
        }
    )

    runs = []
    for i in range(n_samples):
        suggestion = next_run(sweep_config_2params, runs)
        runs.append(suggestion)

    pred_samples = np.asarray([run.config["v1"]["value"] for run in runs])
    true_samples = np.random.normal(0, 1, size=n_samples)
    bins = np.linspace(-2, 2, 10)

    plot_two_distributions(true_samples, pred_samples, bins)

    check_that_samples_are_from_the_same_distribution(pred_samples, true_samples, bins)


def test_rand_lognormal(plot):
    # Calculates that the

    n_samples = 10000

    sweep_config_2params = SweepConfig(
        {
            "method": "random",
            "parameters": {
                "v1": {"distribution": "log_normal", "mu": 2, "sigma": 3},
            },
        }
    )

    runs = []
    for i in range(n_samples):
        suggestion = next_run(sweep_config_2params, runs)
        runs.append(suggestion)

    pred_samples = np.asarray([run.config["v1"]["value"] for run in runs])
    true_samples = np.random.lognormal(2, 3, size=n_samples)

    bins = np.logspace(-1, 5, 30)

    if plot:
        plot_two_distributions(true_samples, pred_samples, bins, xscale="log")

    check_that_samples_are_from_the_same_distribution(pred_samples, true_samples, bins)


def test_rand_loguniform(plot):
    # Calculates that the

    v2_min = 5.0
    v2_max = 6.0
    n_samples = 10000

    sweep_config_2params = SweepConfig(
        {
            "method": "random",
            "parameters": {
                "v2": {"min": v2_min, "max": v2_max, "distribution": "log_uniform"},
            },
        }
    )

    runs = []
    for i in range(n_samples):
        suggestion = next_run(sweep_config_2params, runs)
        runs.append(suggestion)

    pred_samples = np.asarray([run.config["v2"]["value"] for run in runs])
    true_samples = np.random.uniform(np.log(v2_min), np.log(v2_max), size=n_samples)
    true_samples = np.exp(true_samples)

    # the lhs needs to be >= 0 because
    bins = np.logspace(np.log10(v2_min), np.log10(v2_max), 10)

    if plot:
        plot_two_distributions(true_samples, pred_samples, bins, xscale="log")

    check_that_samples_are_from_the_same_distribution(pred_samples, true_samples, bins)

    assert pred_samples.min() >= v2_min
    assert pred_samples.max() <= v2_max


@pytest.mark.parametrize("q", [0.1, 1, 10])
def test_rand_q_lognormal(q, plot):

    n_samples_true = 10000
    n_samples_pred = 10000
    sweep_config_2params = SweepConfig(
        {
            "method": "random",
            "parameters": {
                "v1": {"distribution": "q_log_normal", "mu": 2, "sigma": 2, "q": q},
            },
        }
    )

    runs = []
    for i in range(n_samples_pred):
        suggestion = next_run(sweep_config_2params, runs)
        runs.append(suggestion)

    pred_samples = np.asarray([run.config["v1"]["value"] for run in runs])
    true_samples = np.round(np.random.lognormal(2, 2, size=n_samples_true) / q) * q

    # need the binsize to be >> q
    bins = np.logspace(np.log10(np.exp(-2)), np.log10(np.exp(6)), 10)

    if plot:
        plot_two_distributions(true_samples, pred_samples, bins, xscale="log")

    check_that_samples_are_from_the_same_distribution(pred_samples, true_samples, bins)

    remainder = np.remainder(pred_samples, q)

    # when pred_samples == 0, pred_samples % q = q, so need to test for both remainder = q and
    # remainder = 0 under modular division
    assert np.all(np.isclose(remainder, 0) | np.isclose(remainder, q))
