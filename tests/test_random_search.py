import pytest

from ..config import SweepConfig
import numpy as np
from ..run import next_run
from .._types import ArrayLike
import os
from scipy import stats

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

    # if denom is zero, then both bins have zero counts, to set to 1 to
    # avoid division by zero error (no effect on answer)
    denom = np.sqrt(err_pred ** 2 + err_true ** 2)
    denom[np.isclose(denom, 0)] = 1
    sigma_diff = np.abs(n_pred - n_true) / denom
    sigma_diff[~np.isfinite(sigma_diff)] = 0

    np.testing.assert_array_less(sigma_diff, 5)


def plot_two_distributions(
    samples_true: ArrayLike,
    samples_pred: ArrayLike,
    bins: ArrayLike,
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
    )
    ax.hist(
        samples_pred,
        bins=bins,
        histtype="stepfilled",
        label="pred",
        alpha=0.2,
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
    n_samples = 1000

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


def test_rand_normal(plot):
    # Calculates that the

    n_samples = 1000

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

    if plot:
        plot_two_distributions(true_samples, pred_samples, bins)

    check_that_samples_are_from_the_same_distribution(pred_samples, true_samples, bins)


def test_rand_lognormal(plot):
    # Calculates that the

    n_samples = 1000

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
    v2_max = 100
    n_samples = 1000

    sweep_config_2params = SweepConfig(
        {
            "method": "random",
            "parameters": {
                "v2": {
                    "min": np.log(v2_min),
                    "max": np.log(v2_max),
                    "distribution": "log_uniform",
                },
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

    n_samples_true = 1000
    n_samples_pred = 1000
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


@pytest.mark.parametrize("q", [0.1, 1, 10])
def test_rand_q_normal(q, plot):

    n_samples_true = 1000
    n_samples_pred = 1000
    sweep_config_2params = SweepConfig(
        {
            "method": "random",
            "parameters": {
                "v1": {"distribution": "q_normal", "mu": 4, "sigma": 2, "q": q},
            },
        }
    )

    runs = []
    for i in range(n_samples_pred):
        suggestion = next_run(sweep_config_2params, runs)
        runs.append(suggestion)

    pred_samples = np.asarray([run.config["v1"]["value"] for run in runs])
    true_samples = np.round(np.random.normal(4, 2, size=n_samples_true) / q) * q

    # need the binsize to be >> q
    bins = np.linspace(0, 8, 10)

    if plot:
        plot_two_distributions(true_samples, pred_samples, bins)

    check_that_samples_are_from_the_same_distribution(pred_samples, true_samples, bins)
    remainder = np.remainder(pred_samples, q)

    # when pred_samples == 0, pred_samples % q = q, so need to test for both remainder = q and
    # remainder = 0 under modular division
    assert np.all(np.isclose(remainder, 0) | np.isclose(remainder, q))


@pytest.mark.parametrize("q", [0.1, 1, 10])
def test_rand_q_uniform(q, plot):

    n_samples_true = 1000
    n_samples_pred = 1000
    sweep_config_2params = SweepConfig(
        {
            "method": "random",
            "parameters": {
                "v1": {"distribution": "q_uniform", "min": 0, "max": 100, "q": q},
            },
        }
    )

    runs = []
    for i in range(n_samples_pred):
        suggestion = next_run(sweep_config_2params, runs)
        runs.append(suggestion)

    pred_samples = np.asarray([run.config["v1"]["value"] for run in runs])
    true_samples = np.round(np.random.uniform(0, 100, size=n_samples_true) / q) * q

    # need the binsize to be >> q
    bins = np.linspace(0, 100, 10)

    if plot:
        plot_two_distributions(true_samples, pred_samples, bins)

    check_that_samples_are_from_the_same_distribution(pred_samples, true_samples, bins)
    remainder = np.remainder(pred_samples, q)

    # when pred_samples == 0, pred_samples % q = q, so need to test for both remainder = q and
    # remainder = 0 under modular division
    assert np.all(np.isclose(remainder, 0) | np.isclose(remainder, q))


@pytest.mark.parametrize("q", [0.1, 1, 10])
def test_rand_q_loguniform(q, plot):

    n_samples_pred = 1000
    sweep_config_2params = SweepConfig(
        {
            "method": "random",
            "parameters": {
                "v1": {
                    "distribution": "q_log_uniform",
                    "min": np.log(0.1),
                    "max": np.log(100),
                    "q": q,
                },
            },
        }
    )

    runs = []
    for i in range(n_samples_pred):
        suggestion = next_run(sweep_config_2params, runs)
        runs.append(suggestion)

    pred_samples = np.asarray([run.config["v1"]["value"] for run in runs])
    true_samples = np.round(stats.loguniform(0.1, 100).rvs(1000) / q) * q

    # need the binsize to be >> q
    bins = np.logspace(-1, 2, 10)

    if plot:
        plot_two_distributions(true_samples, pred_samples, bins, xscale="log")

    check_that_samples_are_from_the_same_distribution(pred_samples, true_samples, bins)
    remainder = np.remainder(pred_samples, q)

    # when pred_samples == 0, pred_samples % q = q, so need to test for both remainder = q and
    # remainder = 0 under modular division
    assert np.all(np.isclose(remainder, 0) | np.isclose(remainder, q))


@pytest.mark.parametrize("q", [0.1])
def test_rand_q_beta(q, plot):

    n_samples_pred = 1000
    sweep_config_2params = SweepConfig(
        {
            "method": "random",
            "parameters": {
                "v1": {"distribution": "q_beta", "a": 2, "b": 5, "q": q},
            },
        }
    )

    runs = []
    for i in range(n_samples_pred):
        suggestion = next_run(sweep_config_2params, runs)
        runs.append(suggestion)

    pred_samples = np.asarray([run.config["v1"]["value"] for run in runs])
    true_samples = np.round(np.random.beta(2, 5, 1000) / q) * q

    # need the binsize to be >> q
    bins = np.linspace(0, 1, 10)

    if plot:
        plot_two_distributions(true_samples, pred_samples, bins)

    check_that_samples_are_from_the_same_distribution(pred_samples, true_samples, bins)
    remainder = np.remainder(pred_samples, q)

    # when pred_samples == 0, pred_samples % q = q, so need to test for both remainder = q and
    # remainder = 0 under modular division
    assert np.all(np.isclose(remainder, 0) | np.isclose(remainder, q))


def test_rand_beta(plot):

    n_samples_pred = 1000
    sweep_config_2params = SweepConfig(
        {
            "method": "random",
            "parameters": {
                "v1": {"distribution": "beta", "a": 2, "b": 5},
            },
        }
    )

    runs = []
    for i in range(n_samples_pred):
        suggestion = next_run(sweep_config_2params, runs)
        runs.append(suggestion)

    pred_samples = np.asarray([run.config["v1"]["value"] for run in runs])
    true_samples = np.random.beta(2, 5, 1000)

    # need the binsize to be >> q
    bins = np.linspace(0, 1, 10)

    if plot:
        plot_two_distributions(true_samples, pred_samples, bins)

    check_that_samples_are_from_the_same_distribution(pred_samples, true_samples, bins)
