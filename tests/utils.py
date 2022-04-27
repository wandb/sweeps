import os
from pathlib import Path

import numpy as np

from sweeps._types import ArrayLike

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
    denom = np.sqrt(err_pred**2 + err_true**2)
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
    import inspect

    import matplotlib.pyplot as plt

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


def squiggle(x: ArrayLike) -> np.floating:
    # the maximum of this 1d function is at x=2 and the minimum is at ~3.6 over the
    # interval 0-5
    return np.exp(-((x - 2) ** 2)) + np.exp(-((x - 6) ** 2) / 10) + 1 / (x**2 + 1)


def rosenbrock(x: ArrayLike) -> np.floating:
    # has a minimum at (1, 1, 1, 1, ...) for 4 <= ndim <= 7
    return np.sum((x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
