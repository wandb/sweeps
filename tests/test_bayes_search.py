import platform

import numpy as np
import pytest

import sweeps
from sweeps.run import Run
from sweeps import bayes_search as bayes


def squiggle(x):
    return np.exp(-((x - 2) ** 2)) + np.exp(-((x - 6) ** 2) / 10) + 1 / (x ** 2 + 1)


def rosenbrock(x):
    return np.sum((x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


def test_squiggle():
    f = squiggle
    # we sample a ton of positive examples, ignoring the negative side
    X = np.array([np.random.uniform([0.0], [5.0]) for x in range(200)])
    Y = np.array([f(x) for x in X]).flatten()
    (
        sample,
        prob,
        pred,
        samples,
        vals,
        stds,
        sample_probs,
        prob_of_fail,
        pred_runtimes,
    ) = bayes.next_sample(X, Y, [[-5.0, 5.0]], improvement=1.0)
    assert sample[0] < 0.0, "Greater than 0 {}".format(sample[0])
    # we sample missing a big chunk between 1 and 3
    X = np.append(
        np.array([np.random.uniform([0.0], [1.0]) for x in range(200)]),
        np.array([np.random.uniform([0.0], [1.0]) + 4.0 for x in range(200)]),
        axis=0,
    )
    Y = np.array([f(x) for x in X]).flatten()
    (
        sample,
        prob,
        pred,
        samples,
        vals,
        stds,
        sample_probs,
        prob_of_fail,
        pred_runtimes,
    ) = bayes.next_sample(X, Y, [[0.0, 4.0]])
    assert (
        sample[0] > 1.0 and sample[0] < 4.0
    ), "Sample outside of 1-3 range: {}".format(sample[0])


def test_nans():
    X = np.array([np.random.uniform([0.0], [5.0]) for x in range(200)])
    Y = np.array([np.nan] * 200)
    (
        sample,
        prob,
        pred,
        samples,
        vals,
        stds,
        sample_probs,
        prob_of_fail,
        pred_runtimes,
    ) = bayes.next_sample(X, Y, [[-10, 10]])
    assert sample[0] < 10.0  # trying all NaNs
    X += np.array([np.random.uniform([0.0], [5.0]) for x in range(200)])
    Y += np.array([np.nan] * 200)
    (
        sample,
        prob,
        pred,
        samples,
        vals,
        stds,
        sample_probs,
        prob_of_fail,
        pred_runtimes,
    ) = bayes.next_sample(X, Y, [[-10, 10]])
    assert sample[0] < 10.0


def test_squiggle_int():
    f = squiggle
    X = np.array([np.random.uniform([0.0], [5.0]) for x in range(200)])
    Y = np.array([f(x) for x in X]).flatten()
    (
        sample,
        prob,
        pred,
        samples,
        vals,
        stds,
        sample_probs,
        prob_of_fail,
        pred_runtimes,
    ) = bayes.next_sample(X, Y, [[-10, 10]])
    assert sample[0] < 0.0, "Greater than 0 {}".format(sample[0])


def run_iterations(f, bounds, num_iterations=20):
    X = [np.zeros(len(bounds))]
    y = np.array([f(x) for x in X]).flatten()
    for jj in range(num_iterations):
        (
            sample,
            prob,
            pred,
            samples,
            vals,
            stds,
            sample_probs,
            prob_of_fail,
            pred_runtimes,
        ) = bayes.next_sample(X, y, bounds, improvement=0.1)
        print(
            "X: {} prob(I): {} pred: {} value: {}".format(sample, prob, pred, f(sample))
        )
        X = np.append(X, np.array([sample]), axis=0)
        y = np.array([f(x) for x in X]).flatten()


def run_iterations_chunked(f, bounds, num_iterations=3, chunk_size=5):
    X = [np.zeros(len(bounds))]
    y = np.array([f(x) for x in X]).flatten()
    for jj in range(num_iterations):
        sample_X = None
        for cc in range(chunk_size):
            (
                sample,
                prob,
                pred,
                samples,
                vals,
                stds,
                sample_probs,
                prob_of_fail,
                pred_runtimes,
            ) = bayes.next_sample(X, y, bounds, current_X=sample_X, improvement=0.1)
            if sample_X is None:
                sample_X = np.array([sample])
            else:
                sample_X = np.append(sample_X, np.array([sample]), axis=0)
            sample_X = np.append(X, np.array([sample]), axis=0)
        X = np.append(X, sample_X, axis=0)
        y = np.array([f(x) for x in X]).flatten()


def test_iterations_squiggle():
    run_iterations(squiggle, [[0.0, 5.0]])


def test_iterations_rosenbrock():
    dimensions = 4
    run_iterations(rosenbrock, [[0.0, 5.0]] * dimensions)


def test_iterations_squiggle_chunked():
    run_iterations_chunked(squiggle, [[0.0, 5.0]])


# search with 0 runs - hardcoded results
def test_runs_bayes(sweep_config_grid_search_2params_with_metric):
    np.random.seed(73)
    bs = sweeps.BayesianSearch()
    runs = []
    sweep = {"config": sweep_config_grid_search_2params_with_metric, "runs": runs}
    params, info = bs.next_run(sweep)
    assert params["v1"]["value"] == 7 and params["v2"]["value"] == 6


# search with 2 finished runs - hardcoded results
def test_runs_bayes_runs2(sweep_config_grid_search_2params_with_metric):
    np.random.seed(73)
    bs = sweeps.BayesianSearch()
    r1 = Run(
        "b",
        "finished",
        {"v1": {"value": 7}, "v2": {"value": 6}},
        {"zloss": 1.2},
        [
            {"loss": 1.2},
        ],
    )
    r2 = Run(
        "b", "finished", {"v1": {"value": 1}, "v2": {"value": 8}}, {"loss": 0.4}, []
    )
    # need two (non running) runs before we get a new set of parameters
    runs = [r1, r2]
    sweep = {"config": sweep_config_grid_search_2params_with_metric, "runs": runs}
    params, info = bs.next_run(sweep)
    assert params["v1"]["value"] == 2 and params["v2"]["value"] == 9


# search with 2 finished runs - hardcoded results - missing metric
def test_runs_bayes_runs2_missingmetric(sweep_config_grid_search_2params_with_metric):
    np.random.seed(73)
    bs = sweeps.BayesianSearch()
    r1 = Run(
        "b", "finished", {"v1": {"value": 7}, "v2": {"value": 5}}, {"xloss": 0.2}, []
    )
    runs = [r1, r1]
    sweep = {"config": sweep_config_grid_search_2params_with_metric, "runs": runs}
    params, info = bs.next_run(sweep)
    assert params["v1"]["value"] == 1 and params["v2"]["value"] == 1


# search with 2 finished runs - hardcoded results - missing metric
def test_runs_bayes_runs2_missingmetric_acc(sweep_config_2params_acc):
    np.random.seed(73)
    bs = sweeps.BayesianSearch()
    r1 = Run(
        "b", "finished", {"v1": {"value": 7}, "v2": {"value": 5}}, {"xloss": 0.2}, []
    )
    runs = [r1, r1]
    sweep = {"config": sweep_config_2params_acc, "runs": runs}
    params, info = bs.next_run(sweep)
    assert params["v1"]["value"] == 1 and params["v2"]["value"] == 1


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="problem with test on mac, TODO: look into this",
)
def test_runs_bayes_nan(sweep_config_grid_search_2params_with_metric):
    np.random.seed(73)
    bs = sweeps.BayesianSearch()
    r1 = Run(
        "b",
        "finished",
        {"v1": {"value": 7}, "v2": {"value": 6}},
        {},
        [
            {"loss": float("NaN")},
        ],
    )
    r2 = Run(
        "b",
        "finished",
        {"v1": {"value": 1}, "v2": {"value": 8}},
        {"loss": float("NaN")},
        [],
    )
    r3 = Run(
        "b",
        "finished",
        {"v1": {"value": 2}, "v2": {"value": 3}},
        {},
        [
            {"loss": "NaN"},
        ],
    )
    r4 = Run(
        "b", "finished", {"v1": {"value": 4}, "v2": {"value": 5}}, {"loss": "NaN"}, []
    )
    # need two (non running) runs before we get a new set of parameters
    runs = [r1, r2, r3, r4]
    sweep = {"config": sweep_config_grid_search_2params_with_metric, "runs": runs}
    params, info = bs.next_run(sweep)
    assert params["v1"]["value"] == 10 and params["v2"]["value"] == 2


def test_runs_bayes_categorical_list(sweep_config_2params_categorical):
    np.random.seed(73)
    bs = sweeps.BayesianSearch()
    r1 = Run(
        "b", "finished", {"v1": {"value": [3, 4]}, "v2": {"value": 5}}, {"acc": 0.2}, []
    )
    runs = [r1, r1]
    sweep = {"config": sweep_config_2params_categorical, "runs": runs}
    params, info = bs.next_run(sweep)
    assert (
        params["v1"]["value"] == [(7, 8), ["9", [10, 11]]]
        and params["v2"]["value"] == 1
    )
