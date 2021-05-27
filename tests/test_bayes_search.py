from typing import Callable, Optional, Tuple

import numpy as np
import numpy.typing as npt

from sweeps import bayes_search as bayes
from sweeps.types import integer, floating
from sweeps import SweepRun, RunState, next_run, SweepConfig

from .test_random_search import check_that_samples_are_from_the_same_distribution


def squiggle(x: npt.ArrayLike) -> np.floating:
    # the maximum of this 1d function is at x=2 and the minimum is at ~3.6 over the
    # interval 0-5
    return np.exp(-((x - 2) ** 2)) + np.exp(-((x - 6) ** 2) / 10) + 1 / (x ** 2 + 1)


def rosenbrock(x: npt.ArrayLike) -> np.floating:
    # has a minimum at (1, 1, 1, 1, ...) for 4 <= ndim <= 7
    return np.sum((x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


def run_iterations(
    f: Callable[[npt.ArrayLike], floating],
    bounds: npt.ArrayLike,
    num_iterations: integer = 20,
    x_init: Optional[npt.ArrayLike] = None,
    improvement: floating = 0.1,
    optimium: Optional[npt.ArrayLike] = None,
    atol: Optional[npt.ArrayLike] = 0.2,
    chunk_size: integer = 1,
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:

    if x_init is not None:
        X = x_init
    else:
        X = [np.zeros(len(bounds))]

    y = np.array([f(x) for x in X]).flatten()

    counter = 0
    for jj in range(int(np.ceil(num_iterations / chunk_size))):
        sample_X = None
        for cc in range(chunk_size):
            if counter >= num_iterations:
                break
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
            ) = bayes.next_sample(
                X, y, bounds, current_X=sample_X, improvement=improvement
            )
            if sample_X is None:
                sample_X = np.array([sample])
            sample_X = np.append(sample_X, np.array([sample]), axis=0)
            counter += 1
            print(
                "X: {} prob(I): {} pred: {} value: {}".format(
                    sample, prob, pred, f(sample)
                )
            )

        X = np.append(X, sample_X, axis=0)
        print(X[np.argmin(y)])
        y = np.array([f(x) for x in X]).flatten()

    if optimium is not None:
        optimium = np.asarray(optimium)
        representative_sample = X[np.argmin(y)]
        np.testing.assert_array_less(np.abs(representative_sample - optimium), atol)

    return X, y


def test_squiggle_explores_parameter_space():
    # This test checks whether the bayes algorithm correctly explores the parameter space
    # we sample a ton of positive examples, ignoring the negative side
    X = np.random.uniform(0, 5, 200)[:, None]
    Y = squiggle(X.ravel())
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
    X = np.vstack(
        (np.random.uniform(0, 1, 200)[:, None], np.random.uniform(4, 5, 200)[:, None])
    )
    Y = squiggle(X.ravel())
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
    ) = bayes.next_sample(X, Y, [[0.0, 5.0]])
    assert (
        sample[0] > 1.0 and sample[0] < 4.0
    ), "Sample outside of 1-3 range: {}".format(sample[0])


def test_squiggle_convergence():
    x_init = np.random.uniform(0, 5, 1)[:, None]
    run_iterations(squiggle, [[0.0, 5.0]], 200, x_init, optimium=[3.6], atol=0.2)


def test_squiggle_convergence_to_maximum():
    # This test checks whether the bayes algorithm correctly explores the parameter space
    # we sample a ton of positive examples, ignoring the negative side
    def f(x):
        return -squiggle(x)

    x_init = np.random.uniform(0, 5, 1)[:, None]
    run_iterations(f, [[0.0, 5.0]], 200, x_init, optimium=[2], atol=0.2)


def test_nans():
    def f(x):
        return np.zeros_like(x) * np.nan

    X = np.random.uniform(0, 5, 200)[:, None]
    new_x, new_y = run_iterations(f, [[-10, 10]], 1, X)
    assert new_x[-1][0] < 10.0
    assert np.isnan(new_y[-1])
    new_x += np.random.uniform(0, 5, len(new_x))[:, None]
    new_x, new_y = run_iterations(f, [[-10, 10]], 1, X)
    assert new_x[-1][0] < 10.0
    assert np.isnan(new_y[-1])
    print(new_x)


def test_squiggle_int():
    f = squiggle
    X = np.random.uniform(0, 5, 200)[:, None]
    new_X, new_y = run_iterations(f, [[-10, 10]], 1, X)
    sample = new_X[-1][0]
    assert sample < 0.0, "Greater than 0 {}".format(sample)
    assert np.isclose(sample % 1, 0)


def test_iterations_rosenbrock():
    dimensions = 4
    # x_init = np.random.uniform(0, 2, size=(1, dimensions))
    x_init = np.zeros((1, dimensions))
    run_iterations(
        rosenbrock,
        [[0.0, 2.0]] * dimensions,
        200,
        x_init,
        optimium=[1, 1, 1, 1],
        atol=0.2,
        improvement=0.1,
    )


def test_iterations_squiggle_chunked():
    run_iterations(
        squiggle,
        [[0.0, 5.0]],
        chunk_size=5,
        num_iterations=200,
        optimium=[3.6],
        improvement=0.1,
    )


def test_bayes_search_with_zero_runs_begins_correctly(
    sweep_config_bayes_search_2params_with_metric,
):
    run = next_run(sweep_config_bayes_search_2params_with_metric, [])
    assert isinstance(run.config["v1"]["value"], int) and isinstance(
        run.config["v2"]["value"], float
    )
    v1 = sweep_config_bayes_search_2params_with_metric["parameters"]["v1"]
    v1_min, v1_max = v1["min"], v1["max"]
    v2 = sweep_config_bayes_search_2params_with_metric["parameters"]["v2"]
    v2_min, v2_max = v2["min"], v2["max"]
    assert v1_min <= run.config["v1"]["value"] <= v1_max
    assert v2_min <= run.config["v2"]["value"] <= v2_max


# search with 2 finished runs - hardcoded results
def test_runs_bayes_runs2(sweep_config_bayes_search_2params_with_metric):
    def loss_func(run: SweepRun) -> floating:
        return (run.config["v1"]["value"] - 5) ** 2 + (
            run.config["v2"]["value"] - 2
        ) ** 2

    r1 = SweepRun(
        name="b",
        state=RunState.finished,
        history=[
            {"loss": 5.0},
        ],
        config={"v1": {"value": 7}, "v2": {"value": 6}},
        summary_metrics={"zloss": 1.2},
    )
    r2 = SweepRun(
        name="b2",
        state=RunState.finished,
        config={"v1": {"value": 1}, "v2": {"value": 8}},
        summary_metrics={"loss": 52.0},
        history=[],
    )
    # need two (non running) runs before we get a new set of parameters

    runs = [r1, r2]

    for _ in range(200):
        suggestion = next_run(sweep_config_bayes_search_2params_with_metric, runs)
        metric = {"loss": loss_func(suggestion)}
        suggestion.history = [metric]
        suggestion.state = RunState.finished
        runs.append(suggestion)

    best_run = min(runs, key=lambda r: r.metric_extremum("loss", "minimum"))
    best_x = np.asarray(
        [best_run.config["v1"]["value"], best_run.config["v2"]["value"]]
    )
    optimum = np.asarray([5.0, 2.0])
    np.testing.assert_array_less(np.abs(best_x - optimum), 0.2)


# search with 2 finished runs - hardcoded results - missing metric
def test_runs_bayes_runs2_missingmetric(sweep_config_bayes_search_2params_with_metric):

    r1 = SweepRun(
        name="b",
        state=RunState.finished,
        history=[
            {"xloss": 5.0},
        ],
        config={"v1": {"value": 7}, "v2": {"value": 6}},
        summary_metrics={"zloss": 1.2},
    )
    r2 = SweepRun(
        name="b2",
        state=RunState.finished,
        config={"v1": {"value": 1}, "v2": {"value": 8}},
        summary_metrics={"xloss": 52.0},
        history=[],
    )

    runs = [r1, r2]
    for _ in range(200):
        suggestion = next_run(sweep_config_bayes_search_2params_with_metric, runs)
        suggestion.state = RunState.finished
        runs.append(suggestion)

    # should just choose random runs in this case as they are all imputed with the same value (zero)
    # for the loss function
    check_that_samples_are_from_the_same_distribution(
        [run.config["v2"]["value"] for run in runs],
        np.random.uniform(1, 10, 202),
        np.linspace(1, 10, 11),
    )


# search with 2 finished runs - hardcoded results - missing metric
def test_runs_bayes_runs2_missingmetric_acc():

    config = SweepConfig(
        {
            "method": "bayes",
            "metric": {
                "name": "acc",
                "goal": "maximize",
            },
            "parameters": {"v1": {"min": 1, "max": 10}, "v2": {"min": 1, "max": 10}},
        }
    )

    r1 = SweepRun(
        name="b",
        state=RunState.finished,
        history=[
            {"xloss": 5.0},
        ],
        config={"v1": {"value": 7}, "v2": {"value": 6}},
        summary_metrics={"zloss": 1.2},
    )
    r2 = SweepRun(
        name="b2",
        state=RunState.finished,
        config={"v1": {"value": 1}, "v2": {"value": 8}},
        summary_metrics={"xloss": 52.0},
        history=[],
    )

    runs = [r1, r2]
    for _ in range(200):
        suggestion = next_run(config, runs)
        suggestion.state = RunState.finished
        runs.append(suggestion)

    # should just choose random runs in this case as they are all imputed with the same value (zero)
    # for the loss function
    check_that_samples_are_from_the_same_distribution(
        [run.config["v2"]["value"] for run in runs],
        np.random.uniform(1, 10, 202),
        np.linspace(1, 10, 11),
    )


"""
@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="problem with test on mac, TODO: look into this",
)
def test_runs_bayes_nan(sweep_config_bayes_search_2params_with_metric):
    np.random.seed(73)
    bs = sweeps.BayesianSearch()
    r1 = SweepRun(
        "b",
        "finished",
        {"v1": {"value": 7}, "v2": {"value": 6}},
        {},
        [
            {"loss": float("NaN")},
        ],
    )
    r2 = SweepRun(
        "b",
        "finished",
        {"v1": {"value": 1}, "v2": {"value": 8}},
        {"loss": float("NaN")},
        [],
    )
    r3 = SweepRun(
        "b",
        "finished",
        {"v1": {"value": 2}, "v2": {"value": 3}},
        {},
        [
            {"loss": "NaN"},
        ],
    )
    r4 = SweepRun(
        "b", "finished", {"v1": {"value": 4}, "v2": {"value": 5}}, {"loss": "NaN"}, []
    )
    # need two (non running) runs before we get a new set of parameters
    runs = [r1, r2, r3, r4]
    sweep = {"config": sweep_config_bayes_search_2params_with_metric, "runs": runs}
    params, info = bs.next_run(sweep)
    assert params["v1"]["value"] == 10 and params["v2"]["value"] == 2


def test_runs_bayes_categorical_list(sweep_config_2params_categorical):
    np.random.seed(73)
    bs = sweeps.BayesianSearch()
    r1 = SweepRun(
        "b", "finished", {"v1": {"value": [3, 4]}, "v2": {"value": 5}}, {"acc": 0.2}, []
    )
    runs = [r1, r1]
    sweep = {"config": sweep_config_2params_categorical, "runs": runs}
    params, info = bs.next_run(sweep)
    assert (
        params["v1"]["value"] == [(7, 8), ["9", [10, 11]]]
        and params["v2"]["value"] == 1
    )
"""
