import os
import json
from typing import Callable, Optional, Tuple, Iterable, Dict, Union

import pytest
import numpy as np

from .. import bayes_search as bayes
from .._types import integer, floating, ArrayLike
from .. import SweepRun, RunState, next_run, SweepConfig

from .test_random_search import check_that_samples_are_from_the_same_distribution


def squiggle(x: ArrayLike) -> np.floating:
    # the maximum of this 1d function is at x=2 and the minimum is at ~3.6 over the
    # interval 0-5
    return np.exp(-((x - 2) ** 2)) + np.exp(-((x - 6) ** 2) / 10) + 1 / (x ** 2 + 1)


def rosenbrock(x: ArrayLike) -> np.floating:
    # has a minimum at (1, 1, 1, 1, ...) for 4 <= ndim <= 7
    return np.sum((x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


def run_bayes_search(
    f: Callable[[SweepRun], floating],
    config: SweepConfig,
    init_runs: Iterable[SweepRun] = (),
    improvement: floating = 0.1,
    num_iterations: integer = 20,
    optimium: Optional[Dict[str, floating]] = None,
    atol: Optional[Union[Dict[str, floating], floating]] = 0.2,
):

    metric_name = config["metric"]["name"]
    opt_goal = config["metric"]["goal"]

    runs = list(init_runs)
    for _ in range(num_iterations):
        suggested_run = bayes.bayes_search_next_run(
            runs, config, minimum_improvement=improvement
        )
        suggested_run.state = RunState.finished
        metric = f(suggested_run)
        if suggested_run.summary_metrics is None:
            suggested_run.summary_metrics = {}  # pragma: no cover
        suggested_run.summary_metrics[metric_name] = metric
        runs.append(suggested_run)

    if optimium is not None:
        best_run = (min if opt_goal == "minimize" else max)(
            runs, key=lambda run: run.metric_extremum(metric_name, opt_goal)
        )
        for param_name in config["parameters"]:
            left_comp = best_run.config[param_name]["value"]
            right_comp = (
                optimium if isinstance(optimium, float) else optimium[param_name]
            )
            np.testing.assert_allclose(left_comp, right_comp, atol=atol)


@pytest.mark.parametrize(
    "x",
    [
        {"distribution": "normal", "mu": 2, "sigma": 4},
        {"distribution": "log_uniform", "min": -2, "max": 3},
        {"min": 0.0, "max": 5.0},
        {"min": 0, "max": 5},
        {"distribution": "q_uniform", "min": 0.0, "max": 10.0, "q": 0.25},
    ],
)
def test_squiggle_convergence_full(x):
    def y(x: SweepRun) -> floating:
        return squiggle(x.config["x"]["value"])

    run = SweepRun(
        config={"x": {"value": np.random.uniform(0, 5)}}, state=RunState.finished
    )
    run.summary_metrics["y"] = y(run)

    runs = [run]

    config = SweepConfig(
        {
            "method": "bayes",
            "metric": {"name": "y", "goal": "maximize"},
            "parameters": {"x": x},
        }
    )

    run_bayes_search(y, config, runs, num_iterations=200, optimium={"x": 2.0})


def run_iterations(
    f: Callable[[ArrayLike], floating],
    bounds: ArrayLike,
    num_iterations: integer = 20,
    x_init: Optional[ArrayLike] = None,
    improvement: floating = 0.1,
    optimium: Optional[ArrayLike] = None,
    atol: Optional[ArrayLike] = 0.2,
    chunk_size: integer = 1,
) -> Tuple[ArrayLike, ArrayLike]:

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
            (sample, prob, pred, _, _,) = bayes.next_sample(
                sample_X=X,
                sample_y=y,
                X_bounds=bounds,
                current_X=sample_X,
                improvement=improvement,
            )
            if sample_X is None:
                sample_X = np.array([sample])
            else:
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
    (sample, prob, pred, _, _,) = bayes.next_sample(
        sample_X=X, sample_y=Y, X_bounds=[[-5.0, 5.0]], improvement=1.0
    )
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
        _,
        _,
    ) = bayes.next_sample(sample_X=X, sample_y=Y, X_bounds=[[0.0, 5.0]])
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


def test_squiggle_int():
    f = squiggle
    X = np.random.uniform(0, 5, 200)[:, None]
    new_X, new_y = run_iterations(f, [[-10, 10]], 1, X)
    sample = new_X[-1][0]
    assert sample < 0.0, "Greater than 0 {}".format(sample)
    assert np.isclose(sample % 1, 0)


def test_iterations_rosenbrock():
    dimensions = 3
    # x_init = np.random.uniform(0, 2, size=(1, dimensions))
    x_init = np.zeros((1, dimensions))
    run_iterations(
        rosenbrock,
        [[0.0, 2.0]] * dimensions,
        300,
        x_init,
        optimium=[1, 1, 1],
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
def test_runs_bayes_runs2_missingmetric():

    config = SweepConfig(
        {
            "metric": {"name": "loss", "goal": "minimize"},
            "method": "bayes",
            "parameters": {
                "v2": {"min": 1, "max": 10},
            },
        }
    )

    r1 = SweepRun(
        name="b",
        state=RunState.finished,
        history=[
            {"xloss": 5.0},
        ],
        config={"v2": {"value": 6}},
        summary_metrics={"zloss": 1.2},
    )
    r2 = SweepRun(
        name="b2",
        state=RunState.finished,
        config={"v2": {"value": 8}},
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


def test_runs_bayes_nan(sweep_config_bayes_search_2params_with_metric):
    r1 = SweepRun(
        name="b",
        state=RunState.finished,
        config={"v1": {"value": 7}, "v2": {"value": 6}},
        summary_metrics={},
        history=[
            {"loss": float("NaN")},
        ],
    )
    r2 = SweepRun(
        name="b",
        state=RunState.finished,
        config={"v1": {"value": 1}, "v2": {"value": 8}},
        summary_metrics={"loss": float("NaN")},
        history=[],
    )
    r3 = SweepRun(
        name="b",
        state=RunState.finished,
        config={"v1": {"value": 2}, "v2": {"value": 3}},
        summary_metrics={},
        history=[
            {"loss": "NaN"},
        ],
    )
    r4 = SweepRun(
        name="b",
        state=RunState.finished,
        config={"v1": {"value": 4}, "v2": {"value": 5}},
        summary_metrics={"loss": "NaN"},
        history=[],
    )
    # need two (non running) runs before we get a new set of parameters
    runs = [r1, r2, r3, r4]

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


def test_runs_bayes_categorical_list():

    v2_min = 1
    v2_max = 10

    config = {
        "method": "bayes",
        "metric": {
            "name": "acc",
            "goal": "maximize",
        },
        "parameters": {
            "v1": {
                "distribution": "categorical",
                "values": [(2, 3), [3, 4], ["5", "6"], [(7, 8), ["9", [10, 11]]]],
            },
            "v2": {"min": v2_min, "max": v2_max},
        },
    }

    def loss_func(x: SweepRun) -> floating:
        v2_acc = 0.5 * (x.config["v2"]["value"] - v2_min) / (v2_max - v2_min)
        v1_acc = [0.1, 0.2, 0.5, 0.1][
            config["parameters"]["v1"]["values"].index(x.config["v1"]["value"])
        ]
        return v1_acc + v2_acc

    r1 = SweepRun(
        name="b",
        state=RunState.finished,
        config={"v1": {"value": [3, 4]}, "v2": {"value": 5}},
        history=[],
    )
    r1.summary_metrics = {"acc": loss_func(r1)}

    r2 = SweepRun(
        name="b",
        state=RunState.finished,
        config={"v1": {"value": (2, 3)}, "v2": {"value": 5}},
        history=[],
    )
    r2.summary_metrics = {"acc": loss_func(r2)}

    runs = [r1, r2]
    for _ in range(200):
        suggestion = next_run(config, runs)
        metric = {"acc": loss_func(suggestion)}
        suggestion.history = [metric]
        suggestion.state = RunState.finished
        runs.append(suggestion)

    best_run = max(runs, key=lambda r: r.metric_extremum("acc", "maximum"))
    best_x = [best_run.config["v1"]["value"], best_run.config["v2"]["value"]]

    assert best_x[0] == ["5", "6"]
    assert np.abs(best_x[1] - 10) < 0.2


def test_bayes_can_handle_preemptible_or_preempting_runs():

    v2_min = 1
    v2_max = 10

    config = {
        "method": "bayes",
        "metric": {
            "name": "acc",
            "goal": "maximize",
        },
        "parameters": {
            "v1": {
                "distribution": "categorical",
                "values": [(2, 3), [3, 4], ["5", "6"], [(7, 8), ["9", [10, 11]]]],
            },
            "v2": {"min": v2_min, "max": v2_max},
        },
    }

    def loss_func(x: SweepRun) -> floating:
        v2_acc = 0.5 * (x.config["v2"]["value"] - v2_min) / (v2_max - v2_min)
        v1_acc = [0.1, 0.2, 0.5, 0.1][
            config["parameters"]["v1"]["values"].index(x.config["v1"]["value"])
        ]
        return v1_acc + v2_acc

    r1 = SweepRun(
        name="b",
        state=RunState.preempted,
        config={"v1": {"value": [3, 4]}, "v2": {"value": 5}},
        history=[],
    )
    r1.summary_metrics = {"acc": loss_func(r1)}

    # this should not raise, and should produce the same result as if r1.state was running
    seed = np.random.get_state()
    pred = next_run(config, [r1])
    r1.state = RunState.running
    np.random.set_state(seed)
    true = next_run(config, [r1])
    assert pred.config == true.config

    r2 = SweepRun(
        name="b",
        state=RunState.preempting,
        config={"v1": {"value": (2, 3)}, "v2": {"value": 5}},
        history=[],
    )
    r2.summary_metrics = {"acc": loss_func(r2)}

    seed = np.random.get_state()
    pred = next_run(config, [r2])
    r2.state = RunState.running
    np.random.set_state(seed)
    true = next_run(config, [r2])
    assert pred.config == true.config


def test_that_constant_parameters_are_sampled_correctly():
    # from https://console.cloud.google.com/logs/query;cursorTimestamp=2021-07-07T01:02:35.636458264Z;pinnedLogId=2021-07-07T01:02:35.636458264Z%2Fxl7xiuf6empym;query=jsonPayload.message%3D%22Mismatch%20detected%20in%20shadow%20sweep%20provider%22%0AjsonPayload.data.config%20!~%20%22%5C%22method%5C%22:%20%5C%22grid%5C%22%22%0Atimestamp%3D%222021-07-07T01:02:35.636458264Z%22%0AinsertId%3D%22xl7xiuf6empym%22%0Atimestamp%3D%222021-07-07T01:02:35.636458264Z%22%0AinsertId%3D%22xl7xiuf6empym%22?project=wandb-production
    config = {
        "name": "Elastic Sweep 07-06 1027",
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "IC_plus_Hitrate_risk_adj"},
        "parameters": {
            "tol": {"values": [1e-05, 0.0001, 0.0002, 2.5e-05, 5e-05, 7.5e-05]},
            "alpha": {
                "max": 6.907755278982137,
                "min": -9.210340371976182,
                "distribution": "log_uniform",
            },
            "model": {"value": "ElasticNet"},
            "target": {"values": ["1D_spec", "5D_spec", "10D_spec"]},
            "l1_ratio": {"max": 1, "min": 0, "distribution": "uniform"},
            "selection": {"values": ["cyclic", "random"]},
            "decay_days": {"values": [1, 2, 3, 4, 5, 6]},
            "cat_columns": {"value": ["Sector", "Industry", "market_simple_regime"]},
            "num_columns": {
                "value": [
                    "sales_a1.median",
                    "sales_a1.std_dev",
                    "sales_a1.median_slope",
                    "ebit_a1.median",
                    "ebit_a1.std_dev",
                    "ebit_a1.median_slope",
                    "Log Market Cap",
                    "option_skew",
                    "option_3m_90_vol",
                    "option_3m_105_vol",
                ]
            },
            "ppf_target_by": {"values": [None, "Sector", "Industry"]},
            "ppf_the_target": {"value": True},
        },
    }

    null = None
    true = True
    false = False
    runs = [
        {
            "name": "cv4uof8s",
            "config": {
                "tol": {"desc": null, "value": 7.5e-05},
                "alpha": {"desc": null, "value": 114.94621156856498},
                "model": {"desc": null, "value": "ElasticNet"},
                "_wandb": {
                    "desc": null,
                    "value": {
                        "t": {
                            "1": [3, 5],
                            "2": [3, 5],
                            "3": [2],
                            "4": "3.7.9",
                            "5": "0.10.31",
                            "8": [1, 5],
                        },
                        "framework": "tensorflow",
                        "cli_version": "0.10.31",
                        "is_jupyter_run": true,
                        "python_version": "3.7.9",
                        "is_kaggle_kernel": false,
                    },
                },
                "target": {"desc": null, "value": "1D_spec"},
                "l1_ratio": {"desc": null, "value": 0.7475487707624942},
                "selection": {"desc": null, "value": "random"},
                "decay_days": {"desc": null, "value": 4},
                "cat_columns": {
                    "desc": null,
                    "value": ["Sector", "Industry", "market_simple_regime"],
                },
                "num_columns": {
                    "desc": null,
                    "value": [
                        "sales_a1.median",
                        "sales_a1.std_dev",
                        "sales_a1.median_slope",
                        "ebit_a1.median",
                        "ebit_a1.std_dev",
                        "ebit_a1.median_slope",
                        "Log Market Cap",
                        "option_skew",
                        "option_3m_90_vol",
                        "option_3m_105_vol",
                    ],
                },
                "ppf_target_by": {"desc": null, "value": "Industry"},
                "ppf_the_target": {"desc": null, "value": true},
            },
            "summaryMetrics": {},
            "state": "failed",
        },
        {
            "config": {
                "tol": {"desc": null, "value": 7.5e-05},
                "alpha": {"desc": null, "value": 0.07239397556453357},
                "model": {"desc": null, "value": "ElasticNet"},
                "_wandb": {
                    "desc": null,
                    "value": {
                        "t": {
                            "1": [3, 5],
                            "2": [3, 5],
                            "3": [2],
                            "4": "3.7.9",
                            "5": "0.10.31",
                            "8": [1, 5],
                        },
                        "framework": "tensorflow",
                        "cli_version": "0.10.31",
                        "is_jupyter_run": true,
                        "python_version": "3.7.9",
                        "is_kaggle_kernel": false,
                    },
                },
                "target": {"desc": null, "value": "1D_spec"},
                "l1_ratio": {"desc": null, "value": 0.9317758724316296},
                "selection": {"desc": null, "value": "random"},
                "decay_days": {"desc": null, "value": 6},
                "cat_columns": {
                    "desc": null,
                    "value": ["Sector", "Industry", "market_simple_regime"],
                },
                "num_columns": {
                    "desc": null,
                    "value": [
                        "sales_a1.median",
                        "sales_a1.std_dev",
                        "sales_a1.median_slope",
                        "ebit_a1.median",
                        "ebit_a1.std_dev",
                        "ebit_a1.median_slope",
                        "Log Market Cap",
                        "option_skew",
                        "option_3m_90_vol",
                        "option_3m_105_vol",
                    ],
                },
                "ppf_target_by": {"desc": null, "value": null},
                "ppf_the_target": {"desc": null, "value": true},
            },
            "summaryMetrics": {},
            "state": "failed",
            "name": "633fjdhd",
        },
        {
            "name": "77ywpu3j",
            "summaryMetrics": {},
            "state": "failed",
            "config": {
                "tol": {"desc": null, "value": 5e-05},
                "alpha": {"desc": null, "value": 0.002774852781906077},
                "model": {"desc": null, "value": "ElasticNet"},
                "_wandb": {
                    "desc": null,
                    "value": {
                        "t": {
                            "1": [3, 5],
                            "2": [3, 5],
                            "3": [2],
                            "4": "3.7.9",
                            "5": "0.10.31",
                            "8": [1, 5],
                        },
                        "framework": "tensorflow",
                        "cli_version": "0.10.31",
                        "is_jupyter_run": true,
                        "python_version": "3.7.9",
                        "is_kaggle_kernel": false,
                    },
                },
                "target": {"desc": null, "value": "10D_spec"},
                "l1_ratio": {"desc": null, "value": 0.01483000099874121},
                "selection": {"desc": null, "value": "cyclic"},
                "decay_days": {"desc": null, "value": 6},
                "cat_columns": {
                    "desc": null,
                    "value": ["Sector", "Industry", "market_simple_regime"],
                },
                "num_columns": {
                    "desc": null,
                    "value": [
                        "sales_a1.median",
                        "sales_a1.std_dev",
                        "sales_a1.median_slope",
                        "ebit_a1.median",
                        "ebit_a1.std_dev",
                        "ebit_a1.median_slope",
                        "Log Market Cap",
                        "option_skew",
                        "option_3m_90_vol",
                        "option_3m_105_vol",
                    ],
                },
                "ppf_target_by": {"desc": null, "value": null},
                "ppf_the_target": {"desc": null, "value": true},
            },
        },
        {
            "name": "06z3nutf",
            "summaryMetrics": {},
            "config": {
                "tol": {"desc": null, "value": 0.0001},
                "alpha": {"desc": null, "value": 0.005228908948292734},
                "model": {"desc": null, "value": "ElasticNet"},
                "_wandb": {
                    "desc": null,
                    "value": {
                        "t": {
                            "1": [3, 5],
                            "2": [3, 5],
                            "3": [2],
                            "4": "3.7.9",
                            "5": "0.10.31",
                            "8": [1, 5],
                        },
                        "framework": "tensorflow",
                        "cli_version": "0.10.31",
                        "is_jupyter_run": true,
                        "python_version": "3.7.9",
                        "is_kaggle_kernel": false,
                    },
                },
                "target": {"desc": null, "value": "1D_spec"},
                "l1_ratio": {"desc": null, "value": 0.974230005151591},
                "selection": {"desc": null, "value": "random"},
                "decay_days": {"desc": null, "value": 1},
                "cat_columns": {
                    "desc": null,
                    "value": ["Sector", "Industry", "market_simple_regime"],
                },
                "num_columns": {
                    "desc": null,
                    "value": [
                        "sales_a1.median",
                        "sales_a1.std_dev",
                        "sales_a1.median_slope",
                        "ebit_a1.median",
                        "ebit_a1.std_dev",
                        "ebit_a1.median_slope",
                        "Log Market Cap",
                        "option_skew",
                        "option_3m_90_vol",
                        "option_3m_105_vol",
                    ],
                },
                "ppf_target_by": {"desc": null, "value": null},
                "ppf_the_target": {"desc": null, "value": true},
            },
            "state": "failed",
        },
        {
            "summaryMetrics": {},
            "config": {
                "tol": {"desc": null, "value": 0.0001},
                "alpha": {"desc": null, "value": 2.2736520853608524},
                "model": {"desc": null, "value": "ElasticNet"},
                "_wandb": {
                    "desc": null,
                    "value": {
                        "t": {
                            "1": [3, 5],
                            "2": [3, 5],
                            "3": [2],
                            "4": "3.7.9",
                            "5": "0.10.31",
                            "8": [1, 5],
                        },
                        "framework": "tensorflow",
                        "cli_version": "0.10.31",
                        "is_jupyter_run": true,
                        "python_version": "3.7.9",
                        "is_kaggle_kernel": false,
                    },
                },
                "target": {"desc": null, "value": "1D_spec"},
                "l1_ratio": {"desc": null, "value": 0.6713230851920869},
                "selection": {"desc": null, "value": "random"},
                "decay_days": {"desc": null, "value": 1},
                "cat_columns": {
                    "desc": null,
                    "value": ["Sector", "Industry", "market_simple_regime"],
                },
                "num_columns": {
                    "desc": null,
                    "value": [
                        "sales_a1.median",
                        "sales_a1.std_dev",
                        "sales_a1.median_slope",
                        "ebit_a1.median",
                        "ebit_a1.std_dev",
                        "ebit_a1.median_slope",
                        "Log Market Cap",
                        "option_skew",
                        "option_3m_90_vol",
                        "option_3m_105_vol",
                    ],
                },
                "ppf_target_by": {"desc": null, "value": "Industry"},
                "ppf_the_target": {"desc": null, "value": true},
            },
            "state": "failed",
            "name": "1ddlc02y",
        },
    ]

    suggestion = next_run(config, [SweepRun(**run) for run in runs])
    for key in suggestion.config:
        if key in config["parameters"] and key != "ppf_target_by":
            assert suggestion.config[key]["value"] is not None


def test_metric_extremum_in_bayes_search():
    # from https://console.cloud.google.com/logs/query;query=ygnwe8ptupj33get%0A;timeRange=2021-08-03T21:34:50.082Z%2F2021-08-03T21:34:59.082Z;summaryFields=:false:32:beginning;cursorTimestamp=2021-08-03T21:34:51.189649752Z?project=wandb-production
    data_path = f"{os.path.dirname(__file__)}/data/ygnwe8ptupj33get.decoded.json"
    with open(data_path, "r") as f:
        data = json.load(f)
    _, _, _, y = bayes._construct_gp_data(
        [SweepRun(**r) for r in data["jsonPayload"]["data"]["runs"]],
        data["jsonPayload"]["data"]["config"],
    )
    np.testing.assert_array_less(np.abs(y + 98), 5)
