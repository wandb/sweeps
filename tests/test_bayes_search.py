import json
import logging
import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pytest
from sweeps import RunState, SweepConfig, SweepRun
from sweeps import bayes_search as bayes
from sweeps import next_run
from sweeps._types import ArrayLike, floating, integer

from .test_random_search import check_that_samples_are_from_the_same_distribution

logger = logging.getLogger(__name__)
BAYES_RANDOM_FALLBACK_SAMPLES = 100


def squiggle(x: ArrayLike) -> np.floating:
    # the maximum of this 1d function is at x=2 and the minimum is at ~3.6 over the
    # interval 0-5
    return np.exp(-((x - 2) ** 2)) + np.exp(-((x - 6) ** 2) / 10) + 1 / (x**2 + 1)


def rosenbrock(x: ArrayLike) -> np.floating:
    # has a minimum at (1, 1, 1, 1, ...) for 4 <= ndim <= 7
    return np.sum((x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


# A tiny deterministic convex problem for fast Bayes convergence tests.
# The loss is minimized when x is exactly 0.3, so tests can verify that the
# optimizer moves toward a known best point without spending hundreds of
# iterations on noisier objective functions.
def quadratic_loss(run: SweepRun) -> floating:
    return (run.config["x"]["value"] - 0.3) ** 2


def quadratic_run(x: floating) -> SweepRun:
    run = SweepRun(config={"x": {"value": x}}, state=RunState.finished)
    run.summary_metrics = {"loss": quadratic_loss(run)}
    return run


def simulate_bayes_search(
    f: Callable[[SweepRun], floating],
    config: SweepConfig,
    init_runs: Iterable[SweepRun] = (),
    improvement: floating = 0.1,
    num_iterations: integer = 20,
    run_state: RunState = RunState.finished,
) -> List[SweepRun]:

    metric_name = config["metric"]["name"]

    runs = list(init_runs)
    for _ in range(num_iterations):
        suggested_run = bayes.bayes_search_next_run(
            runs, config, minimum_improvement=improvement
        )
        suggested_run.state = run_state
        metric = f(suggested_run)
        if suggested_run.summary_metrics is None:
            suggested_run.summary_metrics = {}  # pragma: no cover
        suggested_run.summary_metrics[metric_name] = metric
        runs.append(suggested_run)

    if logger.isEnabledFor(logging.DEBUG):
        for run in runs:
            logger.debug(
                "Bayes search test run: config=%s state=%s", run.config, run.state
            )

    return runs


def assert_best_run_matches_optimum(
    runs: Iterable[SweepRun],
    config: SweepConfig,
    optimum: Dict[str, floating],
    atol: float = 0.2,
    run_state: RunState = RunState.finished,
) -> None:
    metric_name = config["metric"]["name"]
    opt_goal = config["metric"]["goal"]
    best_run = (min if opt_goal == "minimize" else max)(
        [r for r in runs if r.state == run_state],
        key=lambda run: run.metric_extremum(metric_name, opt_goal),  # type: ignore
    )
    for param_name in config["parameters"]:
        left_comp = best_run.config[param_name]["value"]
        right_comp = optimum[param_name]
        # Verify that the best observed run landed near the expected optimum.
        np.testing.assert_allclose(left_comp, right_comp, atol=atol)


def assert_suggestions_match_uniform_distribution(
    runs: List[SweepRun],
    param_name: str,
    min_value: floating,
    max_value: floating,
) -> None:
    # If Bayes has no usable metric signal, all observed points have the same
    # imputed objective value. In that case the GP cannot guide the search, so
    # suggestions should fall back to approximately uniform random sampling.
    samples = [run.config[param_name]["value"] for run in runs]
    check_that_samples_are_from_the_same_distribution(
        samples,
        np.random.uniform(min_value, max_value, len(samples)),
        np.linspace(min_value, max_value, 11),
    )


@pytest.mark.parametrize(
    "x",
    [
        {"distribution": "normal", "mu": 2, "sigma": 4},
        {"distribution": "log_uniform_values", "min": np.exp(-2), "max": np.exp(3)},
        {"min": 0.0, "max": 5.0},
        {"min": 0, "max": 5},
        {"distribution": "q_uniform", "min": 0.0, "max": 10.0, "q": 0.25},
    ],
)
def test_bayes_search_handles_supported_parameter_distributions(x):
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

    generated_runs = list(simulate_bayes_search(y, config, runs, num_iterations=3))
    assert len(generated_runs) == 4


def test_bayes_search_converges_on_simple_quadratic():
    config = SweepConfig(
        {
            "method": "bayes",
            "metric": {"name": "loss", "goal": "minimize"},
            "parameters": {"x": {"min": 0.0, "max": 1.0}},
        }
    )

    runs = simulate_bayes_search(
        quadratic_loss,
        config,
        init_runs=[quadratic_run(0.0), quadratic_run(1.0)],
        num_iterations=6,
    )

    best_loss = min(run.metric_extremum("loss", "minimum") for run in runs)
    assert best_loss < 0.001


def run_iterations(
    f: Callable[[ArrayLike], floating],
    bounds: ArrayLike,
    num_iterations: integer = 20,
    x_init: Optional[ArrayLike] = None,
    improvement: floating = 0.1,
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
            (sample, prob, pred, _, _, _,) = bayes.next_sample(
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
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Bayes next_sample test iteration: X=%s prob(I)=%s pred=%s value=%s",
                    sample,
                    prob,
                    pred,
                    f(sample),
                )
        assert sample_X is not None
        sample_X = np.asarray(sample_X, dtype=np.float64)

        X = np.append(X, sample_X, axis=0)
        y = np.array([f(x) for x in X]).flatten()

    return X, y


def test_squiggle_explores_unobserved_parameter_space():
    # The observed samples all have x >= 0, but the search bounds also allow
    # x < 0. A high improvement target should make expected improvement favor
    # the unobserved side instead of only exploiting the observed region.
    X = np.random.uniform(0, 5, 200)[:, None]
    Y = squiggle(X.ravel())
    (sample, prob, pred, _, _, _,) = bayes.next_sample(
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
        _,
    ) = bayes.next_sample(sample_X=X, sample_y=Y, X_bounds=[[0.0, 5.0]])
    assert (
        sample[0] > 1.0 and sample[0] < 4.0
    ), "Sample outside of 1-3 range: {}".format(sample[0])


def test_next_sample_converges_on_simple_quadratic():
    def f(x):
        return (x[0] - 0.3) ** 2

    x_init = np.array([[0.0], [1.0]])
    _, y = run_iterations(f, [[0.0, 1.0]], 6, x_init)
    assert np.min(y) < 0.001


def test_next_sample_converges_to_squiggle_minimum():
    x_init = np.array([[0.0], [5.0]])
    _, y = run_iterations(squiggle, [[0.0, 5.0]], 8, x_init)
    assert np.min(y) < 0.75


def test_next_sample_converges_to_squiggle_maximum():
    def f(x):
        return -squiggle(x)

    x_init = np.array([[0.0], [5.0]])
    _, y = run_iterations(f, [[0.0, 5.0]], 8, x_init)
    assert np.min(y) < -1.3


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


def test_next_sample_converges_on_rosenbrock():
    dimensions = 3
    x_init = np.zeros((1, dimensions))
    new_X, new_y = run_iterations(
        rosenbrock,
        [[0.0, 2.0]] * dimensions,
        30,
        x_init,
        improvement=0.1,
    )
    assert np.min(new_y) < 0.3
    assert new_X.shape == (31, dimensions)
    assert new_y.shape == (31,)


def test_iterations_squiggle_chunked():
    new_X, new_y = run_iterations(
        squiggle,
        [[0.0, 5.0]],
        x_init=np.array([[0.0], [5.0]]),
        chunk_size=5,
        num_iterations=8,
        improvement=0.1,
    )
    assert np.min(new_y) < 0.76
    assert new_X.shape == (10, 1)
    assert new_y.shape == (10,)


def test_next_sample_with_one_observation_uses_provided_candidates(monkeypatch):
    def choose_second_candidate(num_candidates):
        return 1

    monkeypatch.setattr(bayes.np.random, "choice", choose_second_candidate)

    sample, prob, pred, pred_std, expected_improvement, warnings = bayes.next_sample(
        sample_X=np.array([[0.25]]),
        sample_y=np.array([3.5]),
        test_X=[[0.1], [0.9]],
    )

    # Verify the single-observation fallback chooses from test_X.
    np.testing.assert_allclose(sample, [0.9])
    assert prob == 1.0
    assert pred == 3.5
    assert np.isnan(pred_std)
    assert np.isnan(expected_improvement)
    assert warnings == ""


def test_train_gaussian_process_limits_current_samples(capsys, monkeypatch):
    class FakeGaussianProcess:
        def predict(self, X):
            return np.zeros(np.asarray(X).shape[0])

    fit_calls = []

    def fake_fit_normalized_gaussian_process(X, y, nu=1.5):
        fit_calls.append((np.asarray(X), np.asarray(y)))
        return FakeGaussianProcess(), 0.0, 1.0

    monkeypatch.setattr(
        bayes, "fit_normalized_gaussian_process", fake_fit_normalized_gaussian_process
    )
    sample_X = np.linspace(0.0, 1.0, 6)[:, None]
    sample_y = (sample_X[:, 0] - 0.5) ** 2
    current_X = np.linspace(0.0, 1.0, 6)[:, None]

    gp, y_mean, y_stddev = bayes.train_gaussian_process(
        sample_X,
        sample_y,
        current_X=current_X,
        max_samples=10,
    )

    assert "dropping some currently running parameters" in capsys.readouterr().out
    assert isinstance(gp, FakeGaussianProcess)
    assert y_mean == 0.0
    assert y_stddev == 1.0
    assert len(fit_calls) == 2
    assert fit_calls[0][0].shape == (6, 1)
    assert fit_calls[0][1].shape == (6,)
    assert fit_calls[1][0].shape == (11, 1)
    assert fit_calls[1][1].shape == (11,)
    # Verify current_X was trimmed to max_samples - 5 before fantasy fitting.
    np.testing.assert_allclose(fit_calls[1][0][-5:], current_X[:5])
    np.testing.assert_allclose(fit_calls[1][1][-5:], np.zeros(5))


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

    for _ in range(100):
        suggestion = next_run(sweep_config_bayes_search_2params_with_metric, runs)
        metric = {"loss": loss_func(suggestion)}
        suggestion.history = [metric]
        suggestion.state = RunState.finished
        runs.append(suggestion)

    best_run = min(runs, key=lambda r: r.metric_extremum("loss", "minimum"))
    assert best_run.metric_extremum("loss", "minimum") < 2.5
    assert all("expected_improvement" in run.search_info for run in runs[2:])


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
    for _ in range(BAYES_RANDOM_FALLBACK_SAMPLES):
        suggestion = next_run(config, runs)
        suggestion.state = RunState.finished
        runs.append(suggestion)
        assert (
            "Some dimmensions of kernel are close to their bounds"
            in suggestion.search_info["warnings"]
        )

    assert_suggestions_match_uniform_distribution(runs, "v2", 1, 10)


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
    for _ in range(BAYES_RANDOM_FALLBACK_SAMPLES):
        suggestion = next_run(config, runs)
        suggestion.state = RunState.finished
        runs.append(suggestion)
        assert (
            "Some dimmensions of kernel are close to their bounds"
            in suggestion.search_info["warnings"]
        )

    assert_suggestions_match_uniform_distribution(runs, "v2", 1, 10)


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

    for _ in range(BAYES_RANDOM_FALLBACK_SAMPLES):
        suggestion = next_run(sweep_config_bayes_search_2params_with_metric, runs)
        suggestion.state = RunState.finished
        runs.append(suggestion)
        assert (
            "Some dimmensions of kernel are close to their bounds"
            in suggestion.search_info["warnings"]
        )

    assert_suggestions_match_uniform_distribution(runs, "v2", 1, 10)


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
    for _ in range(20):
        suggestion = next_run(config, runs)
        metric = {"acc": loss_func(suggestion)}
        suggestion.history = [metric]
        suggestion.state = RunState.finished
        runs.append(suggestion)

    for run in runs:
        assert run.config["v1"]["value"] in config["parameters"]["v1"]["values"]
        assert v2_min <= run.config["v2"]["value"] <= v2_max

    best_run = max(runs, key=lambda r: r.metric_extremum("acc", "maximum"))
    best_x = [best_run.config["v1"]["value"], best_run.config["v2"]["value"]]
    assert best_x[0] == ["5", "6"]


def test_bayes_categorical_list_values_normalize_and_round_trip():
    values = [(2, 3), [3, 4], ["5", "6"], [(7, 8), ["9", [10, 11]]]]
    config = bayes.bayes_baseline_validate_and_fill(
        {
            "method": "bayes",
            "metric": {"name": "acc", "goal": "maximize"},
            "parameters": {"v1": {"distribution": "categorical", "values": values}},
        }
    )
    runs = [
        SweepRun(
            state=RunState.finished,
            config={"v1": {"value": value}},
            summary_metrics={"acc": index},
        )
        for index, value in enumerate(values)
    ]

    params, sample_X, _, _, _ = bayes._construct_gp_data(runs, config)
    v1_index = params.param_names_to_index["v1"]
    v1_param = params.param_names_to_param["v1"]

    # Verify categorical values map to evenly spaced normalized GP inputs.
    np.testing.assert_allclose(sample_X[:, v1_index], [0.25, 0.5, 0.75, 1.0])
    assert v1_param.ppf(np.array([0.1, 0.4, 0.7, 0.9])) == values


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
    data["jsonPayload"]["data"]["config"]["metric"]["impute"] = "worst"
    _, _, _, y, _ = bayes._construct_gp_data(
        [SweepRun(**r) for r in data["jsonPayload"]["data"]["runs"]],
        data["jsonPayload"]["data"]["config"],
    )
    np.testing.assert_array_less(np.abs(y + 98), 5)


# search with 2 finished runs - metrics are ignored because they are boolean
def test_runs_bayes_runs2_boolmetric():

    config = SweepConfig(
        {
            "metric": {"name": "xloss", "goal": "minimize"},
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
            {"xloss": True},
        ],
        config={"v2": {"value": 6}},
        summary_metrics={"zloss": 1.2},
    )
    r2 = SweepRun(
        name="b2",
        state=RunState.finished,
        config={"v2": {"value": 8}},
        summary_metrics={"xloss": False},
        history=[],
    )

    runs = [r1, r2]
    for _ in range(BAYES_RANDOM_FALLBACK_SAMPLES):
        suggestion = next_run(config, runs)
        suggestion.state = RunState.finished
        runs.append(suggestion)
        assert (
            "Some dimmensions of kernel are close to their bounds"
            in suggestion.search_info["warnings"]
        )

    assert_suggestions_match_uniform_distribution(runs, "v2", 1, 10)


def test_bayes_impute_best():

    sweep_config = {
        "method": "bayes",
        "metric": {"name": "loss", "goal": "minimize", "impute": "best"},
        "parameters": {"a": {"min": 0.0, "max": 1.0}},
    }

    runs = [
        SweepRun(
            name="a",
            state=RunState.failed,  # This wont be stopped because already stopped
            history=[
                {"loss": 10},
                {"loss": 8},
                {"loss": 10},
            ],
            config={"a": {"value": 0.2}},
        ),
        SweepRun(
            name="b",
            state=RunState.failed,  # This should be stopped
            history=[
                {"loss": 10},
                {"loss": 6},
                {"loss": 10},
            ],
            config={"a": {"value": 0.4}},
        ),
        SweepRun(
            name="c",
            state=RunState.failed,  # This passes band 1 but not band 2
            history=[
                {"loss": 10},
                {"loss": 4},
                {"loss": 4},
                {"loss": 10},
            ],
            config={"a": {"value": 0.6}},
        ),
        SweepRun(
            name="d",
            state=RunState.failed,
            history=[
                {"loss": 10},
                {"loss": 2},
                {"loss": 2},
                {"loss": 10},
            ],
            config={"a": {"value": 0.8}},
        ),
        SweepRun(
            name="e",
            state=RunState.failed,
            history=[
                {"loss": 10},
                {"loss": 1},
                {"loss": 1},
                {"loss": 10},
            ],
            config={"a": {"value": 0.9}},
        ),
    ]

    def opt_func(run):
        return 10 - run.config["a"]["value"] * 10

    # check that best finds the answer
    generated_runs = simulate_bayes_search(
        opt_func,
        sweep_config,
        init_runs=runs,
        num_iterations=5,
    )
    assert_best_run_matches_optimum(
        generated_runs, sweep_config, {"a": 1.0}, atol=0.001
    )

    # check that worst doesn't
    sweep_config["metric"]["impute"] = "worst"

    with pytest.raises(AssertionError):
        generated_runs = simulate_bayes_search(
            opt_func,
            sweep_config,
            init_runs=runs,
            num_iterations=5,
        )
        assert_best_run_matches_optimum(
            generated_runs, sweep_config, {"a": 1.0}, atol=0.001
        )


def test_construct_gp_data_imputes_failed_run_with_latest_history_metric():
    config = bayes.bayes_baseline_validate_and_fill(
        SweepConfig(
            {
                "method": "bayes",
                "metric": {"name": "loss", "goal": "minimize", "impute": "latest"},
                "parameters": {"a": {"min": 0.0, "max": 1.0}},
            }
        )
    )
    runs = [
        SweepRun(
            state=RunState.failed,
            history=[
                {"loss": 5.0},
                {"loss": 1.0},
                {"loss": float("nan")},
                {"loss": "bad"},
                {"loss": 3.0},
            ],
            summary_metrics={"loss": 0.5},
            config={"a": {"value": 0.25}},
        ),
        SweepRun(
            state=RunState.finished,
            history=[{"loss": 4.0}],
            config={"a": {"value": 0.75}},
        ),
    ]

    _, sample_X, current_X, sample_y, _ = bayes._construct_gp_data(runs, config)

    assert sample_X.shape == (2, 1)
    assert len(current_X) == 0
    # Verify failed runs use the latest valid history metric, not best or summary.
    np.testing.assert_allclose(sample_y, [3.0, 4.0])


def test_impute_latest_without_valid_metric_uses_failed_value():
    run = SweepRun(
        history=[
            {"other": 1.0},
            {"loss": None},
            {"loss": False},
        ]
    )

    # Verify latest falls back to the failed value when no numeric history exists.
    assert bayes.impute("minimize", "loss", bayes.ImputeStrategy.latest, run=run) == 0.0


def test_construct_gp_data_imputes_missing_finished_metric_with_failed_value():
    config = bayes.bayes_baseline_validate_and_fill(
        SweepConfig(
            {
                "method": "bayes",
                "metric": {"name": "loss", "goal": "minimize", "impute": "best"},
                "parameters": {"x": {"min": 0.0, "max": 1.0}},
            }
        )
    )
    run = SweepRun(
        config={"x": {"value": 0.4}},
        state=RunState.finished,
        history=[{"other": 1.0}],
        summary_metrics={},
    )

    _, sample_X, current_X, sample_y, warnings = bayes._construct_gp_data([run], config)

    assert sample_X.shape == (1, 1)
    assert len(current_X) == 0
    # Verify best imputation falls back to the failed value for a missing metric.
    np.testing.assert_allclose(sample_y, [0.0])
    assert warnings == ""


def test_bayes_impute_while_running_best_includes_running_run():
    def y(x: SweepRun) -> floating:
        return squiggle(x.config["x"]["value"])

    run = SweepRun(config={"x": {"value": 2.0}}, state=RunState.running)
    run.summary_metrics["y"] = y(run)

    config = SweepConfig(
        {
            "method": "bayes",
            "metric": {
                "name": "y",
                "goal": "maximize",
                "impute_while_running": "best",
            },
            "parameters": {
                "x": {
                    "distribution": "log_uniform_values",
                    "min": np.exp(-2),
                    "max": np.exp(3),
                }
            },
        }
    )

    config = bayes.bayes_baseline_validate_and_fill(config)
    _, sample_X, current_X, sample_y, _ = bayes._construct_gp_data([run], config)
    assert sample_X.shape == (1, 1)
    assert len(current_X) == 0
    # Verify a running maximizing run contributes its metric with minimizer sign flip.
    np.testing.assert_allclose(sample_y[0], -y(run))


def test_bayes_impute_while_running_best_guides_search(monkeypatch):
    def fixed_random_sample(
        x_bounds: ArrayLike, num_test_samples: integer
    ) -> ArrayLike:
        candidates = np.linspace(0.0, 1.0, int(num_test_samples))[:, None]
        candidates[min(300, len(candidates) - 1), 0] = 0.3
        return candidates

    monkeypatch.setattr(bayes, "random_sample", fixed_random_sample)

    config = SweepConfig(
        {
            "method": "bayes",
            "metric": {
                "name": "loss",
                "goal": "minimize",
                "impute_while_running": "best",
            },
            "parameters": {"x": {"min": 0.0, "max": 1.0}},
        }
    )

    initial_runs = [quadratic_run(0.0), quadratic_run(1.0)]
    for run in initial_runs:
        run.state = RunState.running

    generated_runs = simulate_bayes_search(
        quadratic_loss,
        config,
        init_runs=initial_runs,
        num_iterations=5,
        run_state=RunState.running,
    )

    best_loss = min(run.metric_extremum("loss", "minimum") for run in generated_runs)
    assert best_loss < 0.001
