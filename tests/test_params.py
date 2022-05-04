import pytest

import numpy as np

from sweeps import SweepRun, RunState
from sweeps.params import HyperParameter, HyperParameterSet


def test_hyperparameterset_initialize():

    _ = HyperParameterSet(
        [
            HyperParameter(
                "inv_log_uniform",
                {
                    "min": 0,
                    "max": 1,
                    "distribution": "inv_log_uniform",
                },
            ),
            HyperParameter("constant", {"value": 1}),
            HyperParameter("categorical", {"values": [1, 2, 3]}),
        ]
    )

    with pytest.raises(TypeError):
        _ = HyperParameterSet(
            [
                HyperParameter("constant", {"value": 1}),
                "not-a-hyperparameter",
            ]
        )


def test_hyperparameterset_normalize_runs():

    valid_set = HyperParameterSet(
        [
            HyperParameter("v1", {"value": 1}),
            HyperParameter("v2", {"values": [1, 2]}),
        ]
    )
    r1 = SweepRun(
        name="b",
        state=RunState.finished,
        config={"v1": {"value": 1}, "v2": {"value": 1}},
        history=[],
    )
    r2 = SweepRun(
        name="b",
        state=RunState.finished,
        config={"v1": {"value": 1}, "v2": {"value": 2}},
        history=[],
    )
    normalized_runs = valid_set.normalize_runs_as_array([r1, r2])
    assert normalized_runs.shape == (2, 1)

    valid_set = HyperParameterSet(
        [
            HyperParameter("v1", {"value": 1}),
            HyperParameter("v2", {"values": [1, 2]}),
        ]
    )
    r1 = SweepRun(
        name="b",
        state=RunState.finished,
        config={"v1": {"value": 1}},
        history=[],
    )
    normalized_runs = valid_set.normalize_runs_as_array([r1, r2])
    assert normalized_runs.shape == (2, 1)

    valid_set = HyperParameterSet(
        [
            HyperParameter("v1", {"value": 1}),
            HyperParameter("v2", {"values": [1, np.inf]}),
        ]
    )
    r1 = SweepRun(
        name="b",
        state=RunState.finished,
        config={"v1": {"value": 1}, "v2": {"value": 1}},
        history=[],
    )
    r2 = SweepRun(
        name="b",
        state=RunState.finished,
        config={"v1": {"value": 1}, "v2": {"value": np.inf}},
        history=[],
    )
    normalized_runs = valid_set.normalize_runs_as_array([r1, r2])
    assert normalized_runs.shape == (2, 1)


def test_hyperparameterset_from_config():

    # simple case of hyperparameters from config
    sweep_config = {
        "method": "grid",
        "metric": {"name": "loss", "goal": "minimize"},
        "parameters": {
            "a": {
                "parameters": {
                    "b": {"value": 1},
                    "c": {
                        "parameters": {"d": {"value": 1}},
                    },
                },
            },
        },
    }
    hps = HyperParameterSet.from_config(sweep_config["parameters"])
    # sorting of config items ensures order of hyperparameters
    _delimiter = HyperParameterSet.NESTING_DELIMITER
    assert hps[0]._name_and_value() == (f"a{_delimiter}b", 1)
    assert hps[1]._name_and_value() == (f"a{_delimiter}c{_delimiter}d", 1)

    # Error case
    bad_run_config = {
        "method": "grid",
        "metric": {"name": "loss", "goal": "minimize"},
        "parameters": {
            42: {
                "parameters": {
                    "b": 42,
                },
            },
        },
    }
    with pytest.raises(AssertionError):
        _ = HyperParameterSet.from_config(bad_run_config)


def test_hyperparameterset_to_config():

    # simple case of hyperparameters to config
    _delimiter = HyperParameterSet.NESTING_DELIMITER
    hps = HyperParameterSet(
        [
            HyperParameter(f"a{_delimiter}b", {"value": 1}),
            HyperParameter(f"a{_delimiter}c{_delimiter}d", {"value": 1}),
        ]
    )
    desired_run_config = {"a": {"value": {"b": 1, "c": {"d": 1}}}}
    run_config = hps.to_config()
    assert desired_run_config == run_config

    # Error case - Name conflict upon nesting
    hps = HyperParameterSet(
        [
            HyperParameter(f"a{_delimiter}b", {"value": 1}),
            HyperParameter(f"a{_delimiter}c{_delimiter}d", {"value": 1}),
            HyperParameter(f"a{_delimiter}c", {"value": 1}),
        ]
    )
    with pytest.raises(ValueError):
        _ = hps.to_config()
