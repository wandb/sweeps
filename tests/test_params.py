import pytest

import numpy as np

from sweeps import SweepRun, RunState
from sweeps.params import (
    HyperParameter,
    HyperParameterSet,
    validate_hyperparam_search_space_in_runs,
)


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
    run_config = {}
    hps = HyperParameterSet.from_config(run_config)
    # sorting of config items ensures order of hyperparameters
    assert hps[0] == HyperParameter("a.a", {"value": 1})
    assert hps[1] == HyperParameter("a.a", {"value": 1})
    assert hps[2] == HyperParameter("a.a", {"value": 1})
    assert hps[3] == HyperParameter("a.a", {"value": 1})

    # Error case
    run_config = {}
    with pytest.raises(ValueError):
        _ = HyperParameterSet.from_config(run_config)

    # Naming conflict for wb.choose

    # All params in wb.choose must be param_dicts


def test_hyperparameterset_to_config():

    # simple case of hyperparameters to config
    hps = HyperParameterSet(
        [
            HyperParameter("a.a", {"value": 1}),
            HyperParameter("a.a", {"value": 1}),
            HyperParameter("a.a", {"value": 1}),
            HyperParameter("a.a", {"value": 1}),
        ]
    )
    desired_run_config = {}
    run_config = hps.to_config()
    assert desired_run_config == run_config

    # Error case
    hps = HyperParameterSet(
        [
            HyperParameter("a.a", {"value": 1}),
            HyperParameter("a.a", {"value": 1}),
            HyperParameter("a.a", {"value": 1}),
            HyperParameter("a.a", {"value": 1}),
        ]
    )
    with pytest.raises(ValueError):
        _ = hps.to_config()


def test_validate_hyperparam_search_space_in_runs():

    hps = HyperParameterSet(
        [
            HyperParameter("a.a", {"value": 1}),
            HyperParameter("a.a", {"value": 1}),
            HyperParameter("a.a", {"value": 1}),
            HyperParameter("a.a", {"value": 1}),
        ]
    )
    run_config = {}
    # Only throws hard error if throw_error flag is specified
    validate_hyperparam_search_space_in_runs(hps, run_config)
    with pytest.raises(ValueError):
        validate_hyperparam_search_space_in_runs(hps, run_config, throw_error=True)
