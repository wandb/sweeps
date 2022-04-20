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
        config={"v1": {"value": 1}, "v2": {"value": 2}},
        history=[],
    )
    normalized_runs = valid_set.normalize_runs_as_array([r1, r2])
    assert normalized_runs.shape == (2, 1)
