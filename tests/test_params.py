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

def test_make_run_config_from_params():

    # Nested values by default use '.' as delimiter
    runconf = make_run_config_from_params(
        HyperParameterSet(
            [
                HyperParameter("a.a", {"value": 1}),
                HyperParameter("a.b", {"value": 2, "nested": True}),
                HyperParameter("a.c", {"value": 3, "nested": True}),
                HyperParameter("a.d.e", {"value": 4, "nested": True}),
            ]
        )
    )
    assert runconf == {
        "a.a": {"value": 1},
        "a.b": {"value": 2},
        "a.c": {"value": 3},
        "a.d.e": {"value": 4},
        "a": {"value": {"b": 2, "c": 3, "d": {"e": 4}}},
    }

    # Nested values can't overwrite existing non-nested keys
    params = HyperParameterSet(
        [
            HyperParameter("a", {"value": 1}),
            HyperParameter("a.c", {"value": 2, "nested": True}),
        ]
    )
    with pytest.raises(ValueError):
        make_run_config_from_params(params)


@pytest.mark.parametrize("delimiter", [".", "_", "foo"])
def test_make_run_config_from_params_custom_delimiters(delimiter):

    runconf = make_run_config_from_params(
        HyperParameterSet(
            [
                HyperParameter(
                    f"a{delimiter}b",
                    {"value": 1, "nested": True, "nest_delimiter": delimiter},
                ),
                HyperParameter(
                    f"a{delimiter}c",
                    {"value": 2, "nested": True, "nest_delimiter": delimiter},
                ),
            ]
        )
    )
    assert runconf == {
        f"a{delimiter}b": {"value": 1},
        f"a{delimiter}c": {"value": 2},
        "a": {"value": {"b": 1, "c": 2}},
    }

    # Throw error if delimiters are different
    params = HyperParameterSet(
        [
            HyperParameter(
                f"a{delimiter}b",
                {"value": 1, "nested": True, "nest_delimiter": delimiter},
            ),
            HyperParameter("a-c", {"value": 2, "nested": True, "nest_delimiter": "-"}),
        ]
    )
    with pytest.raises(ValueError):
        make_run_config_from_params(params)