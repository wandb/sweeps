from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import sweeps.grid_search as grid_search_module
from sweeps.config import SweepConfig
from sweeps.grid_search import yaml_hash, grid_search_next_runs
from sweeps.run import RunState, SweepRun, next_run, next_runs


def kernel_for_grid_search_tests(
    runs: List[SweepRun],
    config: SweepConfig,
    answers: Sequence[Tuple],
    randomize: bool,
    check_order: bool = False,
    vectorize: bool = False,
) -> None:
    """This kernel assumes that sweep config has two categorical parameters
    named v1 and v2."""

    suggested_parameters = [
        (
            run.config["v1"]["value"],
            run.config["v2"]["value"],
        )
        for run in runs
    ]

    def handle_suggestion(suggestion: SweepRun):
        assert suggestion.search_info is None
        assert suggestion.state == RunState.pending
        runs.append(suggestion)
        suggested_parameters.append(
            (
                suggestion.config["v1"]["value"],
                suggestion.config["v2"]["value"],
            )
        )

    done = False
    while not done:
        if not vectorize:
            suggestion = next_run(config, runs, randomize_order=randomize)
            if suggestion is None:  # done
                break
            handle_suggestion(suggestion)
        else:
            suggestions = next_runs(config, runs, randomize_order=randomize, n=500)
            for suggestion in suggestions:
                if suggestion is None:
                    done = True
                    break
                handle_suggestion(suggestion)

    assert len(answers) == len(suggested_parameters)
    for i, key in enumerate(suggested_parameters):
        if check_order:
            assert answers[i] == key
        else:
            assert key in answers


@pytest.mark.parametrize("vectorize", [True, False])
@pytest.mark.parametrize("randomize", [True, False])
def test_grid_from_start_with_and_without_randomize(
    sweep_config_2params_grid_search, randomize, vectorize
):
    kernel_for_grid_search_tests(
        [],
        sweep_config_2params_grid_search,
        randomize=randomize,
        vectorize=vectorize,
        answers=[(1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)],
    )


@pytest.mark.parametrize("vectorize", [True, False])
@pytest.mark.parametrize("randomize", [True, False])
def test_grid_search_starting_from_in_progress(
    sweep_config_2params_grid_search, randomize, vectorize
):
    runs = [
        SweepRun(config={"v1": {"value": 2}, "v2": {"value": 4}}),
        SweepRun(config={"v1": {"value": 1}, "v2": {"value": 5}}),
    ]
    kernel_for_grid_search_tests(
        runs,
        sweep_config_2params_grid_search,
        randomize=randomize,
        answers=[(1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)],
        vectorize=vectorize,
    )


@pytest.mark.parametrize("vectorize", [True, False])
def test_grid_search_with_list_values(vectorize):
    # https://sentry.io/organizations/weights-biases/issues/2501125152/?project=5812400&query=is%3Aresolved&statsPeriod=14d
    config = SweepConfig(
        {
            "method": "grid",
            "parameters": {
                "v1": {
                    "values": ["", [9, 5]],
                },
                "v2": {
                    "values": [256, 512],
                },
            },
        }
    )

    kernel_for_grid_search_tests(
        [],
        config,
        randomize=False,
        answers=[("", 256), ("", 512), ([9, 5], 256), ([9, 5], 512)],
        vectorize=vectorize,
    )


@pytest.mark.parametrize("vectorize", [True, False])
def test_grid_search_duplicated_values_are_not_duplicated_in_answer(vectorize):
    duplicated_config = SweepConfig(
        {
            "method": "grid",
            "parameters": {
                "v1": {"values": [None, 2, 3, "a", (2, 3), 3]},
                "v2": {"values": ["a", "b", "c'"]},
            },
        }
    )

    runs = []
    kernel_for_grid_search_tests(
        runs,
        duplicated_config,
        randomize=True,
        vectorize=vectorize,
        answers=[
            (
                None,
                "a",
            ),
            (
                2,
                "a",
            ),
            (
                3,
                "a",
            ),
            ("a", "a"),
            (
                (2, 3),
                "a",
            ),
            (
                None,
                "b",
            ),
            (
                2,
                "b",
            ),
            (
                3,
                "b",
            ),
            ((2, 3), "b"),
            ("a", "b"),
            (
                None,
                "c'",
            ),
            (
                2,
                "c'",
            ),
            (
                3,
                "c'",
            ),
            (
                (2, 3),
                "c'",
            ),
            ("a", "c'"),
        ],
    )
    assert len(runs) == 15


def test_grid_search_constant_val_is_propagated():
    config_const = SweepConfig(
        {
            "method": "grid",
            "parameters": {
                "v1": {"values": ["a", "b", "c'"]},
                "v2": {"value": 1},
            },
        }
    )

    runs = []
    run = next_run(config_const, runs)
    assert "v1" in run.config
    assert "v2" in run.config


@pytest.mark.parametrize("vectorize", [True, False])
def test_grid_search_constant_vals_only(vectorize):
    config_const = SweepConfig(
        {
            "method": "grid",
            "parameters": {
                "v1": {"value": "a"},
                "v2": {"value": "b"},
            },
        }
    )

    kernel_for_grid_search_tests(
        [],
        config_const,
        answers=[
            ("a", "b"),
        ],
        randomize=False,
        vectorize=vectorize,
    )


@pytest.mark.parametrize("vectorize", [True, False])
@pytest.mark.parametrize("randomize", [True, False])
def test_grid_search_dict_val_is_propagated(randomize, vectorize):
    config_const = SweepConfig(
        {
            "method": "grid",
            "parameters": {
                "v1": {"values": ["a", "b", "c'"]},
                "v2": {
                    "values": [
                        {"a": "b"},
                        {"c": "d", "b": "g"},
                        {"e": {"f": "g"}},
                        {"a": "b"},
                        {"b": "g", "c": "d"},
                    ]
                },
            },
        }
    )

    kernel_for_grid_search_tests(
        [],
        config_const,
        randomize=randomize,
        vectorize=vectorize,
        answers=[
            ("a", {"a": "b"}),
            ("a", {"c": "d", "b": "g"}),
            ("a", {"e": {"f": "g"}}),
            ("b", {"a": "b"}),
            ("b", {"c": "d", "b": "g"}),
            ("b", {"e": {"f": "g"}}),
            ("c'", {"a": "b"}),
            ("c'", {"c": "d", "b": "g"}),
            ("c'", {"e": {"f": "g"}}),
        ],
    )


@pytest.mark.parametrize("vectorize", [True, False])
def test_grid_search_anaconda1_order(vectorize):
    config_const = SweepConfig(
        {
            "method": "grid",
            "parameters": {
                "v1": {"values": ["a", "b", "c", "a"]},
                "v2": {"values": [1, 2, 3]},
            },
        }
    )

    kernel_for_grid_search_tests(
        [],
        config_const,
        randomize=False,
        answers=[
            ("a", 1),
            ("a", 2),
            ("a", 3),
            ("b", 1),
            ("b", 2),
            ("b", 3),
            ("c", 1),
            ("c", 2),
            ("c", 3),
        ],
        check_order=True,
        vectorize=vectorize,
    )


@pytest.mark.parametrize("vectorize", [True, False])
@pytest.mark.parametrize("randomize", [True, False])
def test_int_uniform_bounded_grid_search(randomize, vectorize):
    int_uniform_config = SweepConfig(
        {
            "method": "grid",
            "parameters": {
                "v1": {"distribution": "int_uniform", "min": -52, "max": 10},
                "v2": {"value": "test"},
            },
        }
    )

    kernel_for_grid_search_tests(
        [],
        int_uniform_config,
        randomize=randomize,
        vectorize=vectorize,
        answers=[(v, "test") for v in range(-52, 11)],
    )


@pytest.mark.parametrize("vectorize", [True, False])
@pytest.mark.parametrize("randomize", [True, False])
def test_q_uniform_bounded_grid_search(randomize, vectorize):
    int_uniform_config = SweepConfig(
        {
            "method": "grid",
            "parameters": {
                "v1": {
                    "distribution": "q_uniform",
                    "min": -9.211,
                    "max": 23.23,
                    "q": 2.341,
                },
                "v2": {"value": "test"},
            },
        }
    )

    kernel_for_grid_search_tests(
        [],
        int_uniform_config,
        randomize=randomize,
        vectorize=vectorize,
        answers=[(v, "test") for v in np.arange(-9.211, 23.23, 2.341)],
    )


def test_nested_grid_search_advances():
    request_body = {
        "config": {
            "method": "grid",
            "metric": {"goal": "minimize", "name": "loss"},
            "command": ["${env}", "python3", "-m", "hellowsweep"],
            "project": "sweep_test",
            "parameters": {
                "p1": {"values": ["a", "b", "c"]},
                "p2": {
                    "parameters": {
                        "n1": {"values": [0.01, 0.001]},
                        "n2": {"value": 0.9},
                        "n3": {"values": [1, 2, 3]},
                    }
                },
            },
            "description": "hellosweep: A sample program for getting started with wandb sweeps.",
        },
        "nArgs": 1,
        "requestId": "9eff8681-25cf-4b9f-9561-a8675e2100f7",
        "runs": [
            {
                "name": "6s48hvhy",
                "config": {
                    "p1": {"desc": None, "value": "a"},
                    "p2": {"desc": None, "value": {"n1": 0.01, "n2": 0.9, "n3": 1}},
                    "_wandb": {
                        "desc": None,
                        "value": {
                            "t": {
                                "1": [55],
                                "2": [55],
                                "3": [2, 23, 37],
                                "4": "3.10.7",
                                "5": "0.13.6.dev1",
                                "8": [4, 5],
                            },
                            "start_time": 1671759722.030263,
                            "cli_version": "0.13.6.dev1",
                            "is_jupyter_run": False,
                            "python_version": "3.10.7",
                            "is_kaggle_kernel": False,
                        },
                    },
                },
                "state": "finished",
                "summaryMetrics": {
                    "loss": 0.009000000000000001,
                    "_step": 0,
                    "label": "a",
                    "_wandb": {"runtime": 0},
                    "_runtime": 0.33578920364379883,
                    "_timestamp": 1671759722.3660522,
                },
            },
        ],
    }

    config = request_body["config"]
    runs = [SweepRun(**run) for run in request_body["runs"]]

    result = next_runs(config, runs, validate=False, n=1)

    assert not all(
        [
            result[0].config[pname]["value"] == runs[0].config[pname]["value"]
            for pname in result[0].config
        ]
    )


def test_yaml_hash_float():
    assert yaml_hash(3000000.0) == yaml_hash(3000000)


def test_grid_search_caches_repeated_yaml_hash_values(monkeypatch):
    large_value = [f"col_{i}" for i in range(100)]
    sweep_config = {
        "method": "grid",
        "parameters": {
            "cols": {"values": [large_value, ["next"]]},
        },
    }
    runs = [SweepRun(config={"cols": {"value": list(large_value)}}) for _ in range(5)]

    yaml_hash_calls: List[str] = []
    original_yaml_hash = grid_search_module.yaml_hash

    def counting_yaml_hash(value: Any) -> str:
        yaml_hash_calls.append(repr(value))
        return original_yaml_hash(value)

    monkeypatch.setattr(grid_search_module, "yaml_hash", counting_yaml_hash)

    result = next_runs(sweep_config, runs)

    assert len(result) == 1
    assert result[0] is not None
    assert result[0].config["cols"]["value"] == ["next"]
    assert yaml_hash_calls.count(repr(large_value)) == 2


def test_grid_search_matches_integer_float_values():
    config = SweepConfig(
        {
            "method": "grid",
            "parameters": {
                "v1": {"values": [3000000, "next"]},
                "v2": {"value": "test"},
            },
        }
    )

    runs = [SweepRun(config={"v1": {"value": 3000000.0}, "v2": {"value": "test"}})]

    suggestion = next_runs(config, runs)[0]

    assert suggestion is not None
    assert suggestion.config["v1"]["value"] == "next"


def test_grid_search_keeps_bool_and_int_values_distinct():
    config = SweepConfig(
        {
            "method": "grid",
            "parameters": {
                "v1": {"values": [True, 1]},
                "v2": {"value": "test"},
            },
        }
    )

    runs = [SweepRun(config={"v1": {"value": True}, "v2": {"value": "test"}})]

    suggestion = next_runs(config, runs)[0]

    assert suggestion is not None
    assert suggestion.config["v1"]["value"] == 1


def test_grid_search_matches_nan_values():
    config = SweepConfig(
        {
            "method": "grid",
            "parameters": {
                "v1": {"values": [float("nan"), 1]},
                "v2": {"value": "test"},
            },
        }
    )

    runs = [SweepRun(config={"v1": {"value": float("nan")}, "v2": {"value": "test"}})]

    suggestion = next_runs(config, runs)[0]

    assert suggestion is not None
    assert suggestion.config["v1"]["value"] == 1


def test_grid_search_keeps_list_and_tuple_values_distinct():
    config = SweepConfig(
        {
            "method": "grid",
            "parameters": {
                "v1": {"values": [[2, 3], (2, 3)]},
                "v2": {"value": "test"},
            },
        }
    )

    runs = [SweepRun(config={"v1": {"value": [2, 3]}, "v2": {"value": "test"}})]

    suggestion = next_runs(config, runs)[0]

    assert suggestion is not None
    assert suggestion.config["v1"]["value"] == (2, 3)


def test_grid_search_matches_dict_values_regardless_of_key_order():
    config = SweepConfig(
        {
            "method": "grid",
            "parameters": {
                "v1": {"values": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]},
                "v2": {"value": "test"},
            },
        }
    )

    runs = [
        SweepRun(config={"v1": {"value": {"b": 2, "a": 1}}, "v2": {"value": "test"}})
    ]

    suggestion = next_runs(config, runs)[0]

    assert suggestion is not None
    assert suggestion.config["v1"]["value"] == {"a": 3, "b": 4}


# Tests for grid_search_next_runs, focused on dict-valued parameter deduplication.


def _make_grid_config(
    param_name: str,
    values: List,
    extra_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    parameters: Dict[str, Any] = {
        param_name: {"values": values},
    }
    if extra_params:
        parameters.update(extra_params)
    return {
        "method": "grid",
        "parameters": parameters,
    }


def _make_run(config: dict, name: str = "run") -> SweepRun:
    return SweepRun(name=name, state=RunState.finished, config=config)


# --- Dict-valued parameter tests ---


def test_dict_valued_param_with_injected_key():
    """A run whose dict-valued param has an extra runtime key should still
    be recognized as covering its grid point.

    This is the core config pollution bug: sinergym injects env_name into
    env_params.value, changing yaml_hash so anaconda re-suggests the point.
    """
    grid_values = [
        {"env_id": "Env-v0", "max_steps": 1000},
        {"env_id": "Env-v1", "max_steps": 2000},
    ]
    sweep_config = _make_grid_config("env_params", grid_values)

    runs = [
        _make_run(
            {
                "env_params": {
                    "value": {
                        "env_id": "Env-v0",
                        "max_steps": 1000,
                        "env_name": "SAC_abc123",
                    }
                }
            },
            name="run-0",
        ),
        _make_run(
            {
                "env_params": {
                    "value": {
                        "env_id": "Env-v1",
                        "max_steps": 2000,
                        "env_name": "SAC_def456",
                    }
                }
            },
            name="run-1",
        ),
    ]

    result = grid_search_next_runs(runs, sweep_config)
    assert result == [
        None
    ], f"Expected no more suggestions (all grid points covered), got {result}"


def test_dict_valued_param_without_injection():
    """Baseline: runs with clean configs (no injected keys) should be
    recognized as covering their grid points."""
    grid_values = [
        {"env_id": "Env-v0", "max_steps": 1000},
        {"env_id": "Env-v1", "max_steps": 2000},
    ]
    sweep_config = _make_grid_config("env_params", grid_values)

    runs = [
        _make_run(
            {"env_params": {"value": {"env_id": "Env-v0", "max_steps": 1000}}},
            name="run-0",
        ),
        _make_run(
            {"env_params": {"value": {"env_id": "Env-v1", "max_steps": 2000}}},
            name="run-1",
        ),
    ]

    result = grid_search_next_runs(runs, sweep_config)
    assert result == [None]


def test_dict_valued_param_missing_run_still_suggested():
    """When only some grid points are covered (even with injected keys),
    the uncovered point should still be suggested."""
    grid_values = [
        {"env_id": "Env-v0", "max_steps": 1000},
        {"env_id": "Env-v1", "max_steps": 2000},
        {"env_id": "Env-v2", "max_steps": 3000},
    ]
    sweep_config = _make_grid_config("env_params", grid_values)

    runs = [
        _make_run(
            {
                "env_params": {
                    "value": {
                        "env_id": "Env-v0",
                        "max_steps": 1000,
                        "env_name": "SAC_abc",
                    }
                }
            },
            name="run-0",
        ),
        _make_run(
            {
                "env_params": {
                    "value": {
                        "env_id": "Env-v2",
                        "max_steps": 3000,
                        "env_name": "SAC_def",
                    }
                }
            },
            name="run-2",
        ),
    ]

    result = grid_search_next_runs(runs, sweep_config)
    assert len(result) == 1
    assert result[0] is not None
    assert result[0].config["env_params"]["value"] == {
        "env_id": "Env-v1",
        "max_steps": 2000,
    }


def test_dict_valued_param_multiple_injected_keys():
    """Multiple extra keys injected should all be stripped."""
    grid_values = [
        {"env_id": "Env-v0", "reward_scale": 0.5},
    ]
    sweep_config = _make_grid_config("env_params", grid_values)

    runs = [
        _make_run(
            {
                "env_params": {
                    "value": {
                        "env_id": "Env-v0",
                        "reward_scale": 0.5,
                        "env_name": "SAC_run1",
                        "created_at": "2026-03-27",
                        "internal_id": 42,
                    }
                }
            },
            name="run-0",
        ),
    ]

    result = grid_search_next_runs(runs, sweep_config)
    assert result == [None]


# --- Scalar-valued parameter tests ---


def test_scalar_param_unaffected():
    """Scalar-valued parameters should work exactly as before."""
    sweep_config = _make_grid_config("learning_rate", [0.001, 0.01, 0.1])

    runs = [
        _make_run({"learning_rate": {"value": 0.001}}, name="run-0"),
        _make_run({"learning_rate": {"value": 0.01}}, name="run-1"),
    ]

    result = grid_search_next_runs(runs, sweep_config)
    assert len(result) == 1
    assert result[0] is not None
    assert result[0].config["learning_rate"]["value"] == 0.1


def test_scalar_param_all_covered():
    """All scalar grid points covered -> no suggestions."""
    sweep_config = _make_grid_config("batch_size", [16, 32, 64])

    runs = [
        _make_run({"batch_size": {"value": 16}}, name="run-0"),
        _make_run({"batch_size": {"value": 32}}, name="run-1"),
        _make_run({"batch_size": {"value": 64}}, name="run-2"),
    ]

    result = grid_search_next_runs(runs, sweep_config)
    assert result == [None]


# --- Mixed parameter tests ---


def test_mixed_scalar_and_dict_params():
    """Grid with both scalar and dict-valued params, dict has injected keys."""
    sweep_config = {
        "method": "grid",
        "parameters": {
            "algorithm": {"value": "SAC"},
            "env_params": {
                "values": [
                    {"env_id": "Env-v0", "max_steps": 1000},
                    {"env_id": "Env-v1", "max_steps": 2000},
                ],
            },
        },
    }

    runs = [
        _make_run(
            {
                "algorithm": {"value": "SAC"},
                "env_params": {
                    "value": {
                        "env_id": "Env-v0",
                        "max_steps": 1000,
                        "env_name": "SAC_abc",
                    }
                },
            },
            name="run-0",
        ),
        _make_run(
            {
                "algorithm": {"value": "SAC"},
                "env_params": {
                    "value": {
                        "env_id": "Env-v1",
                        "max_steps": 2000,
                        "env_name": "SAC_def",
                    }
                },
            },
            name="run-1",
        ),
    ]

    result = grid_search_next_runs(runs, sweep_config)
    assert result == [None]


def test_nested_dict_value_with_injected_key():
    """Dict values with nested dicts should also be handled correctly."""
    grid_values = [
        {"env_id": "Env-v0", "reward_kwargs": {"lambda_energy": 0.001}},
    ]
    sweep_config = _make_grid_config("env_params", grid_values)

    runs = [
        _make_run(
            {
                "env_params": {
                    "value": {
                        "env_id": "Env-v0",
                        "reward_kwargs": {"lambda_energy": 0.001},
                        "env_name": "SAC_injected",
                    }
                }
            },
            name="run-0",
        ),
    ]

    result = grid_search_next_runs(runs, sweep_config)
    assert result == [None]


def test_empty_runs_suggests_first_grid_point():
    """No runs -> suggest the first grid point (regression guard)."""
    grid_values = [
        {"env_id": "Env-v0", "max_steps": 1000},
        {"env_id": "Env-v1", "max_steps": 2000},
    ]
    sweep_config = _make_grid_config("env_params", grid_values)

    result = grid_search_next_runs([], sweep_config)
    assert len(result) == 1
    assert result[0] is not None
    assert result[0].config["env_params"]["value"] == {
        "env_id": "Env-v0",
        "max_steps": 1000,
    }
