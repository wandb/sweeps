from typing import List, Sequence, Tuple

import numpy as np
import pytest
from sweeps.config import SweepConfig
from sweeps.grid_search import yaml_hash
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
