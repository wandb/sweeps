import pytest
import itertools

from typing import List, Tuple
from ..run import RunState, SweepRun, next_run
from ..config import SweepConfig
from ..grid_search import yaml_hash


def kernel_for_grid_search_tests(
    runs: List[SweepRun],
    config: SweepConfig,
    randomize: bool,
) -> List[Tuple[str]]:
    """This kernel assumes that sweep config has two categorical parameters
    named v1 and v2."""

    answers = list(
        set(
            itertools.product(
                [yaml_hash(v) for v in config["parameters"]["v1"]["values"]],
                [yaml_hash(v) for v in config["parameters"]["v2"]["values"]],
            )
        )
    )
    suggested_parameters = [
        (
            yaml_hash(run.config["v1"]["value"]),
            yaml_hash(run.config["v2"]["value"]),
        )
        for run in runs
    ]

    while True:
        suggestion = next_run(config, runs, randomize_order=randomize)
        if suggestion is None:  # done
            break
        assert suggestion.search_info is None
        assert suggestion.state == RunState.pending
        runs.append(suggestion)
        suggested_parameters.append(
            (
                yaml_hash(suggestion.config["v1"]["value"]),
                yaml_hash(suggestion.config["v2"]["value"]),
            )
        )

    assert len(answers) == len(suggested_parameters)
    for key in suggested_parameters:
        assert key in answers

    return suggested_parameters


@pytest.mark.parametrize("randomize", [True, False])
def test_grid_from_start_with_and_without_randomize(
    sweep_config_2params_grid_search, randomize
):
    kernel_for_grid_search_tests(
        [], sweep_config_2params_grid_search, randomize=randomize
    )


@pytest.mark.parametrize("randomize", [True, False])
def test_grid_search_starting_from_in_progress(
    sweep_config_2params_grid_search, randomize
):
    runs = [
        SweepRun(config={"v1": {"value": 2}, "v2": {"value": 4}}),
        SweepRun(config={"v1": {"value": 1}, "v2": {"value": 5}}),
    ]
    kernel_for_grid_search_tests(
        runs, sweep_config_2params_grid_search, randomize=randomize
    )


def test_grid_search_with_list_values():
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
    )


def test_grid_search_duplicated_values_are_not_duplicated_in_answer():
    duplicated_config = SweepConfig(
        {
            "method": "grid",
            "parameters": {
                "v1": {"values": [None, 2, 3, "a", (2, 3), 3]},
                "v2": {"values": ["a", "b", "c'"]},
                "v3": {"value": 1},
            },
        }
    )

    runs = []
    kernel_for_grid_search_tests(runs, duplicated_config, randomize=True)
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


@pytest.mark.parametrize("randomize", [True, False])
def test_grid_search_dict_val_is_propagated(randomize):
    config_const = SweepConfig(
        {
            "method": "grid",
            "parameters": {
                "v1": {"values": ["a", "b", "c'"]},
                "v2": {
                    "values": [{"a": "b"}, {"c": "d"}, {"e": {"f": "g"}}, {"a": "b"}]
                },
            },
        }
    )

    hashes = kernel_for_grid_search_tests([], config_const, randomize=randomize)

    # assert that only 9 hyperparameter combos are iterated over, as there is one duplicate in v2
    assert len(hashes) == 3 * 3
