import pytest

from typing import List, Sequence, Tuple
from ..run import RunState, SweepRun, next_run
from ..config import SweepConfig


def kernel_for_grid_search_tests(
    runs: List[SweepRun],
    config: SweepConfig,
    answers: Sequence[Tuple],
    randomize: bool,
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

    while True:
        suggestion = next_run(config, runs, randomize_order=randomize)
        if suggestion is None:  # done
            break
        assert suggestion.search_info is None
        assert suggestion.state == RunState.pending
        runs.append(suggestion)
        suggested_parameters.append(
            (
                suggestion.config["v1"]["value"],
                suggestion.config["v2"]["value"],
            )
        )

    assert len(answers) == len(suggested_parameters)
    for key in suggested_parameters:
        assert key in answers


@pytest.mark.parametrize("randomize", [True, False])
def test_grid_from_start_with_and_without_randomize(
    sweep_config_2params_grid_search, randomize
):
    kernel_for_grid_search_tests(
        [],
        sweep_config_2params_grid_search,
        randomize=randomize,
        answers=[(1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)],
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
        runs,
        sweep_config_2params_grid_search,
        randomize=randomize,
        answers=[(1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)],
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
        answers=[("", 256), ("", 512), ([9, 5], 256), ([9, 5], 512)],
    )


def test_grid_search_duplicated_values_are_not_duplicated_in_answer():
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


@pytest.mark.parametrize("randomize", [True, False])
def test_grid_search_dict_val_is_propagated(randomize):
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
