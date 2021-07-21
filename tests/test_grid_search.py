import pytest
import itertools

from typing import List
from ..run import RunState, SweepRun, next_run
from ..config import SweepConfig


def kernel_for_grid_search_tests(
    runs: List[SweepRun], config: SweepConfig, randomize: bool
) -> None:
    """This kernel assumes that sweep config has two categorical parameters
    named v1 and v2."""

    answers = sorted(
        list(
            itertools.product(
                config["parameters"]["v1"]["values"],
                config["parameters"]["v2"]["values"],
            )
        )
    )
    suggested_parameters = [
        (run.config["v1"]["value"], run.config["v2"]["value"]) for run in runs
    ]

    while True:
        suggestion = next_run(config, runs, randomize_order=randomize)
        if suggestion is None:  # done
            break
        assert suggestion.search_info is None
        assert suggestion.state == RunState.pending
        runs.append(suggestion)
        suggested_parameters.append(
            (suggestion.config["v1"]["value"], suggestion.config["v2"]["value"])
        )

    # assert that the grid search iterates over all possible parameters and stops when
    # it exhausts the list of possibilities. do not assert anything about the order
    # in which parameter suggestions are made.
    suggested_parameters = sorted(suggested_parameters)
    assert answers == suggested_parameters


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
