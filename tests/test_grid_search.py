import pytest
import itertools

from typing import List
from sweeps.run import RunState, Run
from sweeps.grid_search import grid_search_next_run
from sweeps.config import SweepConfig


def kernel_for_grid_search_tests(
    runs: List[Run], config: SweepConfig, randomize: bool
) -> None:
    answers = sorted(
        list(
            itertools.product(
                config["parameters"]["v1"]["values"],
                config["parameters"]["v2"]["values"],
            )
        )
    )
    suggested_parameters = [(run.config["v1"], run.config["v2"]) for run in runs]

    while True:
        next_run = grid_search_next_run(runs, config, randomize_order=randomize)
        if next_run is None:  # done
            break
        assert next_run.optimizer_info is None
        assert next_run.state == RunState.proposed
        runs.append(next_run)
        suggested_parameters.append((next_run.config["v1"], next_run.config["v2"]))

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
    runs = [Run(config={"v1": 2, "v2": 4}), Run(config={"v1": 1, "v2": 5})]
    kernel_for_grid_search_tests(
        runs, sweep_config_2params_grid_search, randomize=randomize
    )
