import random

import numpy as np
from ..run import next_run, next_runs


def reset_seed():
    np.random.seed(12)
    random.seed(13)


def test_batch_bayes_search_same_answer_as_single_search(
    sweep_config_bayes_search_2params_with_metric,
):
    reset_seed()
    runs = []
    for _ in range(10):
        suggestion = next_run(sweep_config_bayes_search_2params_with_metric, runs)
        runs.append(suggestion)

    reset_seed()
    suggestions = next_runs(sweep_config_bayes_search_2params_with_metric, [], n=10)
    assert suggestions == runs


def test_batch_grid_search_same_answer_as_single_search(
    sweep_config_2params_grid_search,
):
    runs = []
    for _ in range(7):
        suggestion = next_run(sweep_config_2params_grid_search, runs)
        runs.append(suggestion)

    suggestions = next_runs(sweep_config_2params_grid_search, [], n=7)

    assert len(suggestions) == len(runs)
    for s in suggestions:
        assert s in runs


def test_batch_random_search_same_answer_as_single_search(
    sweep_config_bayes_search_2params_with_metric,
):

    config = {
        "method": "random",
        "parameters": {"v1": {"values": [1, 2, 3]}, "v2": {"values": [4, 5]}},
    }

    reset_seed()
    runs = []
    for _ in range(10):
        suggestion = next_run(config, runs)
        runs.append(suggestion)

    reset_seed()
    suggestions = next_runs(config, [], n=10)
    assert suggestions == runs
