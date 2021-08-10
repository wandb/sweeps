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
