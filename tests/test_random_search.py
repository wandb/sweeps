import random

import numpy as np
import pytest

from sweeps import random_search

sweep_config_2params = {
    "parameters": {"v1": {"min": 3, "max": 5}, "v2": {"min": 5, "max": 6}}
}


sweep_config_bad = {
    "parameters": {"v1": {"max": 3, "min": 5}, "v2": {"min": 5, "max": 6}}
}


def test_rand_single():
    random.seed(73)
    np.random.seed(73)
    gs = random_search.RandomSearch()
    runs = []
    sweep = {"config": sweep_config_2params, "runs": runs}
    params, info = gs.next_run(sweep)
    assert info is None
    assert params["v1"]["value"] == 3 and params["v2"]["value"] == 6


def test_rand_bad():
    random.seed(73)
    np.random.seed(73)
    gs = random_search.RandomSearch()
    runs = []
    sweep = {"config": sweep_config_bad, "runs": runs}
    with pytest.raises(ValueError):
        _ = gs.next_run(sweep)
