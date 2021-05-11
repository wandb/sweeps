import pytest


@pytest.fixture()
def sweep_config_2params():
    return {
        "metric": {"name": "loss"},
        "parameters": {"v1": {"min": 1, "max": 10}, "v2": {"min": 1, "max": 10}},
    }


@pytest.fixture()
def sweep_config_2params_acc():
    return {
        "metric": {
            "name": "acc",
            "goal": "maximize",
        },
        "parameters": {"v1": {"min": 1, "max": 10}, "v2": {"min": 1, "max": 10}},
    }


@pytest.fixture()
def sweep_config_2params_categorical():
    return {
        "metric": {
            "name": "acc",
            "goal": "maximize",
        },
        "parameters": {
            "v1": {
                "distribution": "categorical",
                "values": [(2, 3), [3, 4], ["5", "6"], [(7, 8), ["9", [10, 11]]]],
            },
            "v2": {"min": 1, "max": 10},
        },
    }
