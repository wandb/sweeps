import pytest
from sweeps.config import SweepConfig


def pytest_addoption(parser):
    parser.addoption(
        "--plot",
        action="store_true",
        help="Plot true and predicted distributions "
        "for tests involving random sampling.",
    )


@pytest.fixture
def plot(request):
    return request.config.getoption("--plot")


@pytest.fixture()
def sweep_config_bayes_search_2params_with_metric():
    return SweepConfig(
        {
            "metric": {"name": "loss", "goal": "minimize"},
            "method": "bayes",
            "parameters": {"v1": {"min": 1, "max": 10}, "v2": {"min": 1.0, "max": 10}},
        }
    )


@pytest.fixture()
def sweep_config_2params_grid_search():
    return SweepConfig(
        {
            "method": "grid",
            "parameters": {"v1": {"values": [1, 2, 3]}, "v2": {"values": [4, 5]}},
        }
    )


@pytest.fixture()
def sweep_config_2params_acc():
    return SweepConfig(
        {
            "metric": {
                "name": "acc",
                "goal": "maximize",
            },
            "parameters": {"v1": {"min": 1, "max": 10}, "v2": {"min": 1, "max": 10}},
        }
    )


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
