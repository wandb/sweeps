import pytest
from ..config import SweepConfig


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
            "parameters": {
                "v1": {"min": 1, "max": 10},
                "v2": {"min": 1.0, "max": 10.0},
            },
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
