from .config.cfg import SweepConfig
from .run import SweepRun
from .params import HyperParameterSet

from typing import Union


def random_search_next_run(sweep_config: Union[dict, SweepConfig]) -> SweepRun:
    """Suggest runs with Hyperparameters sampled randomly from specified distributions.

    >>> suggestion = random_search_next_run({'method': 'random', 'parameters': {'a': {'min': 1., 'max': 2.}}})

    Args:
        sweep_config: The sweep's config.
    """

    # ensure that the sweepconfig is properly formatted
    sweep_config = SweepConfig(sweep_config)

    if sweep_config["method"] != "random":
        raise ValueError("Invalid sweep configuration for random_search_next_run.")

    params = HyperParameterSet.from_config(sweep_config["parameters"])

    for param in params:
        param.value = param.sample()

    return SweepRun(config=params.to_config())
