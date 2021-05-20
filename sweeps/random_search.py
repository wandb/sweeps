"""Random Search."""

from .config.cfg import SweepConfig
from .run import Run
from .params import HyperParameterSet


def random_search_next_run(config: SweepConfig) -> Run:
    params = HyperParameterSet.from_config(config["parameters"])

    for param in params:
        param.value = param.sample()

    return Run(config=params.to_config())
