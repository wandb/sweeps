"""Random Search."""

from .config.cfg import SweepConfig
from .sweeprun import SweepRun
from .params import HyperParameterSet

from typing import Union


def random_search_next_run(sweep_config: Union[dict, SweepConfig]) -> SweepRun:
    # ensure that the sweepconfig is properly formatted
    sweep_config = SweepConfig(sweep_config)

    params = HyperParameterSet.from_config(sweep_config["parameters"])

    for param in params:
        param.value = param.sample()

    return SweepRun(config=params.to_config())
