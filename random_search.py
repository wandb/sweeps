from .config.cfg import SweepConfig
from .run import SweepRun
from .params import HyperParameterSet

from typing import Union


def random_search_next_run(
    sweep_config: Union[dict, SweepConfig], validate: bool = False
) -> SweepRun:
    """Suggest runs with Hyperparameters sampled randomly from specified distributions.

    >>> suggestion = random_search_next_run({'method': 'random', 'parameters': {'a': {'min': 1., 'max': 2.}}})

    Args:
        sweep_config: The sweep's config.
        validate: Whether to validate `sweep_config` against the SweepConfig JSONschema.
           If true, will raise a Validation error if `sweep_config` does not conform to
           the schema. If false, will attempt to run the sweep with an unvalidated schema.

    Returns:
        The suggested run.
    """

    # ensure that the sweepconfig is properly formatted
    if validate:
        sweep_config = SweepConfig(sweep_config)

    if sweep_config["method"] != "random":
        raise ValueError("Invalid sweep configuration for random_search_next_run.")

    params = HyperParameterSet.from_config(sweep_config["parameters"])

    for param in params:
        param.value = param.sample()

    return SweepRun(config=params.to_config())
