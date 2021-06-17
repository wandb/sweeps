import itertools
import random
from typing import List, Optional, Union

from .config.cfg import SweepConfig
from .run import SweepRun
from .params import HyperParameter, HyperParameterSet


def grid_search_next_run(
    runs: List[SweepRun],
    sweep_config: Union[dict, SweepConfig],
    validate: bool = False,
    randomize_order: bool = False,
) -> Optional[SweepRun]:
    """Suggest runs with Hyperparameters drawn from a grid.

    >>> suggestion = grid_search_next_run([], {'method': 'grid', 'parameters': {'a': {'values': [1, 2, 3]}}})
    >>> assert suggestion.config['a']['value'] == 1

    Args:
        runs: The runs in the sweep.
        sweep_config: The sweep's config.
        randomize_order: Whether to randomize the order of the grid search.
        validate: Whether to validate `sweep_config` against the SweepConfig JSONschema.
           If true, will raise a Validation error if `sweep_config` does not conform to
           the schema. If false, will attempt to run the sweep with an unvalidated schema.

    Returns:
        The suggested run.
    """

    # make sure the sweep config is valid
    if validate:
        sweep_config = SweepConfig(sweep_config)

    if sweep_config["method"] != "grid":
        raise ValueError("Invalid sweep configuration for grid_search_next_run.")

    if "parameters" not in sweep_config:
        raise ValueError('Grid search requires "parameters" section')
    params = HyperParameterSet.from_config(sweep_config["parameters"])

    # Check that all parameters are categorical or constant
    for p in params:
        if p.type != HyperParameter.CATEGORICAL and p.type != HyperParameter.CONSTANT:
            raise ValueError(
                "Parameter %s is a disallowed type with grid search. Grid search requires all parameters to be categorical or constant"
                % p.name
            )

    # we can only deal with discrete params in a grid search
    discrete_params = HyperParameterSet(
        [p for p in params if p.type == HyperParameter.CATEGORICAL]
    )

    # build an iterator over all combinations of param values
    param_names = [p.name for p in discrete_params]
    param_values = [p.config["values"] for p in discrete_params]

    all_param_values = set(itertools.product(*param_values))
    param_values_seen = set(
        [tuple(run.config[name]["value"] for name in param_names) for run in runs]
    )

    # this is O(N) due to the O(1) complexity of individual hash lookups; previous implementation was O(N^2)
    remaining_params = list(all_param_values - param_values_seen)

    if randomize_order:
        random.shuffle(remaining_params)

    # we have searched over the entire parameter space
    if len(remaining_params) == 0:
        return None

    for param, value in zip(discrete_params, remaining_params[0]):
        param.value = value

    return SweepRun(config=discrete_params.to_config())
