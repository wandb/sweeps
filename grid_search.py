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
    discrete_params = [p for p in params if p.type == HyperParameter.CATEGORICAL]

    # build an iterator over all combinations of param values
    param_names = [p.name for p in discrete_params]
    param_values = [p.config["values"] for p in discrete_params]
    param_value_set = list(itertools.product(*param_values))

    if randomize_order:
        random.shuffle(param_value_set)

    new_value_set = next(
        (
            value_set
            for value_set in param_value_set
            # check if parameter set is contained in some run
            if not _runs_contains_param_values(runs, dict(zip(param_names, value_set)))
        ),
        None,
    )

    # handle the case where we couldn't find a unique parameter set
    if new_value_set is None:
        return None

    # set next_run_params based on our new set of params
    for param, value in zip(discrete_params, new_value_set):
        param.value = value

    return SweepRun(config=params.to_config())


def _run_contains_param_values(run, params):
    for key, value in params.items():
        if key not in run.config:
            return False
        if not run.config[key]["value"] == value:
            return False
    return True


def _runs_contains_param_values(runs, params):
    return any(_run_contains_param_values(run, params) for run in runs)
