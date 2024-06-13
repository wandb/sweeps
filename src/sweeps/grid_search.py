import hashlib
import itertools
import random
import typing
from typing import Any, List, Optional, Union

import numpy as np
import yaml

from . import util
from .config.cfg import SweepConfig
from .params import HyperParameter, HyperParameterSet
from .run import SweepRun


def yaml_hash(value: Any) -> str:
    if isinstance(value, float):
        # Convert integer floats to ints, so that e.g. 3000000.0 == 3000000
        # Generally this isn't a problem, but when a run config value is something like "3e6" it's interpreted as a float
        # when creating the hyperparameter set from the sweep config, but becomes an int when it becomes part of the
        # run config in AgentHeartbeat (this happens in the line "configStr := string(config)" in mutation.go)
        if value.is_integer():
            value = int(value)
    return hashlib.md5(
        yaml.dump(value, default_flow_style=True, sort_keys=True).encode("ascii")
    ).hexdigest()


def grid_search_next_runs(
    runs: List[SweepRun],
    sweep_config: Union[dict, SweepConfig],
    validate: bool = False,
    n: int = 1,
    randomize_order: bool = False,
) -> List[Optional[SweepRun]]:
    """Suggest runs with Hyperparameters drawn from a grid.

    >>> suggestion = grid_search_next_runs([], {'method': 'grid', 'parameters': {'a': {'values': [1, 2, 3]}}})
    >>> assert suggestion[0].config['a']['value'] == 1

    Args:
        runs: The runs in the sweep.
        sweep_config: The sweep's config.
        randomize_order: Whether to randomize the order of the grid search.
        n: The number of runs to draw
        validate: Whether to validate `sweep_config` against the SweepConfig JSONschema.
           If true, will raise a Validation error if `sweep_config` does not conform to
           the schema. If false, will attempt to run the sweep with an unvalidated schema.

    Returns:
        The suggested runs.
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
        if p.type not in [
            HyperParameter.CATEGORICAL,
            HyperParameter.CONSTANT,
            HyperParameter.INT_UNIFORM,
            HyperParameter.Q_UNIFORM,
        ]:
            raise ValueError(
                f"Parameter {p.name} is a disallowed type with grid search. Grid search requires all parameters "
                f"to be categorical, constant, int_uniform, or q_uniform. Specification of probabilities for "
                f"categorical parameters is disallowed in grid search"
            )

    # convert bounded int_uniform and q_uniform parameters to categorical parameters
    for i, p in enumerate(params):
        if p.type == HyperParameter.INT_UNIFORM:
            params[i] = HyperParameter(
                p.name,
                {
                    "distribution": "categorical",
                    "values": [
                        val for val in range(p.config["min"], p.config["max"] + 1)
                    ],
                },
            )
        elif p.type == HyperParameter.Q_UNIFORM:
            params[i] = HyperParameter(
                p.name,
                {
                    "distribution": "categorical",
                    "values": np.arange(
                        p.config["min"], p.config["max"], p.config["q"]
                    ).tolist(),
                },
            )

    # we can only deal with discrete params in a grid search
    discrete_params = HyperParameterSet(
        [p for p in params if p.type == HyperParameter.CATEGORICAL]
    )

    # build an iterator over all combinations of param values
    param_names = [p.name for p in discrete_params]
    param_values = [p.config["values"] for p in discrete_params]
    param_hashes = [
        [yaml_hash(value) for value in p.config["values"]] for p in discrete_params
    ]
    value_hash_lookup = {
        name: dict(zip(hashes, vals))
        for name, vals, hashes in zip(param_names, param_values, param_hashes)
    }

    all_param_hashes = list(itertools.product(*param_hashes))
    if randomize_order:
        random.shuffle(all_param_hashes)

    param_hashes_seen: typing.Set[typing.Tuple] = set()
    for run in runs:
        hashes: typing.List[str] = []
        for name in param_names:
            nested_key: typing.List[str] = name.split(
                HyperParameterSet.NESTING_DELIMITER
            )
            nested_key.insert(1, "value")

            if util.dict_has_nested_key(run.config, nested_key):
                hashes.append(yaml_hash(util.get_nested_value(run.config, nested_key)))
        param_hashes_seen.add(tuple(hashes))

    hash_gen = (
        hash_val for hash_val in all_param_hashes if hash_val not in param_hashes_seen
    )

    retval: List[Optional[SweepRun]] = []
    for _ in range(n):

        # this is O(1)
        next_hash = next(hash_gen, None)

        # we have searched over the entire parameter space
        if next_hash is None:
            retval.append(None)
            return retval

        for param, hash_val in zip(discrete_params, next_hash):
            param.value = value_hash_lookup[param.name][hash_val]

        run = SweepRun(config=params.to_config())
        retval.append(run)

        param_hashes_seen.add(next_hash)

    return retval
