import hashlib
import itertools
import logging
import math
import random
import typing
from typing import Any, List, Optional, Union

import numpy as np
import yaml

from . import util
from .config.cfg import SweepConfig
from .params import HyperParameter, HyperParameterSet
from .run import SweepRun

logger = logging.getLogger(__name__)


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


def _freeze_value(value: Any) -> typing.Hashable:
    if isinstance(value, float):
        if math.isnan(value):
            return ("float", "nan")
        if value.is_integer():
            value = int(value)

    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                sorted(
                    (
                        (_freeze_value(key), _freeze_value(item_value))
                        for key, item_value in value.items()
                    ),
                    key=lambda item: repr(item[0]),
                )
            ),
        )

    if isinstance(value, list):
        return ("list", tuple(_freeze_value(item) for item in value))

    if isinstance(value, tuple):
        return ("tuple", tuple(_freeze_value(item) for item in value))

    if isinstance(value, (set, frozenset)):
        return (
            type(value).__name__,
            tuple(sorted((_freeze_value(item) for item in value), key=repr)),
        )

    try:
        hash(value)
    except TypeError:
        return ("repr", type(value).__name__, repr(value))

    return ("scalar", type(value).__name__, value)


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
    param_value_keys = [
        [_freeze_value(value) for value in values] for values in param_values
    ]

    # for dict-valued params, collect the union of keys across all grid point
    # values. this is used to strip runtime injected keys before freezing, so
    # that extra keys don't cause key mismatches and duplicate suggestions.
    param_known_keys: typing.Dict[str, typing.Set[str]] = {}
    for name, values in zip(param_names, param_values):
        keys: typing.Set[str] = set()
        for val in values:
            if isinstance(val, dict):
                keys.update(val.keys())
        if keys:
            param_known_keys[name] = keys

    value_key_lookup = {
        name: dict(zip(value_keys, vals))
        for name, vals, value_keys in zip(param_names, param_values, param_value_keys)
    }

    all_param_keys = list(itertools.product(*param_value_keys))
    if randomize_order:
        random.shuffle(all_param_keys)

    param_keys_seen: typing.Set[typing.Tuple[typing.Hashable, ...]] = set()
    expected_tuple_len = len(param_names)
    for run in runs:
        keys: typing.List[typing.Hashable] = []
        missing_params: typing.List[str] = []
        for name in param_names:
            nested_key: typing.List[str] = name.split(
                HyperParameterSet.NESTING_DELIMITER
            )
            nested_key.insert(1, "value")

            if util.dict_has_nested_key(run.config, nested_key):
                run_value = util.get_nested_value(run.config, nested_key)
                if name in param_known_keys and isinstance(run_value, dict):
                    run_value = {
                        k: v
                        for k, v in run_value.items()
                        if k in param_known_keys[name]
                    }
                keys.append(_freeze_value(run_value))
            else:
                missing_params.append(name)

        if len(keys) != expected_tuple_len:
            logger.warning(
                "grid_search_dedupe_incomplete_hash_tuple",
                extra={
                    "run_name": run.name,
                    "expected_params": expected_tuple_len,
                    "found_params": len(keys),
                    "missing_params": missing_params,
                    "config_top_level_keys": (
                        sorted(run.config.keys()) if run.config else []
                    ),
                },
            )

        param_keys_seen.add(tuple(keys))

    key_gen = (key for key in all_param_keys if key not in param_keys_seen)

    retval: List[Optional[SweepRun]] = []
    for _ in range(n):
        # Advance to the first grid point not already covered by a prior run.
        next_key = next(key_gen, None)

        # we have searched over the entire parameter space
        if next_key is None:
            retval.append(None)
            return retval

        for param, value_key in zip(discrete_params, next_key):
            param.value = value_key_lookup[param.name][value_key]

        run = SweepRun(config=params.to_config())
        retval.append(run)

        param_keys_seen.add(next_key)

    return retval
