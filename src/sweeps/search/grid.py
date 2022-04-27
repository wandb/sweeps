import hashlib
import itertools
from typing import Any, List, Optional, Sequence

import numpy as np
import yaml

from ..params import (
    HyperParameter,
    HyperParameterSet,
    make_run_config_from_params,
)
from ..run import SweepRun
from .abstract import AbstractSearch


def yaml_hash(value: Any) -> str:
    return hashlib.md5(
        yaml.dump(value, default_flow_style=True, sort_keys=True).encode("ascii")
    ).hexdigest()


class GridSearch(AbstractSearch):
    """Suggest runs with Hyperparameters drawn from a grid of all possible values."""

    def _next_runs(
        self,
        runs: List[SweepRun],
        *args,
        n: int = 1,
        randomize_order: bool = False,
        **kwargs,
    ) -> Sequence[Optional[SweepRun]]:  # type: ignore

        # Check that all parameters are categorical or constant
        for p in self.params:
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
        for i, p in enumerate(self.params):
            if p.type == HyperParameter.INT_UNIFORM:
                self.params[i] = HyperParameter(
                    p.name,
                    {
                        "distribution": "categorical",
                        "values": [
                            val for val in range(p.config["min"], p.config["max"] + 1)
                        ],
                    },
                )
            elif p.type == HyperParameter.Q_UNIFORM:
                self.params[i] = HyperParameter(
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
            [p for p in self.params if p.type == HyperParameter.CATEGORICAL]
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
            np.random.shuffle(all_param_hashes)

        param_hashes_seen = set(
            [
                tuple(
                    yaml_hash(run.config[name]["value"])
                    for name in param_names
                    if name in run.config
                )
                for run in runs
            ]
        )

        hash_gen = (
            hash_val
            for hash_val in all_param_hashes
            if hash_val not in param_hashes_seen
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

            run = SweepRun(config=make_run_config_from_params(self.params))
            retval.append(run)

            param_hashes_seen.add(next_hash)

        return retval
