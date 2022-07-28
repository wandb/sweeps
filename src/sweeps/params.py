"""Hyperparameter search parameters."""
import logging
import random
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import jsonschema
import numpy as np
import scipy.stats as stats

from ._types import ArrayLike
from .config import ParamValidationError, fill_parameter
from .run import SweepRun


def q_log_uniform_v1_ppf(x: ArrayLike, min, max, q):
    r = np.exp(stats.uniform.ppf(x, min, max - min))
    ret_val = np.round(r / q) * q
    if isinstance(q, int):
        return ret_val.astype(int)
    else:
        return ret_val


def inv_log_uniform_v1_ppf(x: ArrayLike, min, max):
    return np.exp(
        -stats.uniform.ppf(
            1 - x,
            min,
            max - min,
        )
    )


def loguniform_v1_ppf(x: ArrayLike, min, max):
    return np.exp(stats.uniform.ppf(x, min, max - min))


class HyperParameter:

    CONSTANT = "param_single_value"
    CATEGORICAL = "param_categorical"
    CATEGORICAL_PROB = "param_categorical_w_probabilities"
    INT_UNIFORM = "param_int_uniform"
    UNIFORM = "param_uniform"

    LOG_UNIFORM_V1 = "param_loguniform"
    LOG_UNIFORM_V2 = "param_loguniform_v2"

    INV_LOG_UNIFORM_V1 = "param_inv_loguniform"
    INV_LOG_UNIFORM_V2 = "param_inv_loguniform_v2"

    Q_UNIFORM = "param_quniform"

    Q_LOG_UNIFORM_V1 = "param_qloguniform"
    Q_LOG_UNIFORM_V2 = "param_qloguniform_v2"

    NORMAL = "param_normal"
    Q_NORMAL = "param_qnormal"
    LOG_NORMAL = "param_lognormal"
    Q_LOG_NORMAL = "param_qlognormal"
    BETA = "param_beta"
    Q_BETA = "param_qbeta"

    def __init__(self, name: str, config: dict):
        """A hyperparameter to optimize.

        >>> parameter = HyperParameter('int_unif_distributed', {'min': 1, 'max': 10})
        >>> assert parameter.config['min'] == 1
        >>> parameter = HyperParameter('normally_distributed', {'distribution': 'normal'})
        >>> assert np.isclose(parameter.config['mu'], 0)

        Args:
            name: The name of the hyperparameter.
            config: Hyperparameter config dict.
        """

        self.name = name

        result = fill_parameter(name, config)
        if result is None:
            raise jsonschema.ValidationError(
                f"invalid hyperparameter configuration: {name}"
            )

        self.type, self.config = result

        if self.config is None or self.type is None:
            raise ValueError(
                "list of allowed schemas has length zero; please provide some valid schemas"
            )

        self.value = (
            None if self.type != HyperParameter.CONSTANT else self.config["value"]
        )

    def value_to_idx(self, value: Any) -> int:
        """Get the index of the value of a categorically distributed HyperParameter.

        >>> parameter = HyperParameter('a', {'values': [1, 2, 3]})
        >>> assert parameter.value_to_idx(2) == 1

        Args:
             value: The value to look up.

        Returns:
            The index of the value.
        """

        if self.type != HyperParameter.CATEGORICAL:
            raise ValueError("Can only call value_to_idx on categorical variable")

        for ii, test_value in enumerate(self.config["values"]):
            if value == test_value:
                return ii

        raise ValueError(
            f"{value} is not a permitted value of the categorical hyperparameter {self.name} "
            f"in the current sweep."
        )

    def cdf(self, x: ArrayLike) -> ArrayLike:
        """Cumulative distribution function (CDF).

        In probability theory and statistics, the cumulative distribution function
        (CDF) of a real-valued random variable X, is the probability that X will
        take a value less than or equal to x.

        Args:
             x: Parameter values to calculate the CDF for. Can be scalar or 1-d.
        Returns:
            Probability that a random sample of this hyperparameter will be less
            than or equal to x.
        """
        if self.type == HyperParameter.CONSTANT:
            return np.zeros_like(x)
        elif self.type == HyperParameter.CATEGORICAL:
            # NOTE: Indices expected for categorical parameters, not values.
            return stats.randint.cdf(x, 0, len(self.config["values"]))
        elif self.type == HyperParameter.CATEGORICAL_PROB:
            return np.cumsum(self.config["probabilities"])[x]
        elif self.type == HyperParameter.INT_UNIFORM:
            return stats.randint.cdf(x, self.config["min"], self.config["max"] + 1)
        elif (
            self.type == HyperParameter.UNIFORM or self.type == HyperParameter.Q_UNIFORM
        ):
            return stats.uniform.cdf(
                x, self.config["min"], self.config["max"] - self.config["min"]
            )
        elif (
            self.type == HyperParameter.LOG_UNIFORM_V1
            or self.type == HyperParameter.Q_LOG_UNIFORM_V1
        ):
            return stats.uniform.cdf(
                np.log(x), self.config["min"], self.config["max"] - self.config["min"]
            )
        elif (
            self.type == HyperParameter.LOG_UNIFORM_V2
            or self.type == HyperParameter.Q_LOG_UNIFORM_V2
        ):
            return stats.uniform.cdf(
                np.log(x),
                np.log(self.config["min"]),
                np.log(self.config["max"]) - np.log(self.config["min"]),
            )
        elif self.type == HyperParameter.INV_LOG_UNIFORM_V1:
            return 1 - stats.uniform.cdf(
                np.log(1 / x),
                self.config["min"],
                self.config["max"] - self.config["min"],
            )
        elif self.type == HyperParameter.INV_LOG_UNIFORM_V2:
            return 1 - stats.uniform.cdf(
                np.log(1 / x),
                -np.log(self.config["max"]),
                np.abs(np.log(self.config["max"]) - np.log(self.config["min"])),
            )
        elif self.type == HyperParameter.NORMAL or self.type == HyperParameter.Q_NORMAL:
            return stats.norm.cdf(x, loc=self.config["mu"], scale=self.config["sigma"])
        elif (
            self.type == HyperParameter.LOG_NORMAL
            or self.type == HyperParameter.Q_LOG_NORMAL
        ):
            return stats.lognorm.cdf(
                x, s=self.config["sigma"], scale=np.exp(self.config["mu"])
            )
        elif self.type == HyperParameter.BETA or self.type == HyperParameter.Q_BETA:
            return stats.beta.cdf(x, a=self.config["a"], b=self.config["b"])
        else:
            raise ValueError("Unsupported hyperparameter distribution type")

    def ppf(self, x: ArrayLike) -> Any:
        """Percentage point function (PPF).

        In probability theory and statistics, the percentage point function is
        the inverse of the CDF: it returns the value of a random variable at the
        xth percentile.

        Args:
             x: Percentiles of the random variable. Can be scalar or 1-d.
        Returns:
            Value of the random variable at the specified percentile.
        """
        if np.any((x < 0.0) | (x > 1.0)):
            raise ValueError("Can't call ppf on value outside of [0,1]")
        elif self.type == HyperParameter.CONSTANT:
            return self.config["value"]
        elif self.type == HyperParameter.CATEGORICAL:
            # Samples uniformly over the values
            retval = [
                self.config["values"][i]
                for i in np.atleast_1d(
                    stats.randint.ppf(x, 0, len(self.config["values"])).astype(int)
                ).tolist()
            ]
            if np.isscalar(x):
                return retval[0]
            return retval
        elif self.type == HyperParameter.CATEGORICAL_PROB:
            # Samples by specified categorical distribution if specified
            cdf = np.cumsum(self.config["probabilities"])
            if np.isscalar(x):
                return self.config["values"][np.argmin(x >= cdf, axis=-1)]
            else:
                return [
                    self.config["values"][i] for i in [np.argmin(cdf >= p) for p in x]
                ]
        elif self.type == HyperParameter.INT_UNIFORM:
            return (
                stats.randint.ppf(x, self.config["min"], self.config["max"] + 1)
                .astype(int)
                .tolist()
            )
        elif self.type == HyperParameter.UNIFORM:
            return stats.uniform.ppf(
                x, self.config["min"], self.config["max"] - self.config["min"]
            )
        elif self.type == HyperParameter.Q_UNIFORM:
            r = stats.uniform.ppf(
                x, self.config["min"], self.config["max"] - self.config["min"]
            )
            ret_val = np.round(r / self.config["q"]) * self.config["q"]
            if isinstance(self.config["q"], int):
                return ret_val.astype(int)
            else:
                return ret_val
        elif self.type == HyperParameter.LOG_UNIFORM_V1:
            return loguniform_v1_ppf(x, self.config["min"], self.config["max"])
        elif self.type == HyperParameter.LOG_UNIFORM_V2:
            return loguniform_v1_ppf(
                x, np.log(self.config["min"]), np.log(self.config["max"])
            )
        elif self.type == HyperParameter.INV_LOG_UNIFORM_V1:
            return inv_log_uniform_v1_ppf(x, self.config["min"], self.config["max"])
        elif self.type == HyperParameter.INV_LOG_UNIFORM_V2:
            return inv_log_uniform_v1_ppf(
                x, -np.log(self.config["max"]), -np.log(self.config["min"])
            )
        elif self.type == HyperParameter.Q_LOG_UNIFORM_V1:
            return q_log_uniform_v1_ppf(
                x, self.config["min"], self.config["max"], self.config["q"]
            )
        elif self.type == HyperParameter.Q_LOG_UNIFORM_V2:
            return q_log_uniform_v1_ppf(
                x,
                np.log(self.config["min"]),
                np.log(self.config["max"]),
                self.config["q"],
            )
        elif self.type == HyperParameter.NORMAL:
            return stats.norm.ppf(x, loc=self.config["mu"], scale=self.config["sigma"])
        elif self.type == HyperParameter.Q_NORMAL:
            r = stats.norm.ppf(x, loc=self.config["mu"], scale=self.config["sigma"])
            ret_val = np.round(r / self.config["q"]) * self.config["q"]
            if isinstance(self.config["q"], int):
                return ret_val.astype(int)
            else:
                return ret_val
        elif self.type == HyperParameter.LOG_NORMAL:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html
            return stats.lognorm.ppf(
                x, s=self.config["sigma"], scale=np.exp(self.config["mu"])
            )
        elif self.type == HyperParameter.Q_LOG_NORMAL:
            r = stats.lognorm.ppf(
                x, s=self.config["sigma"], scale=np.exp(self.config["mu"])
            )
            ret_val = np.round(r / self.config["q"]) * self.config["q"]

            if isinstance(self.config["q"], int):
                return ret_val.astype(int)
            else:
                return ret_val

        elif self.type == HyperParameter.BETA:
            return stats.beta.ppf(x, a=self.config["a"], b=self.config["b"])
        elif self.type == HyperParameter.Q_BETA:
            r = stats.beta.ppf(x, a=self.config["a"], b=self.config["b"])
            ret_val = np.round(r / self.config["q"]) * self.config["q"]
            if isinstance(self.config["q"], int):
                return ret_val.astype(int)
            else:
                return ret_val
        else:
            raise ValueError("Unsupported hyperparameter distribution type")

    def sample(self) -> Any:
        """Randomly sample a value from the distribution of this HyperParameter."""
        return self.ppf(random.uniform(0.0, 1.0))

    def _to_config(self) -> Tuple[str, Dict]:
        config = dict(value=self.value)
        return self.name, config

    def _name_and_value(self) -> Tuple[str, Any]:
        return self.name, self.value


class HyperParameterSet(list):

    NESTING_DELIMITER: str = ".wbnest[599c28ca25da]."

    def __init__(self, items: List[HyperParameter]):
        """A set of HyperParameters.

        >>> hp1 = HyperParameter('a', {'values': [1, 2, 3]})
        >>> hp2 = HyperParameter('b', {'distribution': 'normal'})
        >>> HyperParameterSet([hp1, hp2])

        Args:
            items: A list of HyperParameters to construct the set from.
        """
        self.searchable_params: List[HyperParameter] = []
        self.param_names_to_index: Dict[str, int] = dict()
        self.param_names_to_param: Dict[str, HyperParameter] = dict()

        _searchable_param_index: int = 0
        for item in items:
            if not isinstance(item, HyperParameter):
                raise TypeError(
                    f"Every item in HyperParameterSet must be a HyperParameter, got {item} of type {type(item)}"
                )
            elif not item.type == HyperParameter.CONSTANT:
                # constants do not form part of the search space
                self.searchable_params.append(item)
                self.param_names_to_index[item.name] = _searchable_param_index
                self.param_names_to_param[item.name] = item
                _searchable_param_index += 1

        super().__init__(items)

    @classmethod
    def from_config(cls, config: Dict):
        """Instantiate a HyperParameterSet based the parameters section of a SweepConfig.

        >>> sweep_config = {'method': 'grid', 'parameters': {'a': {'values': [1, 2, 3]}}}
        >>> hps = HyperParameterSet.from_config(sweep_config['parameters'])

        Args:
            config: The parameters section of a SweepConfig.
        """
        hyperparameters: List[HyperParameter] = []

        def _unnest(d: Dict, prefix: str = ""):
            """Recursively search for HyperParameters in a potentially nested dictionary."""
            for key, val in sorted(d.items()):
                assert isinstance(
                    key, str
                ), f"Sweep config keys must be strings, found {key} of type {type(key)}"
                assert isinstance(
                    val, dict
                ), f"Sweep config values must be dicts, found {val} of type {type(val)}"
                try:
                    _hp = HyperParameter(f"{prefix}{key}", val)
                except ParamValidationError:
                    assert (
                        "parameters" in val
                    ), "Param of type DICT must have 'parameters' key"
                    _unnest(
                        val["parameters"],
                        prefix=f"{prefix}{key}{cls.NESTING_DELIMITER}",
                    )
                else:
                    hyperparameters.append(_hp)

        _unnest(config)
        return cls(hyperparameters)

    def to_config(self) -> Dict:
        """Convert a HyperParameterSet to a SweepRun config."""

        def _renest(d: Dict) -> None:
            """Nest a flattened dict based on a delimiter."""
            if isinstance(d, dict):
                for k in sorted(d.keys()):
                    assert isinstance(
                        k, str
                    ), f"Sweep config keys must be strings, found {k} of type {type(k)}"
                    if self.NESTING_DELIMITER in k:
                        subdict: Union[Any, Dict] = d
                        subkeys: List[str] = k.split(self.NESTING_DELIMITER)
                        for i, subkey in enumerate(subkeys[:-1]):
                            if subkey in subdict:
                                subdict = subdict[subkey]
                                if not isinstance(subdict, dict):
                                    conflict_key: str = self.NESTING_DELIMITER.join(
                                        subkeys[: i + 1]
                                    )
                                    raise ValueError(
                                        f"While nesting, found key {subkey} which conflics with key {conflict_key}"
                                    )
                            else:
                                # Create a nested dictionary under the parent key
                                _d: Dict = dict()
                                subdict[subkey] = _d
                                subdict = _d
                        if isinstance(subdict, dict):
                            subdict[subkeys[-1]] = d.pop(k)

        # Add only the hyperparameters which aren't dicts
        config: Dict = dict()
        for param in self:
            _name, _value = param._name_and_value()
            config[_name] = _value
        config = deepcopy(config)
        _renest(config)
        # Because of historical reason the first level of nesting requires "value" key
        for k, v in config.items():
            config[k] = {"value": v}
        return config

    def normalize_runs_as_array(self, runs: List[SweepRun]) -> np.ndarray:
        """Normalize a list of SweepRuns to an ndarray of parameter vectors."""
        normalized_runs: np.ndarray = np.zeros([len(self.searchable_params), len(runs)])
        for param_name, idx in self.param_names_to_index.items():
            _param: HyperParameter = self.param_names_to_param[param_name]
            row: np.ndarray = np.zeros(len(runs))  # default to 0
            for i, run in enumerate(runs):
                if param_name in run.config:
                    _val = run.config[param_name]["value"]
                    if _param.type == HyperParameter.CATEGORICAL:
                        row[i] = _param.value_to_idx(_val)
                    else:
                        row[i] = _val
                else:
                    logging.warning(f"Run does not contain parameter {param_name}")
            if not np.all(np.isfinite(row)):
                logging.warning(f"Found non-finite value in normalized run row {row}")
            # Convert row to CDF, filter out NaNs
            non_nan_indices = ~np.isnan(row)
            normalized_runs[idx, non_nan_indices] = _param.cdf(row[non_nan_indices])
        return np.transpose(normalized_runs)


def make_param_log_deprecation_message(
    param_type: str, replacement_param_type: str
) -> str:
    from .config import schema

    param_schema = schema.dereferenced_sweep_config_jsonschema["definitions"][
        param_type
    ]
    deprecated_distribution_name = param_schema["properties"]["distribution"]["enum"][0]

    replacement_param_schema = schema.dereferenced_sweep_config_jsonschema[
        "definitions"
    ][replacement_param_type]
    replacement_distribution_name = replacement_param_schema["properties"][
        "distribution"
    ]["enum"][0]

    return (
        f"uses {deprecated_distribution_name}, where min/max specify base-e exponents. "
        f"Use {replacement_distribution_name} to specify limit values."
    )


PARAM_DEPRECATION_MAP = {
    HyperParameter.LOG_UNIFORM_V1: make_param_log_deprecation_message(
        HyperParameter.LOG_UNIFORM_V1, HyperParameter.LOG_UNIFORM_V2
    ),
    HyperParameter.INV_LOG_UNIFORM_V1: make_param_log_deprecation_message(
        HyperParameter.INV_LOG_UNIFORM_V1, HyperParameter.INV_LOG_UNIFORM_V2
    ),
    HyperParameter.Q_LOG_UNIFORM_V1: make_param_log_deprecation_message(
        HyperParameter.Q_LOG_UNIFORM_V1, HyperParameter.Q_LOG_UNIFORM_V2
    ),
}
