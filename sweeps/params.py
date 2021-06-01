"""Hyperparameter search parameters."""

import random

from typing import List, Tuple, Dict, Any

import numpy as np
import numpy.typing as npt
import scipy.stats as stats
from copy import deepcopy

import jsonschema

from .run import SweepRun
from .config.schema import (
    sweep_config_jsonschema,
    dereferenced_sweep_config_jsonschema,
    DefaultFiller,
    format_checker,
)


class HyperParameter:

    CONSTANT = "param_single_value"
    CATEGORICAL = "param_categorical"
    INT_UNIFORM = "param_int_uniform"
    UNIFORM = "param_uniform"
    LOG_UNIFORM = "param_loguniform"
    Q_UNIFORM = "param_quniform"
    Q_LOG_UNIFORM = "param_qloguniform"
    NORMAL = "param_normal"
    Q_NORMAL = "param_qnormal"
    LOG_NORMAL = "param_lognormal"
    Q_LOG_NORMAL = "param_qlognormal"
    BETA = "param_beta"
    Q_BETA = "param_qbeta"

    def __init__(self, name: str, config: dict):

        self.name = name

        # names of the parameter definitions that are allowed
        allowed_schemas = [
            d["$ref"].split("/")[-1]
            for d in sweep_config_jsonschema["definitions"]["parameter"]["anyOf"]
        ]

        valid = False
        for schema_name in allowed_schemas:
            # create a jsonschema object to validate against the subschema
            subschema = dereferenced_sweep_config_jsonschema["definitions"][schema_name]

            try:
                jsonschema.Draft7Validator(
                    subschema, format_checker=format_checker
                ).validate(config)
            except jsonschema.ValidationError:
                continue
            else:
                filler = DefaultFiller(subschema, format_checker=format_checker)

                # this sets the defaults, modifying config inplace
                filler.validate(config)

                valid = True
                self.type = schema_name
                self.config = deepcopy(config)

        if not valid:
            raise jsonschema.ValidationError("invalid hyperparameter configuration")

        if self.config is None or self.type is None:
            raise ValueError(
                "list of allowed schemas has length zero; please provide some valid schemas"
            )

        self.value = None

    def value_to_int(self, value: Any) -> int:
        if self.type != HyperParameter.CATEGORICAL:
            raise ValueError("Can only call value_to_int on categorical variable")

        for ii, test_value in enumerate(self.config["values"]):
            if value == test_value:
                return ii

        raise ValueError("Couldn't find {}".format(value))

    def cdf(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Cumulative distribution function
        Inputs: sample from selected distribution at the xth percentile.
        Ouputs: float in the range [0, 1]
        """
        if self.type == HyperParameter.CONSTANT:
            return 0.0
        elif self.type == HyperParameter.CATEGORICAL:
            # NOTE: Indices expected for categorical parameters, not values.
            return stats.randint.cdf(x, 0, len(self.config["values"]))
        elif self.type == HyperParameter.INT_UNIFORM:
            return stats.randint.cdf(x, self.config["min"], self.config["max"] + 1)
        elif (
            self.type == HyperParameter.UNIFORM or self.type == HyperParameter.Q_UNIFORM
        ):
            return stats.uniform.cdf(
                x, self.config["min"], self.config["max"] - self.config["min"]
            )
        elif (
            self.type == HyperParameter.LOG_UNIFORM
            or self.type == HyperParameter.Q_LOG_UNIFORM
        ):
            return stats.loguniform(self.config["min"], self.config["max"]).cdf(x)
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

    def ppf(self, x: npt.ArrayLike) -> Any:
        """
        Percent point function or inverse cdf
        Inputs: x: float in range [0, 1]
        Ouputs: sample from selected distribution at the xth percentile.
        """
        if x < 0.0 or x > 1.0:
            raise ValueError("Can't call ppf on value outside of [0,1]")
        if self.type == HyperParameter.CONSTANT:
            return self.config["value"]
        elif self.type == HyperParameter.CATEGORICAL:
            return self.config["values"][
                int(stats.randint.ppf(x, 0, len(self.config["values"])))
            ]
        elif self.type == HyperParameter.INT_UNIFORM:
            return int(stats.randint.ppf(x, self.config["min"], self.config["max"] + 1))
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
                return int(ret_val)
            else:
                return ret_val
        elif self.type == HyperParameter.LOG_UNIFORM:
            return stats.loguniform(self.config["min"], self.config["max"]).ppf(x)
        elif self.type == HyperParameter.Q_LOG_UNIFORM:
            r = stats.loguniform(self.config["min"], self.config["max"]).ppf(x)
            ret_val = np.round(r / self.config["q"]) * self.config["q"]
            if isinstance(self.config["q"], int):
                return int(ret_val)
            else:
                return ret_val
        elif self.type == HyperParameter.NORMAL:
            return stats.norm.ppf(x, loc=self.config["mu"], scale=self.config["sigma"])
        elif self.type == HyperParameter.Q_NORMAL:
            r = stats.norm.ppf(x, loc=self.config["mu"], scale=self.config["sigma"])
            ret_val = np.round(r / self.config["q"]) * self.config["q"]
            if isinstance(self.config["q"], int):
                return int(ret_val)
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
                return int(ret_val)
            else:
                return ret_val

        elif self.type == HyperParameter.BETA:
            return stats.beta.ppf(x, a=self.config["a"], b=self.config["b"])
        elif self.type == HyperParameter.Q_BETA:
            r = stats.beta.ppf(x, a=self.config["a"], b=self.config["b"])
            ret_val = np.round(r / self.config["q"]) * self.config["q"]
            if isinstance(self.config["q"], int):
                return int(ret_val)
            else:
                return ret_val
        else:
            raise ValueError("Unsupported hyperparameter distribution type")

    def sample(self) -> Any:
        return self.ppf(random.uniform(0.0, 1.0))

    def to_config(self) -> Tuple[str, Dict]:
        config = dict(value=self.value)
        # Remove values list if we have picked a value for this parameter
        # self.config.pop("values", None)
        return self.name, config


class HyperParameterSet(list):
    def __init__(self, items: List[HyperParameter]):
        for item in items:
            if not isinstance(item, HyperParameter):
                raise TypeError(
                    f"each item used to initialize HyperParameterSet must be a HyperParameter, got {item}"
                )

        super().__init__(items)
        self.searchable_params = [
            param for param in self if param.type != HyperParameter.CONSTANT
        ]

        self.param_names_to_index = {}
        self.param_names_to_param = {}

        for ii, param in enumerate(self.searchable_params):
            self.param_names_to_index[param.name] = ii
            self.param_names_to_param[param.name] = param

    @classmethod
    def from_config(cls, config: Dict):
        hpd = cls(
            [
                HyperParameter(param_name, param_config)
                for param_name, param_config in sorted(config.items())
            ]
        )
        return hpd

    def to_config(self) -> dict:
        return dict([param.to_config() for param in self])

    def denormalize_vector(self, X: npt.ArrayLike) -> List[List[Any]]:
        """Converts a list of vectors [0,1] to values in the original space."""
        v = np.zeros(X.shape).tolist()

        for ii, param in enumerate(self.searchable_params):
            for jj, x in enumerate(X[:, ii]):
                v[jj][ii] = param.ppf(x)
        return v

    def convert_runs_to_normalized_vector(self, runs: List[SweepRun]) -> npt.ArrayLike:
        runs_params = [run.config for run in runs]
        X = np.zeros([len(self.searchable_params), len(runs)])

        for key, bayes_opt_index in self.param_names_to_index.items():
            param = self.param_names_to_param[key]
            row = np.array(
                [
                    (
                        param.value_to_int(config[key]["value"])
                        if param.type == HyperParameter.CATEGORICAL
                        else config[key]["value"]
                    )
                    if key in config
                    # filter out incorrectly specified runs
                    else np.nan
                    for config in runs_params
                ]
            )

            X_row = param.cdf(row)
            # only use values where input wasn't nan
            non_nan = ~np.isnan(row)
            X[bayes_opt_index, non_nan] = X_row[non_nan]

        return np.transpose(X)
