"""Hyperparameter search parameters."""

import random

import numpy as np
import scipy.stats as stats

import jsonschema
from .config.schema import (
    sweep_config_jsonschema,
    dereferenced_sweep_config_jsonschema,
    validator_factory,
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

    def __init__(self, name: str, config: dict):

        self.name = name

        # names of the parameter definitions that are allowed
        allowed_schemas = [
            d["$ref"]
            for d in sweep_config_jsonschema["definitions"]["parameter"]["anyOf"]
        ]

        valid = False
        inferred_schema = None
        for schema in allowed_schemas:
            # create a jsonschema object to validate against the subschema
            subschema = dereferenced_sweep_config_jsonschema["definitions"][
                schema
            ].copy()
            subschema["$schema"] = "http://json-schema.org/draft-07/schema#"
            try:
                jsonschema.validate(config, subschema)
            except jsonschema.ValidationError:
                continue
            else:
                valid = True
                self.type = schema
                validator = validator_factory(subschema)

                # this sets the defaults
                validator.validate(config)
                self.config = config

        if not valid:
            raise jsonschema.ValidationError("invalid hyperparameter configuration")

        if inferred_schema is None:
            raise ValueError(
                "list of allowed schemas has length zero; please provide some valid schemas"
            )

    def value_to_int(self, value):
        if self.type != HyperParameter.CATEGORICAL:
            raise ValueError("Can only call value_to_int on categorical variable")

        for ii, test_value in enumerate(self.values):
            if value == test_value:
                return ii

        raise ValueError("Couldn't find {}".format(value))

    def cdf(self, x):
        """
        Cumulative distribution function
        Inputs: sample from selected distribution at the xth percentile.
        Ouputs: float in the range [0, 1]
        """
        if self.type == HyperParameter.CONSTANT:
            return 0.0
        elif self.type == HyperParameter.CATEGORICAL:
            # NOTE: Indices expected for categorical parameters, not values.
            return stats.randint.cdf(x, 0, len(self.values))
        elif self.type == HyperParameter.INT_UNIFORM:
            return stats.randint.cdf(x, self.min, self.max + 1)
        elif (
            self.type == HyperParameter.UNIFORM or self.type == HyperParameter.Q_UNIFORM
        ):
            return stats.uniform.cdf(x, self.min, self.max - self.min)
        elif (
            self.type == HyperParameter.LOG_UNIFORM
            or self.type == HyperParameter.Q_LOG_UNIFORM
        ):
            return stats.loguniform(self.min, self.max).cdf(x)
        elif self.type == HyperParameter.NORMAL or self.type == HyperParameter.Q_NORMAL:
            return stats.norm.cdf(x, loc=self.mu, scale=self.sigma)
        elif (
            self.type == HyperParameter.LOG_NORMAL
            or self.type == HyperParameter.Q_LOG_NORMAL
        ):
            return stats.lognorm.cdf(x, s=self.sigma, scale=np.exp(self.mu))
        else:
            raise ValueError("Unsupported hyperparameter distribution type")

    def ppf(self, x):
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
            return self.config["values"][int(stats.randint.ppf(x, 0, len(self.values)))]
        elif self.type == HyperParameter.INT_UNIFORM:
            return int(stats.randint.ppf(x, self.config["min"], self.config["max"] + 1))
        elif self.type == HyperParameter.UNIFORM:
            return stats.uniform.ppf(x, self.min, self.max - self.min)
        elif self.type == HyperParameter.Q_UNIFORM:
            r = stats.uniform.ppf(x, self.min, self.max - self.min)
            ret_val = np.round(r / self.q) * self.q
            if type(self.q) == int:
                return int(ret_val)
            else:
                return ret_val
        elif self.type == HyperParameter.LOG_UNIFORM:
            return stats.loguniform(self.min, self.max).ppf(x)
        elif self.type == HyperParameter.Q_LOG_UNIFORM:
            r = stats.loguniform(self.min, self.max).ppf(x)
            ret_val = np.round(r / self.q) * self.q
            if type(self.q) == int:
                return int(ret_val)
            else:
                return ret_val
        elif self.type == HyperParameter.NORMAL:
            return stats.norm.ppf(x, loc=self.mu, scale=self.sigma)
        elif self.type == HyperParameter.Q_NORMAL:
            r = stats.norm.ppf(x, loc=self.mu, scale=self.sigma)
            ret_val = np.round(r / self.q) * self.q
            if type(self.q) == int:
                return int(ret_val)
            else:
                return ret_val
        elif self.type == HyperParameter.LOG_NORMAL:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html
            return stats.lognorm.ppf(x, s=self.sigma, scale=np.exp(self.mu))
        elif self.type == HyperParameter.Q_LOG_NORMAL:
            r = stats.lognorm.ppf(x, s=self.sigma, scale=np.exp(self.mu))
            ret_val = np.round(r / self.q) * self.q

            if type(self.q) == int:
                return int(ret_val)
            else:
                return ret_val
        else:
            raise ValueError("Unsupported hyperparameter distribution type")

    def sample(self) -> float:
        return self.ppf(random.uniform(0.0, 1.0))

    def to_config(self):
        config = dict(value=self.value)
        # Remove values list if we have picked a value for this parameter
        self.config.pop("values", None)
        return self.name, config


class HyperParameterSet(list):
    @classmethod
    def from_config(cls, config):
        hpd = cls(
            [
                HyperParameter(param_name, param_config)
                for param_name, param_config in sorted(config.items())
            ]
        )
        return hpd

    def to_config(self):
        return dict([param.to_config() for param in self])

    def index_searchable_params(self):
        self.searchable_params = [
            param for param in self if param.type != HyperParameter.CONSTANT
        ]

        self.param_names_to_index = {}
        self.param_names_to_param = {}

        for ii, param in enumerate(self.searchable_params):
            self.param_names_to_index[param.name] = ii
            self.param_names_to_param[param.name] = param

    def numeric_bounds(self):
        """Gets a set of numeric minimums and maximums for doing ml predictions
        on the hyperparameters."""
        self.searchable_params = [
            param for param in self if param.type != HyperParameter.CONSTANT
        ]

        X_bounds = [[0.0, 0.0]] * len(self.searchable_params)

        self.param_names_to_index = {}
        self.param_names_to_param = {}

        for ii, param in enumerate(self.searchable_params):
            self.param_names_to_index[param.name] = ii
            self.param_names_to_param[param.name] = param
            if param.type == HyperParameter.CATEGORICAL:
                X_bounds[ii] = [0, len(param.values)]
            elif param.type == HyperParameter.INT_UNIFORM:
                X_bounds[ii] = [param.min, param.max]
            elif param.type == HyperParameter.UNIFORM:
                X_bounds[ii] = [param.min, param.max]
            else:
                raise ValueError("Unsupported param type")

        return X_bounds

    def convert_run_to_vector(self, run):
        """Converts run parameters to vectors.

        Should be able to remove.
        """

        run_params = run.config or {}
        X = np.zeros([len(self.searchable_params)])

        # we ignore keys we haven't seen in our spec
        # we don't handle the case where a key is missing from run config
        for key, config_value in run_params.items():
            if key in self.param_names_to_index:
                param = self.param_names_to_param[key]
                bayes_opt_index = self.param_names_to_index[key]
                if param.type == HyperParameter.CATEGORICAL:
                    bayes_opt_value = param.value_to_int(config_value["value"])
                else:
                    bayes_opt_value = config_value["value"]

                X[bayes_opt_index] = bayes_opt_value
        return X

    def denormalize_vector(self, X):
        """Converts a list of vectors [0,1] to values in the original space."""
        v = np.zeros(X.shape).tolist()

        for ii, param in enumerate(self.searchable_params):
            for jj, x in enumerate(X[:, ii]):
                v[jj][ii] = param.ppf(x)
        return v

    def convert_run_to_normalized_vector(self, run):
        """Converts run parameters to vectors with all values compressed to [0,
        1]"""
        run_params = run.config or {}
        X = np.zeros([len(self.searchable_params)])

        # we ignore keys we haven't seen in our spec
        # we don't handle the case where a key is missing from run config
        for key, config_value in run_params.items():
            if key in self.param_names_to_index:
                param = self.param_names_to_param[key]
                bayes_opt_index = self.param_names_to_index[key]
                bayes_opt_value = param.cdf(param.value_to_int(config_value["value"]))

                X[bayes_opt_index] = bayes_opt_value
        return X

    def convert_runs_to_normalized_vector(self, runs):
        runs_params = [run.config or {} for run in runs]
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
                    else float("nan")
                    for config in runs_params
                ]
            )
            X_row = param.cdf(row)

            # only use values where input wasn't nan
            non_nan = row == row
            X[bayes_opt_index, non_nan] = X_row[non_nan]

        return np.transpose(X)
