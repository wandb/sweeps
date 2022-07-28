import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple

import jsonref
import jsonschema
import numpy as np
from jsonschema import Draft7Validator, validators


class ParamValidationError(Exception):
    pass


sweep_config_jsonschema_fname = Path(__file__).parent / "schema.json"
with open(sweep_config_jsonschema_fname, "r") as f:
    sweep_config_jsonschema = json.load(f)


dereferenced_sweep_config_jsonschema = jsonref.JsonRef.replace_refs(
    sweep_config_jsonschema
)

format_checker = jsonschema.FormatChecker()


@format_checker.checks("float")
def float_checker(value):
    return isinstance(value, float)


@format_checker.checks("integer")
def int_checker(value):
    return isinstance(value, int)


validator = Draft7Validator(
    schema=sweep_config_jsonschema, format_checker=format_checker
)


def extend_with_default(validator_class):
    # https://python-jsonschema.readthedocs.io/en/stable/faq/#why-doesn-t-my-schema-s-default-property-set-the-default-on-my-instance
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):

        errored = False
        for error in validate_properties(
            validator,
            properties,
            instance,
            schema,
        ):
            errored = True
            yield error

        if not errored:
            for property, subschema in properties.items():
                if "default" in subschema:
                    instance.setdefault(property, subschema["default"])

    return validators.extend(
        validator_class,
        {"properties": set_defaults},
    )


DefaultFiller = extend_with_default(Draft7Validator)
default_filler = DefaultFiller(
    schema=sweep_config_jsonschema, format_checker=format_checker
)


def fill_parameter(parameter_name: str, config: Dict) -> Optional[Tuple[str, Dict]]:
    # names of the parameter definitions that are allowed
    allowed_schemas = [
        d["$ref"].split("/")[-1]
        for d in sweep_config_jsonschema["definitions"]["parameter"]["anyOf"]
    ]

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
            if schema_name == "param_dict":
                raise ParamValidationError("Parameter dict cannot be filled.")
            validate_min_max(parameter_name, config)
            filler = DefaultFiller(subschema, format_checker=format_checker)
            # this sets the defaults, modifying config inplace
            config = deepcopy(config)
            filler.validate(config)
            return schema_name, config

    return None


def validate_min_max(parameter_name: str, parameter_config: Dict) -> None:
    if "min" in parameter_config and "max" in parameter_config:
        # this comparison is type safe because the jsonschema enforces type uniformity
        if parameter_config["min"] >= parameter_config["max"]:
            raise ValueError(
                f'{parameter_name}: min {parameter_config["min"]} is not '
                f'less than max {parameter_config["max"]}'
            )


def validate_categorical_prob(parameter_name: str, parameter_config: Dict) -> None:
    if "values" in parameter_config and "probabilities" in parameter_config:
        if len(parameter_config["values"]) != len(parameter_config["probabilities"]):
            raise ValueError(
                f"Parameter {parameter_name}: values {parameter_config['values']}"
                f" and probabilities {parameter_config['probabilities']} are not "
                f"the same length"
            )
        if not np.isclose(sum(parameter_config["probabilities"]), 1.0):
            raise ValueError(
                f"Parameter {parameter_name}: Probabilities "
                f"{parameter_config['probabilities']} do not sum to 1"
            )


def check_for_deprecated_distributions(
    parameter_name: str, parameter_config: Dict
) -> None:
    from ..params import HyperParameter, PARAM_DEPRECATION_MAP

    try:
        param = HyperParameter(
            parameter_name, parameter_config
        )  # will raise if parameter config is malformed
    except ParamValidationError:
        pass
    else:
        # check if type is deprecated
        if param.type in PARAM_DEPRECATION_MAP:
            raise ValueError(f"{parameter_name} {PARAM_DEPRECATION_MAP[param.type]}")


def fill_validate_metric(d: Dict) -> Dict:
    d = deepcopy(d)

    if "metric" in d:
        if not isinstance(d["metric"], dict):
            raise ValueError(
                f"invalid type for metric: expected dict, got {type(d['metric'])}"
            )
        if "goal" in d["metric"]:
            if (
                d["metric"]["goal"]
                not in dereferenced_sweep_config_jsonschema["properties"]["metric"][
                    "properties"
                ]["goal"]["enum"]
            ):
                # let it be filled in by the schema default
                del d["metric"]["goal"]

        if "impute" in d["metric"]:
            if (
                d["metric"]["impute"]
                not in dereferenced_sweep_config_jsonschema["properties"]["metric"][
                    "properties"
                ]["impute"]["enum"]
            ):
                # let it be filled in by the schema default
                del d["metric"]["impute"]

        filler = DefaultFiller(
            schema=dereferenced_sweep_config_jsonschema["properties"]["metric"],
            format_checker=format_checker,
        )
        filler.validate(d["metric"])
    return d


def fill_validate_early_terminate(d: Dict) -> Dict:
    d = deepcopy(d)
    if d["early_terminate"]["type"] == "hyperband":
        filler = DefaultFiller(
            schema=dereferenced_sweep_config_jsonschema["definitions"][
                "hyperband_stopping"
            ],
            format_checker=format_checker,
        )
        filler.validate(d["early_terminate"])
    return d


def fill_validate_schema(d: Dict) -> Dict:
    from . import schema_violations_from_proposed_config

    # check that the schema is valid
    violations = schema_violations_from_proposed_config(d)
    if len(violations) != 0:
        raise jsonschema.ValidationError("\n".join(violations))

    validated = deepcopy(d)

    # update the parameters
    filled = {}
    for k, v in validated["parameters"].items():
        try:
            result = fill_parameter(k, v)
        except ParamValidationError:
            continue
        else:
            if result is None:
                raise jsonschema.ValidationError(f"Parameter {k} is malformed")
            _, config = result
            filled[k] = config
    validated["parameters"] = filled

    if "early_terminate" in validated:
        validated = fill_validate_early_terminate(validated)

    return validated
