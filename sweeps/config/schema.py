import json
import jsonref
import jsonschema
from jsonschema import Draft7Validator, validators

from pathlib import Path

sweep_config_jsonschema_fname = Path(__file__).parent / "schema.json"
with open(sweep_config_jsonschema_fname, "r") as f:
    sweep_config_jsonschema = json.load(f)


validator = jsonschema.Draft7Validator(schema=sweep_config_jsonschema)
dereferenced_sweep_config_jsonschema = jsonref.JsonRef(sweep_config_jsonschema)


def extend_with_default(validator_class):
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


validator_factory = extend_with_default(Draft7Validator)
