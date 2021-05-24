import json
import jsonref
from jsonschema import Draft7Validator, validators

from pathlib import Path

sweep_config_jsonschema_fname = Path(__file__).parent / "schema.json"
with open(sweep_config_jsonschema_fname, "r") as f:
    sweep_config_jsonschema = json.load(f)


dereferenced_sweep_config_jsonschema = jsonref.JsonRef.replace_refs(
    sweep_config_jsonschema
)


def extend_with_python_int_float_type_discrimination(validator_class):
    def is_python_int(checker, instance):
        return isinstance(instance, int)

    type_checker = validator_class.TYPE_CHECKER.redefine("integer", is_python_int)

    return validators.extend(validator_class, type_checker=type_checker)


Draft7ValidatorWithIntFloatDiscrimination = (
    extend_with_python_int_float_type_discrimination(Draft7Validator)
)
validator = Draft7ValidatorWithIntFloatDiscrimination(schema=sweep_config_jsonschema)


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


DefaultFiller = extend_with_default(Draft7ValidatorWithIntFloatDiscrimination)
