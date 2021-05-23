"""Base SweepConfig classes."""

import warnings
import json
import yaml
from six.moves import UserDict
from pathlib import Path

from typing import Union

import jsonschema

sweep_config_jsonschema_fname = Path(__file__).parent / "schema.json"
with open(sweep_config_jsonschema_fname, "r") as f:
    sweep_config_jsonschema = json.load(f)


validator = jsonschema.Draft7Validator(schema=sweep_config_jsonschema)


class SweepConfig(UserDict):
    def __init__(self, d: Union[dict, UserDict]):
        super(SweepConfig, self).__init__(d)

        if not isinstance(d, SweepConfig):
            # ensure the data conform to the schema
            schema_violation_messages = []
            for error in validator.iter_errors(dict(self)):
                schema_violation_messages.append(f"{error}")

            # validate min/max - this cannot be done with jsonschema
            # because it does not support comparing values within
            # a json document. so we do it manually here:
            for parameter_name, parameter_dict in self["parameters"].items():
                if "min" in parameter_dict and "max" in parameter_dict:
                    # this comparison is type safe because the jsonschema enforces type uniformity
                    if parameter_dict["min"] >= parameter_dict["max"]:
                        schema_violation_messages.append(
                            f'{parameter_name}: min {parameter_dict["min"]} is not '
                            f'less than max {parameter_dict["max"]}'
                        )

            if len(schema_violation_messages) > 0:
                err_msg = "\n".join(schema_violation_messages)
                raise jsonschema.ValidationError(err_msg)

    def __str__(self):
        return yaml.safe_dump(self.data)

    def save(self, filename):
        with open(filename, "w") as outfile:
            yaml.safe_dump(self.data, outfile, default_flow_style=False)

    def set_local(self):
        self.data.update(dict(controller=dict(type="local")))
        return self

    def set_name(self, name):
        self.data.update(dict(name=name))
        return self

    def set_settings(self, settings):
        self.data.update(dict(settings=settings))
        return self

    def set(self, **kwargs):
        local = kwargs.pop("local", None)
        name = kwargs.pop("name", None)
        settings = kwargs.pop("settings", None)
        if local:
            self.set_local()
        if name:
            self.set_name(name)
        if name:
            self.set_settings(settings)
        for k in kwargs.keys():
            warnings.warn(
                "Unsupported parameter passed to SweepConfig set(): {}".format(k)
            )
        return self


"""
class SweepConfigElement:
    _version_dict: dict = {}

    def __init__(self, module=None, version=None):
        self._module = module
        self._version = version
        self._version_dict.setdefault("wandb", wandb.__version__)
        if module and version:
            self._version_dict.setdefault(module, version)

    def _config(self, base, args, kwargs, root=False):
        kwargs = {k: v for k, v in kwargs.items() if v is not None and k != "self"}
        # remove kwargs if empty
        if kwargs.get("kwargs") == {}:
            del kwargs["kwargs"]
        # if only kwargs specified and only two keys "args" and "kargs"
        special = not args and set(kwargs.keys()) == {"args", "kwargs"}
        if args and kwargs or special:
            d = dict(args=args, kwargs=kwargs)
        elif args:
            d = args
        else:
            d = kwargs
        if base:
            if self._module:
                base = self._module + "." + base
            d = {base: d}
        if root:
            # HACK(jhr): move tune.run to tune
            d = d["tune.run"]
            d = dict(tune=d)
            for m, v in self._version_dict.items():
                d["tune"].setdefault("_wandb", {})
                d["tune"]["_wandb"].setdefault("versions", {})
                d["tune"]["_wandb"]["versions"][m] = v
            return SweepConfig(d)
        return d
"""
