import pytest
import jsonschema
from sweeps import config


# we may change this to warning if we want to not force sweep config
# violation to be so rigid
validation_violation_context = pytest.raises(jsonschema.ValidationError)


def test_invalid_sweep_config_nonuniform_array_elements_categorical():
    invalid_config = {
        "method": "grid",
        "parameters": {
            "v1": {"values": [None, 2, 3]},
        },
    }

    with validation_violation_context:
        _ = config.SweepConfig(invalid_config)


def test_min_max_validation():
    invalid_config = {
        "method": "random",
        "parameters": {
            "v1": {"max": 3, "min": 5},
            "v2": {"min": 5, "max": 6},
        },
    }

    with validation_violation_context:
        _ = config.SweepConfig(invalid_config)
