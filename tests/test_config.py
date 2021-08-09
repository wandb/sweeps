import pytest
import jsonschema
from .. import config


def test_invalid_sweep_config_nonuniform_array_elements_categorical():
    invalid_config = {
        "method": "grid",
        "parameters": {
            "v1": {"values": [None, 2, 3, "a", (2, 3), 3]},
        },
    }

    with pytest.raises(jsonschema.ValidationError):
        _ = config.SweepConfig(invalid_config)

    invalid_config = {
        "method": "grid",
        "parameters": {
            "v1": {"values": [None, 2, 3, "a", (2, 3)]},
        },
    }

    # doesn't raise
    _ = config.SweepConfig(invalid_config)


def test_min_max_validation():
    invalid_config = {
        "method": "random",
        "parameters": {
            "v1": {"max": 3, "min": 5},
            "v2": {"min": 5, "max": 6},
        },
    }

    with pytest.raises(jsonschema.ValidationError):
        _ = config.SweepConfig(invalid_config)


def test_negative_sigma_validation():
    invalid_config = {
        "method": "random",
        "parameters": {
            "v1": {"mu": 0.1, "sigma": -0.1, "distribution": "normal"},
        },
    }
    with pytest.raises(jsonschema.ValidationError):
        _ = config.SweepConfig(invalid_config)


def test_missing_parameters_section():
    invalid_config = {
        "method": "random",
    }

    warnings = config.schema_violations_from_proposed_config(invalid_config)
    assert len(warnings) == 1
