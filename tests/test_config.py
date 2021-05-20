import pytest
import jsonschema
from sweeps import config


# we may change this to warning if we want to not force sweep config
# violation to be so rigid
validation_violation_context = pytest.raises(jsonschema.ValidationError)


def test_invalid_sweep_config_nonuniform_array_elements_categorical(
    sweep_config_dict_1param_invalid_none_grid_search,
):
    with validation_violation_context:
        _ = config.SweepConfig(sweep_config_dict_1param_invalid_none_grid_search)


def test_min_max_validation(sweep_config_invalid_violates_min_max):
    with validation_violation_context:
        _ = config.SweepConfig(sweep_config_invalid_violates_min_max)
