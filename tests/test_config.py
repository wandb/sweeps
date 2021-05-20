import pytest
import jsonschema
from sweeps import config


def test_invalid_sweep_config_nonuniform_array_elements_categorical(
    sweep_config_dict_1param_invalid_none_grid_search,
):
    with pytest.raises(jsonschema.ValidationError):
        _ = config.SweepConfig(sweep_config_dict_1param_invalid_none_grid_search)
