import pytest

from sweeps.config import utils


def test_nested_config_helpers():

    invalid_nested_config = {"foo": {1: 1}}
    with pytest.raises(ValueError):
        _ = utils.unnest_config(invalid_nested_config)

    invalid_nested_config = {1: {"foo": 1}}
    with pytest.raises(ValueError):
        _ = utils.unnest_config(invalid_nested_config)

    invalid_unnested_config = {1: "foo"}
    with pytest.raises(ValueError):
        _ = utils.nest_config(invalid_unnested_config)

    invalid_unnested_config = {"foo": 1, "foo.bar": {"baz": 2}}
    with pytest.raises(ValueError):
        _ = utils.nest_config(invalid_unnested_config)

    valid_nested_config = {"foo": {"bar": 1}}
    unnested_config = utils.unnest_config(valid_nested_config)
    renested_config = utils.nest_config(unnested_config)
    assert valid_nested_config == renested_config

    valid_nested_config = {"foo": {"bar": {"baz": 1, "boz": 2}}}
    unnested_config = utils.unnest_config(valid_nested_config)
    renested_config = utils.nest_config(unnested_config)
    assert valid_nested_config == renested_config
