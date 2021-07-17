import pytest
from jsonschema import ValidationError
from .. import next_run, stop_runs
from ..config import SweepConfig, schema_violations_from_proposed_config
from ..bayes_search import bayes_search_next_run
from ..grid_search import grid_search_next_run
from ..random_search import random_search_next_run
from ..hyperband_stopping import hyperband_stop_runs


@pytest.mark.parametrize("search_type", ["bayes", "grid", "random"])
def test_validation_disable(search_type):
    invalid_schema = {
        "metric": {"name": "loss", "goal": "minimise"},
        "method": search_type,
        "parameters": {
            "v1": {"values": ["a", "b", "c"]},
        },
    }

    with pytest.raises(ValidationError):
        _ = next_run(invalid_schema, [], validate=True)

    with pytest.raises(ValidationError):
        _ = stop_runs(invalid_schema, [], validate=True)
        _ = hyperband_stop_runs([], invalid_schema, validate=True)

    with pytest.raises(ValidationError):
        if search_type == "bayes":
            _ = bayes_search_next_run([], invalid_schema, validate=True)
        elif search_type == "grid":
            _ = grid_search_next_run([], invalid_schema, validate=True)
        elif search_type == "random":
            _ = random_search_next_run(invalid_schema, validate=True)

    # check that no error is raised
    result = next_run(invalid_schema, [], validate=False)
    assert result is not None


def test_validation_not_enough_params():
    schema = {"method": "random", "parameters": {}}

    with pytest.raises(ValidationError):
        SweepConfig(schema)


def test_minmax_type_inference():
    schema = {
        "method": "random",
        "parameters": {"a": {"min": 0, "max": 1, "distribution": "uniform"}},
    }

    violations = schema_violations_from_proposed_config(schema)
    assert len(violations) == 0

    schema = {
        "method": "random",
        "parameters": {"a": {"min": 0.0, "max": 1.0, "distribution": "int_uniform"}},
    }

    violations = schema_violations_from_proposed_config(schema)
    assert len(violations) == 1


@pytest.mark.parametrize("controller_type", ["cloud", "local", "invalid"])
def test_controller(controller_type):
    schema = {
        "controller": {"type": controller_type},
        "method": "random",
        "parameters": {"a": {"values": [1, 2, 3, 4]}},
    }

    if controller_type in ["cloud", "local"]:
        assert SweepConfig(schema)["controller"]["type"] == controller_type
    else:
        with pytest.raises(ValidationError):
            SweepConfig(schema)


def test_invalid_config():
    config = "this is a totally invalid config"
    with pytest.raises(ValueError):
        schema_violations_from_proposed_config(config)
