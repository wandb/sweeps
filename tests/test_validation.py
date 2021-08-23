import pytest
from jsonschema import ValidationError
from .. import next_run, stop_runs, SweepRun
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

    # test that this doesnt raise a keyerror https://sentry.io/organizations/weights-biases/issues/2461042074/?project=5812400&query=is%3Aresolved&statsPeriod=14d
    result = schema_violations_from_proposed_config(schema)
    assert len(result) == 1

    # test that this is rejected at the next_run stage
    with pytest.raises(ValueError):
        next_run(schema, [])

    del schema["parameters"]
    # test that this is rejected at the next_run stage
    with pytest.raises(ValueError):
        next_run(schema, [])


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


def test_sweepconfig_no_method_baseline_validation():
    schema = {
        "parameters": {"a": {"values": [1, 2, 3, 4]}},
    }

    with pytest.raises(ValueError):
        next_run(schema, [], validate=False)


def test_invalid_early_stopping():
    invalid_schema = {
        "method": "bayes",
        "parameters": {
            "v1": {"values": ["a", "b", "c"]},
        },
    }

    with pytest.raises(ValueError):
        stop_runs(invalid_schema, [], validate=False)

    invalid_schema["metric"] = {"name": "loss", "goal": "minimise"}

    with pytest.raises(ValueError):
        stop_runs(invalid_schema, [], validate=False)

    invalid_schema["early_terminate"] = dict()
    invalid_schema["early_terminate"]["type"] = "invalid type"

    with pytest.raises(ValueError):
        stop_runs(invalid_schema, [], validate=False)

    invalid_schema["early_terminate"]["type"] = "hyperband"
    invalid_schema["early_terminate"]["extra_key"] = 1234
    invalid_schema["early_terminate"]["min_iter"] = 100

    to_stop = stop_runs(invalid_schema, [], validate=False)
    assert len(to_stop) == 0


def test_invalid_run_parameter():
    config = {
        "method": "bayes",
        "metric": {"name": "loss", "goal": "minimize"},
        "parameters": {
            "v1": {"values": ["a", "b", "c"]},
        },
    }

    runs = [SweepRun(config={"v1": {"value": "d"}}, summary_metrics={"loss": 5.0})]

    with pytest.raises(ValueError):
        next_run(config, runs, validate=False)
