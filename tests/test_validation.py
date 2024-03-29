import pytest
from jsonschema import ValidationError
from sweeps import SweepRun, next_run, stop_runs
from sweeps.bayes_search import bayes_search_next_runs
from sweeps.config import (
    SweepConfig,
    fill_parameter,
    schema_violations_from_proposed_config,
)
from sweeps.grid_search import grid_search_next_runs
from sweeps.hyperband_stopping import hyperband_stop_runs
from sweeps.random_search import random_search_next_runs


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
            _ = bayes_search_next_runs([], invalid_schema, validate=True)
        elif search_type == "grid":
            _ = grid_search_next_runs([], invalid_schema, validate=True)
        elif search_type == "random":
            _ = random_search_next_runs(invalid_schema, validate=True)

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
    assert len(violations) == 2


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


@pytest.mark.parametrize("search_type", ["bayes", "grid", "random"])
def test_param_dict(search_type):
    # param dict inside param dict
    sweep_config = {
        "method": search_type,
        "metric": {"name": "loss", "goal": "minimize"},
        "parameters": {
            "a": {
                "parameters": {
                    "b": {"values": [1, 2]},
                    "c": {
                        "parameters": {"d": {"value": 1}},
                    },
                },
            },
        },
    }
    run_config_1 = {"a": {"value": {"b": 1, "c": {"d": 1}}}}
    run_config_2 = {"a": {"value": {"b": 2, "c": {"d": 1}}}}
    run = next_run(sweep_config, [SweepRun(config=run_config_1)])
    assert run.config in [run_config_1, run_config_2]

    # naming conflict is ok as long as different nest levels
    sweep_config = {
        "method": search_type,
        "metric": {"name": "loss", "goal": "minimize"},
        "parameters": {
            "a": {
                "parameters": {
                    "b": {"value": 1},
                    "c": {
                        "parameters": {"d": {"value": 2}},
                    },
                },
            },
            "b": {"values": [2, 3]},
        },
    }
    run_config_1 = {
        "a": {"value": {"b": 1, "c": {"d": 2}}},
        "b": {"value": 2},
    }
    run_config_2 = {
        "a": {"value": {"b": 1, "c": {"d": 2}}},
        "b": {"value": 3},
    }
    run = next_run(sweep_config, [SweepRun(config=run_config_1)])
    assert run.config in [run_config_1, run_config_2]


def test_invalid_minmax_with_no_sweepconfig_validation():
    config = {"method": "random", "parameters": {"a": {"max": 0, "min": 1}}}

    with pytest.raises(ValueError):
        fill_parameter("a", config["parameters"]["a"])


@pytest.mark.parametrize(
    "parameter_type", ["log_uniform", "q_log_uniform", "inv_log_uniform"]
)
def test_that_old_distributions_warn(parameter_type):
    schema = {
        "method": "random",
        "parameters": {"a": {"min": 1, "max": 2, "distribution": parameter_type}},
    }

    violations = schema_violations_from_proposed_config(schema)
    assert len(violations) > 0


@pytest.mark.parametrize(
    "parameter_type",
    ["log_uniform_values", "q_log_uniform_values", "inv_log_uniform_values"],
)
def test_that_minmax_validation_fails_on_loguniform_values_types(parameter_type):
    schema = {
        "method": "random",
        "parameters": {"a": {"min": 2, "max": 1, "distribution": parameter_type}},
    }

    with pytest.raises(ValueError):
        fill_parameter("a", schema["parameters"]["a"])


@pytest.mark.parametrize(
    "run_cap",
    [-1, 0, 1, 300, -0.3, 5.1, "1"],
)
def test_that_run_cap_validation_works_for_min_value(run_cap):
    schema = {
        "method": "random",
        "parameters": {"a": {"values": [1, 2, 3, 4]}},
        "run_cap": run_cap,
    }

    violations = schema_violations_from_proposed_config(schema)

    if type(run_cap) != int:
        # -0.3 has two errors
        assert len(violations) >= 1
    elif run_cap <= 0:
        assert len(violations) == 1
    else:
        assert len(violations) == 0
