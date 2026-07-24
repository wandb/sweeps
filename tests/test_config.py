import jsonschema
import pytest
from sweeps import config, grid_search


def test_invalid_sweep_config_nonuniform_array_elements_categorical():

    valid_config = {
        "method": "grid",
        "parameters": {
            "v1": {"values": [None, 2, 3, "a", (2, 3)]},
        },
    }

    # doesn't raise
    _ = config.SweepConfig(valid_config)


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


def test_wrong_prob_length():
    invalid_config = {
        "method": "random",
        "parameters": {
            "v1": {"values": [1, 2, 3], "probabilities": [0.1, 0.2, 0.3, 0.4]}
        },
    }
    with pytest.raises(jsonschema.ValidationError):
        _ = config.SweepConfig(invalid_config)


def test_irregular_probs():
    invalid_config = {
        "method": "random",
        "parameters": {"v1": {"values": [1, 2, 3], "probabilities": [0.1, 0.2, 0.3]}},
    }
    with pytest.raises(jsonschema.ValidationError):
        _ = config.SweepConfig(invalid_config)


def test_categorical_prob_grid():
    invalid_config = {
        "method": "grid",
        "parameters": {"v1": {"values": [1, 2, 3], "probabilities": [0.2, 0.2, 0.6]}},
    }
    with pytest.raises(ValueError):
        sweep_config = config.SweepConfig(invalid_config)
        grid_search.grid_search_next_runs([], sweep_config)


def test_metrics_multi_objective_with_custom_method():
    valid_config = {
        "method": "custom",
        "metrics": [
            {"name": "loss", "goal": "minimize"},
            {"name": "accuracy", "goal": "maximize"},
        ],
        "parameters": {"v1": {"values": [1, 2, 3]}},
    }

    sweep_config = config.SweepConfig(valid_config)
    assert len(sweep_config["metrics"]) == 2


def test_metrics_requires_custom_method():
    invalid_config = {
        "method": "bayes",
        "metrics": [
            {"name": "loss", "goal": "minimize"},
            {"name": "accuracy", "goal": "maximize"},
        ],
        "parameters": {"v1": {"values": [1, 2, 3]}},
    }

    with pytest.raises(jsonschema.ValidationError):
        _ = config.SweepConfig(invalid_config)


def test_metric_and_metrics_mutually_exclusive():
    invalid_config = {
        "method": "custom",
        "metric": {"name": "loss"},
        "metrics": [
            {"name": "loss", "goal": "minimize"},
            {"name": "accuracy", "goal": "maximize"},
        ],
        "parameters": {"v1": {"values": [1, 2, 3]}},
    }

    with pytest.raises(jsonschema.ValidationError):
        _ = config.SweepConfig(invalid_config)


def test_metrics_requires_at_least_two():
    invalid_config = {
        "method": "custom",
        "metrics": [{"name": "loss", "goal": "minimize"}],
        "parameters": {"v1": {"values": [1, 2, 3]}},
    }

    with pytest.raises(jsonschema.ValidationError):
        _ = config.SweepConfig(invalid_config)


def test_metrics_filled_with_defaults():
    valid_config = {
        "method": "custom",
        "metrics": [{"name": "loss"}, {"name": "accuracy"}],
        "parameters": {"v1": {"values": [1, 2, 3]}},
    }

    filled = config.fill_validate_metrics(valid_config)
    assert filled["metrics"][0]["goal"] == "minimize"
    assert filled["metrics"][1]["goal"] == "minimize"


def test_fill_validate_metrics_rejects_non_list():
    invalid_config = {
        "method": "custom",
        "metrics": "not-a-list",
        "parameters": {"v1": {"values": [1, 2, 3]}},
    }

    with pytest.raises(ValueError, match="expected list"):
        config.fill_validate_metrics(invalid_config)


def test_fill_validate_metrics_rejects_non_dict_items():
    invalid_config = {
        "method": "custom",
        "metrics": [{"name": "loss"}, "not-a-dict"],
        "parameters": {"v1": {"values": [1, 2, 3]}},
    }

    with pytest.raises(ValueError, match="expected dict"):
        config.fill_validate_metrics(invalid_config)


def test_fill_validate_metrics_invalid_goal_replaced_with_default():
    valid_config = {
        "method": "custom",
        "metrics": [
            {"name": "loss", "goal": "not-a-real-goal"},
            {"name": "accuracy", "goal": "maximize"},
        ],
        "parameters": {"v1": {"values": [1, 2, 3]}},
    }

    filled = config.fill_validate_metrics(valid_config)
    assert filled["metrics"][0]["goal"] == "minimize"
    assert filled["metrics"][1]["goal"] == "maximize"


def test_fill_validate_metrics_invalid_impute_replaced_with_default():
    valid_config = {
        "method": "custom",
        "metrics": [
            {"name": "loss", "impute": "not-a-real-impute-strategy"},
            {"name": "accuracy", "impute": "best"},
        ],
        "parameters": {"v1": {"values": [1, 2, 3]}},
    }

    filled = config.fill_validate_metrics(valid_config)
    assert filled["metrics"][0]["impute"] == "worst"
    assert filled["metrics"][1]["impute"] == "best"


def test_scheduler_wandb_engine_valid():
    valid_config = {
        "method": "custom",
        "scheduler": {
            "engine": "wandb",
            "source": "scheduler.py",
            "optimizer": "build_study",
            "search_space": "search_space",
        },
        "parameters": {"v1": {"values": [1, 2, 3]}},
    }

    sweep_config = config.SweepConfig(valid_config)
    assert sweep_config["scheduler"]["engine"] == "wandb"


@pytest.mark.parametrize("engine", ["optuna", "ax"])
def test_scheduler_unavailable_engine_raises(engine):
    invalid_config = {
        "method": "custom",
        "scheduler": {
            "engine": engine,
            "source": "scheduler.py",
            "optimizer": "build_study",
            "search_space": "search_space",
        },
        "parameters": {"v1": {"values": [1, 2, 3]}},
    }

    with pytest.raises(jsonschema.ValidationError):
        _ = config.SweepConfig(invalid_config)


def test_scheduler_invalid_engine_rejected():
    invalid_config = {
        "method": "custom",
        "scheduler": {
            "engine": "not-a-real-engine",
            "source": "scheduler.py",
            "optimizer": "build_study",
            "search_space": "search_space",
        },
        "parameters": {"v1": {"values": [1, 2, 3]}},
    }

    with pytest.raises(jsonschema.ValidationError):
        _ = config.SweepConfig(invalid_config)


def test_scheduler_missing_required_field():
    invalid_config = {
        "method": "custom",
        "scheduler": {
            "source": "scheduler.py",
            "optimizer": "build_study",
            "search_space": "search_space",
        },
        "parameters": {"v1": {"values": [1, 2, 3]}},
    }

    with pytest.raises(jsonschema.ValidationError):
        _ = config.SweepConfig(invalid_config)


def test_scheduler_only_engine_required():
    valid_config = {
        "method": "custom",
        "scheduler": {
            "engine": "wandb",
        },
        "parameters": {"v1": {"values": [1, 2, 3]}},
    }

    sweep_config = config.SweepConfig(valid_config)
    assert sweep_config["scheduler"]["engine"] == "wandb"


@pytest.mark.parametrize("field", ["optimizer", "search_space"])
def test_scheduler_optimizer_or_search_space_requires_source(field):
    invalid_config = {
        "method": "custom",
        "scheduler": {
            "engine": "wandb",
            field: "build_study",
        },
        "parameters": {"v1": {"values": [1, 2, 3]}},
    }

    with pytest.raises(jsonschema.ValidationError):
        _ = config.SweepConfig(invalid_config)


def test_scheduler_optimizer_and_search_space_with_source_valid():
    valid_config = {
        "method": "custom",
        "scheduler": {
            "engine": "wandb",
            "source": "scheduler.py",
            "optimizer": "build_study",
            "search_space": "search_space",
        },
        "parameters": {"v1": {"values": [1, 2, 3]}},
    }

    sweep_config = config.SweepConfig(valid_config)
    assert sweep_config["scheduler"]["source"] == "scheduler.py"


def test_scheduler_requires_custom_method():
    invalid_config = {
        "method": "grid",
        "scheduler": {
            "engine": "wandb",
            "source": "scheduler.py",
            "optimizer": "build_study",
            "search_space": "search_space",
        },
        "parameters": {"v1": {"values": [1, 2, 3]}},
    }

    with pytest.raises(jsonschema.ValidationError, match="method: custom"):
        _ = config.SweepConfig(invalid_config)


def test_scheduler_and_early_terminate_mutually_exclusive():
    invalid_config = {
        "method": "custom",
        "scheduler": {
            "engine": "wandb",
            "source": "scheduler.py",
            "optimizer": "build_study",
            "search_space": "search_space",
        },
        "early_terminate": {"type": "hyperband", "min_iter": 3},
        "parameters": {"v1": {"values": [1, 2, 3]}},
    }

    with pytest.raises(jsonschema.ValidationError, match="early_terminate"):
        _ = config.SweepConfig(invalid_config)


def test_early_terminate_without_scheduler_still_works():
    valid_config = {
        "method": "bayes",
        "metric": {"name": "loss", "goal": "minimize"},
        "early_terminate": {"type": "hyperband", "min_iter": 3},
        "parameters": {"v1": {"values": [1, 2, 3]}},
    }

    sweep_config = config.SweepConfig(valid_config)
    assert sweep_config["early_terminate"]["type"] == "hyperband"


def test_metric_and_metrics_error_message_is_specific():
    invalid_config = {
        "method": "custom",
        "metric": {"name": "loss"},
        "metrics": [
            {"name": "loss", "goal": "minimize"},
            {"name": "accuracy", "goal": "maximize"},
        ],
        "parameters": {"v1": {"values": [1, 2, 3]}},
    }

    with pytest.raises(jsonschema.ValidationError, match="cannot both be set"):
        _ = config.SweepConfig(invalid_config)


def test_metrics_requires_custom_method_error_message_is_specific():
    invalid_config = {
        "method": "bayes",
        "metrics": [
            {"name": "loss", "goal": "minimize"},
            {"name": "accuracy", "goal": "maximize"},
        ],
        "parameters": {"v1": {"values": [1, 2, 3]}},
    }

    with pytest.raises(jsonschema.ValidationError, match="method: custom"):
        _ = config.SweepConfig(invalid_config)


def test_parameters_not_required_when_scheduler_defines_search_space():
    valid_config = {
        "method": "custom",
        "scheduler": {
            "engine": "wandb",
            "source": "scheduler.py",
            "optimizer": "build_study",
            "search_space": "search_space",
        },
    }

    sweep_config = config.SweepConfig(valid_config)
    assert "parameters" not in sweep_config


def test_parameters_still_allowed_alongside_scheduler_search_space():
    valid_config = {
        "method": "custom",
        "scheduler": {
            "engine": "wandb",
            "source": "scheduler.py",
            "optimizer": "build_study",
            "search_space": "search_space",
        },
        "parameters": {"v1": {"values": [1, 2, 3]}},
    }

    sweep_config = config.SweepConfig(valid_config)
    assert sweep_config["parameters"]["v1"]["values"] == [1, 2, 3]


def test_parameters_still_required_without_scheduler():
    invalid_config = {"method": "bayes"}

    with pytest.raises(jsonschema.ValidationError, match="`parameters` is required"):
        _ = config.SweepConfig(invalid_config)


def test_parameters_still_required_when_scheduler_has_no_search_space():
    invalid_config = {
        "method": "custom",
        "scheduler": {
            "engine": "wandb",
            "source": "scheduler.py",
            "optimizer": "build_study",
            "search_space": "",
        },
    }

    with pytest.raises(jsonschema.ValidationError, match="`parameters` is required"):
        _ = config.SweepConfig(invalid_config)
