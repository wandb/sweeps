import jsonschema
import pytest

from sweeps.config import SweepConfig, fill_validate_early_terminate
from sweeps.config.schema import fill_validate_metric
from sweeps.hyperband_stopping import hyperband_baseline_validate_and_fill
from sweeps.params import HyperParameter


def test_json_type_inference_int_uniform():
    config = {"min": 0, "max": 1}
    param = HyperParameter("int_unif_param", config)
    assert param.type == HyperParameter.INT_UNIFORM
    assert param.config["nested"] == False
    assert param.config["nest_delimiter"] == "."
    # len is 5 because distribution, nested, and nested_delimiter keys are inferred via default
    assert len(param.config) == 5


def test_json_type_inference_uniform():
    config = {"min": 0.0, "max": 1.0}
    param = HyperParameter("unif_param", config)
    assert param.type == HyperParameter.UNIFORM
    assert param.config["nested"] == False
    assert param.config["nest_delimiter"] == "."
    # len is 5 because distribution, nested, and nested_delimiter keys are inferred via default
    assert len(param.config) == 5


def test_json_type_inference_uniform_mixed():
    config = {"min": 0.0, "max": 1}
    param = HyperParameter("unif_param", config)
    assert param.type == HyperParameter.UNIFORM
    assert param.config["nested"] == False
    assert param.config["nest_delimiter"] == "."
    # len is 5 because distribution, nested, and nested_delimiter keys are inferred via default
    assert len(param.config) == 5

    config = {"min": 0, "max": 1.0}
    param = HyperParameter("unif_param", config)
    assert param.type == HyperParameter.UNIFORM
    assert param.config["nested"] == False
    assert param.config["nest_delimiter"] == "."
    # len is 5 because distribution, nested, and nested_delimiter keys are inferred via default
    assert len(param.config) == 5


def test_json_type_inference_and_imputation_normal():
    config = {"distribution": "normal"}
    param = HyperParameter("normal_param", config)
    assert param.config["mu"] == 0
    assert param.config["sigma"] == 1
    assert param.config["nested"] == False
    assert param.config["nest_delimiter"] == "."
    assert param.type == HyperParameter.NORMAL
    assert len(param.config) == 5


def test_json_type_inference_and_imputation_lognormal():
    config = {"distribution": "log_normal"}
    param = HyperParameter("log_normal_param", config)
    assert param.config["mu"] == 0
    assert param.config["sigma"] == 1
    assert param.config["nested"] == False
    assert param.config["nest_delimiter"] == "."
    assert param.type == HyperParameter.LOG_NORMAL
    assert len(param.config) == 5


def test_json_type_inference_categorical():
    config = {"values": [1, 2, 3]}
    param = HyperParameter("categorical_param", config)
    assert param.type == HyperParameter.CATEGORICAL
    assert param.config["nested"] == False
    assert param.config["nest_delimiter"] == "."
    # len is 2 because distribution key is inferred via default
    assert len(param.config) == 4


def test_json_type_inference_constant():
    config = {"value": "abcd"}
    param = HyperParameter("constant_param", config)
    assert param.type == HyperParameter.CONSTANT
    assert param.config["nested"] == False
    assert param.config["nest_delimiter"] == "."
    # len is 2 because distribution key is inferred via default
    assert len(param.config) == 4


def test_json_type_inference_q_normal():
    config = {"distribution": "q_normal", "q": 0.1}
    param = HyperParameter("q_normal_param", config)
    assert param.config["mu"] == 0
    assert param.config["sigma"] == 1
    assert param.config["nested"] == False
    assert param.config["nest_delimiter"] == "."
    assert param.type == HyperParameter.Q_NORMAL
    assert len(param.config) == 6


def test_json_type_inference_q_beta():
    config = {"distribution": "q_beta"}
    param = HyperParameter("q_beta_param", config)
    assert param.config["a"] == 1
    assert param.config["b"] == 1
    assert param.config["q"] == 1
    assert param.config["nested"] == False
    assert param.config["nest_delimiter"] == "."
    assert param.type == HyperParameter.Q_BETA
    assert len(param.config) == 6


def test_json_type_inference_beta():
    config = {"distribution": "beta"}
    param = HyperParameter("beta_param", config)
    assert param.config["a"] == 1
    assert param.config["b"] == 1
    assert param.config["nested"] == False
    assert param.config["nest_delimiter"] == "."
    assert param.type == HyperParameter.BETA
    assert len(param.config) == 5


def test_totally_invalid_config():
    config = {"invalid": "this config is not valid"}
    with pytest.raises(jsonschema.ValidationError):
        HyperParameter("invalid", config)


def test_validate_does_not_modify_passed_config():
    config = {"distribution": "normal"}
    config_save = config.copy()
    _ = HyperParameter("normal_test", config)
    assert config == config_save


def test_categorical_hyperparameter_no_values():
    config = {"values": []}
    with pytest.raises(jsonschema.ValidationError):
        HyperParameter("invalid_test", config)


def test_uniform_with_integer_min_max():
    # CLI-975
    # https://github.com/wandb/sweeps/pull/16
    config = {"distribution": "uniform", "min": 0, "max": 1}  # integers
    unif_param = HyperParameter("uniform_param", config)  # this should not raise
    assert unif_param.type == HyperParameter.UNIFORM


def test_hyperband_missing_eta_imputed():
    # sentry https://sentry.io/organizations/weights-biases/issues/2500192925/?referrer=slack
    config = {
        "command": [
            "${env}",
            "${interpreter}",
            "${program}",
            "sweep",
            "2dconv_lstm.stat.ac",
            "--dataset",
            "deam",
            "--temp_folder",
            "a_deam",
            "--batch-size",
        ],
        "early_terminate": {"min_iter": 3, "type": "hyperband"},
        "method": "random",
        "metric": {"goal": "minimize", "name": "val/loss"},
        "name": "AC-2DConvLSTM-Stat-DEAM",
        "parameters": {
            "dropout": {"values": ["0.15", "0.2", "0.25", "0.3", "0.4", "0.5"]},
            "lr": {"values": ["0.001", "0.005", "0.01"]},
            "momentum": {"values": ["0.8", "0.9", "0.95"]},
            "n_fft": {"value": 1024},
            "n_mels": {"value": 128},
        },
        "program": "exec.py",
        "project": "mer",
    }

    sc = SweepConfig(config)
    assert sc["early_terminate"]["eta"] == 3


def test_hyperband_missing_eta_imputed_incremental():
    # sentry https://sentry.io/organizations/weights-biases/issues/2500192925/?referrer=slack
    config = {
        "command": [
            "${env}",
            "${interpreter}",
            "${program}",
            "sweep",
            "2dconv_lstm.stat.ac",
            "--dataset",
            "deam",
            "--temp_folder",
            "a_deam",
            "--batch-size",
        ],
        "early_terminate": {"min_iter": 3, "type": "hyperband"},
        "method": "random",
        "metric": {"goal": "minimize", "name": "val/loss"},
        "name": "AC-2DConvLSTM-Stat-DEAM",
        "parameters": {
            "dropout": {"values": ["0.15", "0.2", "0.25", "0.3", "0.4", "0.5"]},
            "lr": {"values": ["0.001", "0.005", "0.01"]},
            "momentum": {"values": ["0.8", "0.9", "0.95"]},
            "n_fft": {"value": 1024},
            "n_mels": {"value": 128},
        },
        "program": "exec.py",
        "project": "mer",
    }

    sc = fill_validate_early_terminate(config)
    assert sc["early_terminate"]["eta"] == 3


def test_hyperband_incremental_corrects_goal():
    # sentry https://sentry.io/organizations/weights-biases/issues/2500192925/?referrer=slack
    config = {
        "command": [
            "${env}",
            "${interpreter}",
            "${program}",
            "sweep",
            "2dconv_lstm.stat.ac",
            "--dataset",
            "deam",
            "--temp_folder",
            "a_deam",
            "--batch-size",
        ],
        "early_terminate": {"min_iter": 3, "type": "hyperband"},
        "method": "random",
        "metric": {"goal": "minimise", "name": "val/loss"},
        "name": "AC-2DConvLSTM-Stat-DEAM",
        "parameters": {
            "dropout": {"values": ["0.15", "0.2", "0.25", "0.3", "0.4", "0.5"]},
            "lr": {"values": ["0.001", "0.005", "0.01"]},
            "momentum": {"values": ["0.8", "0.9", "0.95"]},
            "n_fft": {"value": 1024},
            "n_mels": {"value": 128},
        },
        "program": "exec.py",
        "project": "mer",
    }

    sc = hyperband_baseline_validate_and_fill(config)
    assert sc["early_terminate"]["eta"] == 3
    assert sc["metric"]["goal"] == "minimize"


def test_invalid_metric():
    config = {
        "method": "grid",
        "metric": "{'goal': 'maximize', 'name': 'val/loss'}",
        "parameters": {"a": {"values": [1, 2, 3, 4]}},
    }

    with pytest.raises(ValueError):
        fill_validate_metric(config)
