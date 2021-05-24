from sweeps.params import HyperParameter


def test_json_type_inference_int_uniform():
    config = {"min": 0, "max": 1}
    param = HyperParameter("int_unif_param", config)
    assert param.type == HyperParameter.INT_UNIFORM


def test_json_type_inference_uniform():
    config = {"min": 0.0, "max": 1.0}
    param = HyperParameter("unif_param", config)
    assert param.type == HyperParameter.UNIFORM


def test_json_type_inference_and_imputation_normal():
    config = {"distribution": "normal"}
    param = HyperParameter("normal_param", config)
    assert param.config["mu"] == 0
    assert param.config["sigma"] == 1
    assert param.type == HyperParameter.NORMAL


def test_json_type_inference_and_imputation_lognormal():
    config = {"distribution": "log_normal"}
    param = HyperParameter("log_normal_param", config)
    assert param.config["mu"] == 0
    assert param.config["sigma"] == 1
    assert param.type == HyperParameter.LOG_NORMAL


def test_json_type_inference_categorical():
    config = {"values": [1, 2, 3]}
    param = HyperParameter("categorical_param", config)
    assert param.type == HyperParameter.CATEGORICAL


def test_json_type_inference_constant():
    config = {"value": "abcd"}
    param = HyperParameter("constant_param", config)
    assert param.type == HyperParameter.CONSTANT


def test_json_type_inference_q_normal():
    config = {"distribution": "q_normal", "q": 0.1}
    param = HyperParameter("q_normal_param", config)
    assert param.config["mu"] == 0
    assert param.config["sigma"] == 1
    assert param.type == HyperParameter.Q_NORMAL
