from ..params import HyperParameter


def test_json_type_inference_int_uniform():
    config = {"min": 0, "max": 1}
    param = HyperParameter("int_unif_param", config)
    assert param.type == HyperParameter.INT_UNIFORM
    assert len(param.config) == 2


def test_json_type_inference_uniform():
    config = {"min": 0.0, "max": 1.0}
    param = HyperParameter("unif_param", config)
    assert param.type == HyperParameter.UNIFORM
    assert len(param.config) == 2


def test_json_type_inference_and_imputation_normal():
    config = {"distribution": "normal"}
    param = HyperParameter("normal_param", config)
    assert param.config["mu"] == 0
    assert param.config["sigma"] == 1
    assert param.type == HyperParameter.NORMAL
    assert len(param.config) == 3


def test_json_type_inference_and_imputation_lognormal():
    config = {"distribution": "log_normal"}
    param = HyperParameter("log_normal_param", config)
    assert param.config["mu"] == 0
    assert param.config["sigma"] == 1
    assert param.type == HyperParameter.LOG_NORMAL
    assert len(param.config) == 3


def test_json_type_inference_categorical():
    config = {"values": [1, 2, 3]}
    param = HyperParameter("categorical_param", config)
    assert param.type == HyperParameter.CATEGORICAL
    # TODO(dag): infer distribution key via default
    assert len(param.config) == 1


def test_json_type_inference_constant():
    config = {"value": "abcd"}
    param = HyperParameter("constant_param", config)
    assert param.type == HyperParameter.CONSTANT
    # TODO(dag): infer distribution key via default
    assert len(param.config) == 1


def test_json_type_inference_q_normal():
    config = {"distribution": "q_normal", "q": 0.1}
    param = HyperParameter("q_normal_param", config)
    assert param.config["mu"] == 0
    assert param.config["sigma"] == 1
    assert param.type == HyperParameter.Q_NORMAL
    assert len(param.config) == 4


def test_json_type_inference_q_beta():
    config = {"distribution": "q_beta"}
    param = HyperParameter("q_beta_param", config)
    assert param.config["a"] == 1
    assert param.config["b"] == 1
    assert param.config["q"] == 1
    assert param.type == HyperParameter.Q_BETA
    assert len(param.config) == 4


def test_json_type_inference_beta():
    config = {"distribution": "beta"}
    param = HyperParameter("beta_param", config)
    assert param.config["a"] == 1
    assert param.config["b"] == 1
    assert param.type == HyperParameter.BETA
    assert len(param.config) == 3
