from sweeps import util


def test_dict_has_nested_key_reports_missing_inner_key():
    config = {"env_params": {"value": {"env_id": "Env-v0"}}}

    assert util.dict_has_nested_key(config, ["env_params", "value", "env_id"])
    assert not util.dict_has_nested_key(config, ["env_params", "value", "env_name"])
