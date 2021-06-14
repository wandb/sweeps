import pytest
from jsonschema import ValidationError
from .. import next_run


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

    # check that no error is raised
    result = next_run(invalid_schema, [], validate=False)
    assert result is not None
