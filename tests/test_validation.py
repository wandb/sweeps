import pytest
from jsonschema import ValidationError
from .. import next_run, stop_runs
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
