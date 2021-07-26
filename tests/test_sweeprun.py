import pytest
from .. import SweepRun, RunState


def test_skipped_steps():
    run = SweepRun(
        name="a",
        state=RunState.running,
        history=[
            {"loss": 10},
            {"a": 9},
            {"a": 8},
            {"a": 7},
            {"loss": 6},
            {"a": 5},
            {"a": 4},
            {"a": 3},
            {"a": 2},
            {"loss": 1},
        ],
    )
    assert run.metric_history("loss") == [10, 6, 1]


def test_summary_metric_none():
    run = SweepRun(
        name="a",
        summary_metrics=None,
    )
    with pytest.raises(ValueError):
        run.summary_metric("loss")
