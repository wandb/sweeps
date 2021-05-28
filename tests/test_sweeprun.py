from sweeps import SweepRun, RunState


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
