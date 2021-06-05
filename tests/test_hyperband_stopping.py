from .. import stop_runs, next_run, RunState, SweepRun


def test_hyperband_min_iter_bands():
    sweep_config = {
        "method": "grid",
        "metric": {"name": "loss", "goal": "minimize"},
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 3,
            "eta": 3,
        },
        "parameters": {"a": {"values": [1, 2, 3]}},
    }

    run = next_run(sweep_config, [])
    run.state = RunState.running
    run.history = [{"loss": 10} for _ in range(4)]
    run2 = next_run(sweep_config, [run])
    run2.state = RunState.running
    run2.history = [{"loss": 10 - i} for i in range(10)]
    to_stop = stop_runs(sweep_config, [run, run2])

    assert to_stop[0] is run
    assert to_stop[0].early_terminate_info["bands"][:3] == [3, 9, 27]


def test_hyperband_min_iter_bands_max():
    sweep_config = {
        "method": "grid",
        "metric": {"name": "accuracy", "goal": "maximize"},
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 3,
            "eta": 3,
        },
        "parameters": {"a": {"values": [1, 2, 3]}},
    }

    run = next_run(sweep_config, [])
    run.state = RunState.running
    run.history = [{"accuracy": 10} for _ in range(4)]
    run2 = next_run(sweep_config, [run])
    run2.state = RunState.running
    run2.history = [{"accuracy": 10 + i} for i in range(10)]
    to_stop = stop_runs(sweep_config, [run, run2])

    assert to_stop[0] is run
    assert to_stop[0].early_terminate_info["bands"][:3] == [3, 9, 27]


def test_hyperband_max_iter_bands():
    sweep_config = {
        "method": "grid",
        "metric": {"name": "loss", "goal": "minimize"},
        "early_terminate": {
            "type": "hyperband",
            "max_iter": 81,
            "eta": 3,
            "s": 3,
        },
        "parameters": {"a": {"values": [1, 2, 3]}},
    }

    run = next_run(sweep_config, [])
    run.state = RunState.running
    run.history = [{"loss": 10} for _ in range(4)]
    run2 = next_run(sweep_config, [run])
    run2.state = RunState.running
    run2.history = [{"loss": 10 - i} for i in range(10)]
    to_stop = stop_runs(sweep_config, [run, run2])

    assert to_stop[0] is run
    assert to_stop[0].early_terminate_info["bands"][:3] == [3, 9, 27]


def test_init_from_max_iter():
    sweep_config = {
        "method": "grid",
        "metric": {"name": "loss", "goal": "minimize"},
        "early_terminate": {
            "type": "hyperband",
            "max_iter": 18,
            "eta": 3,
            "s": 2,
        },
        "parameters": {"a": {"values": [1, 2, 3]}},
    }

    run = next_run(sweep_config, [])
    run.state = RunState.running
    run.history = [{"loss": 10} for _ in range(4)]
    run2 = next_run(sweep_config, [run])
    run2.state = RunState.running
    run2.history = [{"loss": 10 - i} for i in range(10)]
    to_stop = stop_runs(sweep_config, [run, run2])

    assert to_stop[0] is run
    assert to_stop[0].early_terminate_info["bands"] == [2, 6]


def test_single_run():

    sweep_config = {
        "method": "grid",
        "metric": {"name": "loss", "goal": "minimize"},
        "early_terminate": {
            "type": "hyperband",
            "max_iter": 18,
            "eta": 3,
            "s": 2,
        },
        "parameters": {"a": {"values": [1, 2, 3]}},
    }

    run = next_run(sweep_config, [])
    run.state = RunState.running
    run.history = [{"loss": 10 - i} for i in range(10)]
    to_stop = stop_runs(sweep_config, [run])
    assert len(to_stop) == 0


def test_2runs_band1_pass():

    sweep_config = {
        "method": "grid",
        "metric": {"name": "loss", "goal": "minimize"},
        "early_terminate": {
            "type": "hyperband",
            "max_iter": 18,
            "eta": 3,
            "s": 2,
        },
        "parameters": {"a": {"values": [1, 2, 3]}},
    }

    run = next_run(sweep_config, [])
    run.state = RunState.running
    run.history = [{"loss": 10}, {"loss": 10}, {"loss": 6}]
    run2 = next_run(sweep_config, [run])
    run2.state = RunState.running
    run2.history = [{"loss": 10 - i} for i in range(10)]
    to_stop = stop_runs(sweep_config, [run, run2])
    assert len(to_stop) == 0


def test_5runs_band1_stop_2():

    sweep_config = {
        "method": "grid",
        "metric": {"name": "loss", "goal": "minimize"},
        "early_terminate": {
            "type": "hyperband",
            "max_iter": 5,
            "eta": 2,
            "s": 2,
        },
        "parameters": {"a": {"values": [1, 2, 3]}},
    }

    runs = [
        SweepRun(
            name="a",
            state=RunState.finished,  # This wont be stopped because already stopped
            history=[
                {"loss": 10},
                {"loss": 9},
            ],
        ),
        SweepRun(
            name="b",
            state=RunState.running,  # This should be stopped
            history=[
                {"loss": 10},
                {"loss": 10},
            ],
        ),
        SweepRun(
            name="c",
            state=RunState.running,  # This passes band 1 but not band 2
            history=[
                {"loss": 10},
                {"loss": 8},
                {"loss": 8},
            ],
        ),
        SweepRun(
            name="d",
            state=RunState.running,
            history=[
                {"loss": 10},
                {"loss": 7},
                {"loss": 7},
            ],
        ),
        SweepRun(
            name="e",
            state=RunState.finished,
            history=[
                {"loss": 10},
                {"loss": 6},
                {"loss": 6},
            ],
        ),
    ]

    to_stop = stop_runs(sweep_config, runs)
    assert to_stop == runs[1:3]


def test_5runs_band1_stop_2_1stnoband():

    sweep_config = {
        "method": "grid",
        "metric": {"name": "loss", "goal": "minimize"},
        "early_terminate": {
            "type": "hyperband",
            "max_iter": 5,
            "eta": 2,
            "s": 2,
        },
        "parameters": {"a": {"values": [1, 2, 3]}},
    }

    runs = [
        SweepRun(
            name="a",
            state=RunState.finished,  # This wont be stopped because already stopped
            history=[
                {"loss": 10},
            ],
        ),
        SweepRun(
            name="b",
            state=RunState.running,  # This should be stopped
            history=[
                {"loss": 10},
                {"loss": 10},
            ],
        ),
        SweepRun(
            name="c",
            state=RunState.running,  # This passes band 1 but not band 2
            history=[
                {"loss": 10},
                {"loss": 8},
                {"loss": 8},
            ],
        ),
        SweepRun(
            name="d",
            state=RunState.running,
            history=[
                {"loss": 10},
                {"loss": 7},
                {"loss": 7},
            ],
        ),
        SweepRun(
            name="e",
            state=RunState.finished,
            history=[
                {"loss": 10},
                {"loss": 6},
                {"loss": 6},
            ],
        ),
    ]

    to_stop = stop_runs(sweep_config, runs)
    assert to_stop == runs[1:3]


def test_eta_3():

    sweep_config = {
        "method": "grid",
        "metric": {"name": "loss", "goal": "minimize"},
        "early_terminate": {
            "type": "hyperband",
            "max_iter": 9,
            "eta": 3,
            "s": 2,
        },
        "parameters": {"a": {"values": [1, 2, 3]}},
    }

    runs = [
        SweepRun(
            name="a",
            state=RunState.finished,  # This wont be stopped because already stopped
            history=[
                {"loss": 10},
                {"loss": 9},
            ],
        ),
        SweepRun(
            name="b",
            state=RunState.running,  # This should be stopped
            history=[
                {"loss": 10},
                {"loss": 10},
            ],
        ),
        SweepRun(
            name="c",
            state=RunState.running,  # This fails the first threeshold but snuck in so we wont kill
            history=[
                {"loss": 10},
                {"loss": 8},
                {"loss": 8},
                {"loss": 3},
            ],
        ),
        SweepRun(
            name="d",
            state=RunState.running,
            history=[
                {"loss": 10},
                {"loss": 7},
                {"loss": 7},
                {"loss": 4},
            ],
        ),
        SweepRun(
            name="e",
            state=RunState.running,  # this passes band 1 but doesn't pass band 2
            history=[
                {"loss": 10},
                {"loss": 6},
                {"loss": 6},
                {"loss": 6},
            ],
        ),
    ]

    # bands are at 1 and 3, thresholds are 7 and 4
    to_stop = stop_runs(sweep_config, runs)
    assert to_stop == [runs[1], runs[-1]]


def test_eta_3_max():

    sweep_config = {
        "method": "grid",
        "metric": {"name": "loss", "goal": "maximize"},
        "early_terminate": {
            "type": "hyperband",
            "max_iter": 9,
            "eta": 3,
            "s": 2,
        },
        "parameters": {"a": {"values": [1, 2, 3]}},
    }

    runs = [
        SweepRun(
            name="a",
            state=RunState.finished,  # This wont be stopped because already stopped
            history=[
                {"loss": -10},
                {"loss": -9},
            ],
        ),
        SweepRun(
            name="b",
            state=RunState.running,  # This should be stopped
            history=[
                {"loss": -10},
                {"loss": -10},
            ],
        ),
        SweepRun(
            name="c",
            state=RunState.running,  # This fails the first threeshold but snuck in so we wont kill
            history=[
                {"loss": -10},
                {"loss": -8},
                {"loss": -8},
                {"loss": -3},
            ],
        ),
        SweepRun(
            name="d",
            state=RunState.running,
            history=[
                {"loss": -10},
                {"loss": -7},
                {"loss": -7},
                {"loss": -4},
            ],
        ),
        SweepRun(
            name="e",
            state=RunState.running,  # this passes band 1 but doesn't pass band 2
            history=[
                {"loss": -10},
                {"loss": -6},
                {"loss": -6},
                {"loss": -6},
            ],
        ),
    ]

    # bands are at 1 and 3, thresholds are 7 and 4
    to_stop = stop_runs(sweep_config, runs)
    assert to_stop == [runs[1], runs[-1]]
