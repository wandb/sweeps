from typing import List

import numpy.typing as npt
from numpy.random import randint
from numpy.random import uniform


from .. import envelope_stopping, stop_runs, SweepRun, RunState
from .._types import integer, floating


def synthetic_loss(
    start: npt.ArrayLike,
    asympt: npt.ArrayLike,
    decay: npt.ArrayLike,
    noise: npt.ArrayLike,
    length: integer,
) -> List[floating]:
    val = start
    history = []
    for ii in range(length):
        history.append(val)
        val += uniform(-noise, noise)
        val -= (val - asympt) * decay
    return history


def synthetic_loss_family(num: integer) -> List[List[floating]]:
    histories = []
    for ii in range(num):
        history = synthetic_loss(
            uniform(4, 20), uniform(2, 3.0), uniform(0.05, 0.4), 0.5, randint(10, 20)
        )
        histories.append(history)
    return histories


def test_envelope_terminate_modules():
    hs = synthetic_loss_family(20)
    m = []
    for h in hs:
        m.append(min(h))
    top_hs = envelope_stopping.histories_for_top_n(hs, m, 5)
    envelope = envelope_stopping.envelope_from_histories(top_hs, 30)
    tries = 10
    for i in range(tries):
        new_history = synthetic_loss(
            20 + uniform(4, 20), 20.0, uniform(0.05, 0.4), 0.5, randint(10, 40)
        )
        assert not envelope_stopping.is_inside_envelope(new_history, envelope)
    for i in range(tries):
        new_history = synthetic_loss(
            uniform(0, 2.0), 20.0, uniform(0, 1.0), 0.5, randint(10, 50)
        )
        print(new_history)
        print(envelope)
        assert envelope_stopping.is_inside_envelope(new_history, envelope)


def test_envelope_terminate_end2end():

    sweep_config = {
        "method": "grid",
        "metric": {"name": "loss", "goal": "minimize"},
        "early_terminate": {
            "type": "envelope",
        },
        "parameters": {"a": {"values": [1, 2, 3]}},
    }

    hs = synthetic_loss_family(20)
    runs = []

    for h in hs:
        history = [{"loss": loss} for loss in h]
        run = SweepRun(
            history=history,
            summary_metrics=min(history, key=lambda x: x["loss"]),
            state=RunState.finished,
        )
        runs.append(run)

    new_history = [
        {"loss": loss}
        for loss in synthetic_loss(
            20 + uniform(4, 20), 20.0, uniform(0.05, 0.4), 0.5, randint(10, 40)
        )
    ]

    run = SweepRun(
        history=new_history,
        summary_metrics=min(new_history, key=lambda x: x["loss"]),
        state=RunState.running,
    )

    to_stop = stop_runs(sweep_config, runs + [run])
    assert len(to_stop) == 1
    assert to_stop[0] is run
