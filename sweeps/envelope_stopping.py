"""Envelope Early Terminate.

Library to help with early termination of runs Here we use a strategy
where we take the top k runs or top k percent of runs and then we build
up an envelope where we stop jobs where the metric doesn't get better
"""

from typing import Union, List

import numpy as np
import numpy.typing as npt

from . import SweepRun, SweepConfig, RunState
from .types import integer, floating


def histories_for_top_n(
    histories: npt.ArrayLike, metrics: npt.ArrayLike, n: integer = 3
) -> npt.ArrayLike:
    metrics = np.array(metrics)
    histories = np.asarray(histories)
    indices = np.argpartition(-metrics, -n)[-n:]
    top_n_histories = []
    for index in indices:
        top_n_histories.append(histories[index])
    return top_n_histories


def envelope_from_histories(
    histories: npt.ArrayLike, envelope_len: integer
) -> npt.ArrayLike:
    envelope = []
    cum_min_hs = []
    for h in histories:
        cur_min = np.inf
        cum_min = []
        for j in range(envelope_len):
            if j < len(h):
                val = h[j]
            else:
                val = np.nan
            cur_min = min(cur_min, val)
            cum_min.append(cur_min)
        cum_min_hs.append(cum_min)
    for jj in range(envelope_len):
        envelope.append(max([h[jj] for h in cum_min_hs]))
    return envelope


def is_inside_envelope(
    history: npt.ArrayLike, envelope: npt.ArrayLike, ignore_first_n_iters: integer = 0
) -> bool:
    if len(history) <= ignore_first_n_iters:
        return True

    min_val = min(history)
    cur_iter = len(history) - 1
    if cur_iter >= len(envelope):
        cur_iter = len(envelope) - 1
    return min_val < envelope[cur_iter]


def envelope_stop_runs(
    runs: List[SweepRun],
    config: Union[dict, SweepConfig],
    fraction: floating = 0.3,
    min_runs: integer = 3,
    start_iter: integer = 3,
) -> List[SweepRun]:

    # validate config and fill in defaults
    config = SweepConfig(config)

    if "metric" not in config:
        raise ValueError('Hyperband stopping requires "metric" section')

    if "early_terminate" not in config:
        raise ValueError('Hyperband stopping requires "early_terminate" section.')
    et_config = config["early_terminate"]

    if et_config["type"] != "envelope":
        raise ValueError("Sweep config is not configured for envelope stopping")

    terminate_runs = []
    metric_name = config["metric"]["name"]
    goal = config["metric"]["goal"]

    complete_run_histories = []
    complete_run_metrics = []
    for run in runs:
        if run.state == RunState.finished:  # complete run
            history = run.metric_history(metric_name)
            if goal == "maximize":
                history = [-x for x in history]
            if len(history) > 0:
                complete_run_histories.append(history)
                complete_run_metrics.append(min(history))

    complete_runs_count = len(complete_run_histories)
    if complete_runs_count < min_runs:
        return []

    n = max(int(np.ceil(complete_runs_count * fraction)), min_runs)

    histories = histories_for_top_n(complete_run_histories, complete_run_metrics, n)
    envelope_len = max([len(h) for h in histories])
    envelope = envelope_from_histories(histories, envelope_len)

    for run in runs:
        if run.state == RunState.running:
            history = run.metric_history(metric_name)
            if goal == "maximize":
                history = [-x for x in history]

            if not is_inside_envelope(
                history, envelope, ignore_first_n_iters=start_iter
            ):
                terminate_runs.append(run)
    return terminate_runs
