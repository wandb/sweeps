from typing import List, Union, Dict, Any

import numpy as np

from .config import SweepConfig
from .run import SweepRun, RunState


def hyperband_stop_runs(
    runs: List[SweepRun],
    config: Union[dict, SweepConfig],
    validate: bool = False,
) -> List[SweepRun]:
    """
    Suggest sweep runs to terminate early using Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization
      https://arxiv.org/pdf/1603.06560.pdf


    >>> to_stop = hyperband_stop_runs(
    ...    [SweepRun(
    ...        name="a",
    ...        state=RunState.finished,  # This is already stopped
    ...        history=[
    ...            {"loss": 10},
    ...            {"loss": 9},
    ...        ],
    ...    ),
    ...    SweepRun(
    ...        name="b",
    ...        state=RunState.running,  # This should be stopped
    ...        history=[
    ...            {"loss": 10},
    ...            {"loss": 10},
    ...        ],
    ...    ),
    ...    SweepRun(
    ...        name="c",
    ...        state=RunState.running,  # This passes band 1 but not band 2
    ...        history=[
    ...            {"loss": 10},
    ...            {"loss": 8},
    ...            {"loss": 8},
    ...        ],
    ...    ),
    ...    SweepRun(
    ...        name="d",
    ...        state=RunState.running,
    ...        history=[
    ...            {"loss": 10},
    ...            {"loss": 7},
    ...            {"loss": 7},
    ...        ],
    ...    ),
    ...    SweepRun(
    ...        name="e",
    ...        state=RunState.finished,
    ...        history=[
    ...            {"loss": 10},
    ...            {"loss": 6},
    ...            {"loss": 6},
    ...        ],
    ...    ),
    ... ],
    ... {
    ...    "method": "grid",
    ...    "metric": {"name": "loss", "goal": "minimize"},
    ...    "early_terminate": {
    ...        "type": "hyperband",
    ...        "max_iter": 5,
    ...        "eta": 2,
    ...        "s": 2,
    ...    },
    ...    "parameters": {"a": {"values": [1, 2, 3]}},
    ... })

    Args:
        runs: The runs in the sweep.
        config: The sweep's config.
        validate: Whether to validate `sweep_config` against the SweepConfig JSONschema.
           If true, will raise a Validation error if `sweep_config` does not conform to
           the schema. If false, will attempt to run the sweep with an unvalidated schema.

    Returns:
        List of runs to stop early.
    """

    # validate config and fill in defaults
    if validate:
        config = SweepConfig(config)

    if "metric" not in config:
        raise ValueError('Hyperband stopping requires "metric" section')

    if "early_terminate" not in config:
        raise ValueError('Hyperband stopping requires "early_terminate" section.')
    et_config = config["early_terminate"]

    if et_config["type"] != "hyperband":
        raise ValueError("Sweep config is not configured for hyperband stopping")

    if "max_iter" in et_config:
        max_iter = et_config["max_iter"]
        s = et_config["s"]
        eta = et_config["eta"]

        band = max_iter
        bands = []
        for i in range(s):
            band /= eta
            if band < 1:
                break
            bands.append(int(band))
        bands = sorted(bands)

    # another way of defining hyperband with min_iter and possibly eta
    elif "min_iter" in et_config:
        min_iter = et_config["min_iter"]
        eta = et_config["eta"]

        band = min_iter
        bands = []
        for i in range(100):
            bands.append(int(band))
            band *= eta
    else:
        raise ValueError(
            'invalid config for hyperband stopping: either "max_iter" or "min_iter" must be specified'
        )
    r = 1.0 / eta

    if len(bands) < 1:
        raise ValueError("Bands must be an array of length at least 1")
    if r < 0 or r > 1:
        raise ValueError("r must be a float between 0 and 1")

    terminate_runs: List[SweepRun] = []
    metric_name = config["metric"]["name"]

    all_run_histories = []  # we're going to look at every run
    for run in runs:
        history = run.metric_history(metric_name, filter_invalid=True)
        if config["metric"]["goal"] == "maximize":
            history = list(map(lambda x: -x, history))
        if len(history) > 0:
            all_run_histories.append(history)

    thresholds = []
    # iterate over the histories at every band and find the threshold for a run to be in the top r percentile
    for band in bands:
        # values of metric at iteration number "band"
        band_values = [h[band] for h in all_run_histories if len(h) > band]
        if len(band_values) == 0:
            threshold = np.inf
        else:
            threshold = sorted(band_values)[int((r) * len(band_values))]
        thresholds.append(threshold)

    info: Dict[str, Any] = {}
    info["lines"] = []
    info["lines"].append(
        "Bands: %s"
        % (
            ", ".join(
                [
                    "%s = %s" % (band, threshold)
                    for band, threshold in zip(bands, thresholds)
                ]
            )
        )
    )

    info["bands"] = bands
    info["thresholds"] = thresholds

    for run in runs:
        if run.state == RunState.running:
            history = run.metric_history(metric_name, filter_invalid=True)
            if config["metric"]["goal"] == "maximize":
                history = list(map(lambda x: -x, history))

            closest_band = -1
            closest_threshold = 0.0
            bandstr = ""
            termstr = ""
            for band, threshold in zip(bands, thresholds):
                if band < len(history):
                    closest_band = band
                    closest_threshold = threshold
                else:
                    break

            if closest_band != -1:  # no bands apply yet
                bandstr = " (Metric: %f Band: %d Threshold %f)" % (
                    min(history),
                    closest_band,
                    closest_threshold,
                )
                if min(history) > closest_threshold:
                    terminate_runs.append(run)
                    termstr = " STOP"

            run_info = info.copy()
            run_info["lines"].append(
                "Run: %s Step: %d%s%s" % (run.name, len(history), bandstr, termstr)
            )
            run.early_terminate_info = run_info

    return terminate_runs
