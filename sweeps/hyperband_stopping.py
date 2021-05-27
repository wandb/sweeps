"""Hyperband Early Terminate."""

from typing import List, Union, Dict

import numpy as np

from .config import SweepConfig
from .sweeprun import SweepRun


def stop_runs_hyperband(
    runs: List[SweepRun],
    config: Union[dict, SweepConfig],
) -> List[SweepRun]:
    """
    Implementation of the Hyperband algorithm from
      Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization
      https://arxiv.org/pdf/1603.06560.pdf

    Arguments
    bands - Array of iterations to potentially early terminate algorithms
    r - float in [0, 1] - fraction of runs to allow to pass through a band
        r=1 means let all runs through and r=0 means let no runs through
    """

    # validate config and fill in defaults
    config = SweepConfig(config)

    if "metric" not in config:
        raise ValueError('Hyperband stopping requires "metric" section')

    if "early_terminate" not in config:
        raise ValueError('Hyperband stopping requires "early_terminate" section.')
    et_config = config["early_terminate"]

    if "hyperband_stopping" not in et_config:
        raise ValueError("Sweep config is not configured for hyperband stopping")
    hb_config = et_config["hyperband_stopping"]

    if "max_iter" in hb_config:
        max_iter = hb_config["max_iter"]
        s = hb_config["s"]
        eta = hb_config["eta"]

        band = max_iter
        bands = []
        for i in range(s):
            band /= eta
            if band < 1:
                break
            bands.append(int(band))
        bands = sorted(bands)

    # another way of defining hyperband with min_iter and possibly eta
    elif "min_iter" in hb_config:
        min_iter = hb_config["min_iter"]
        eta = hb_config["eta"]

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
        history = run.metric_history(metric_name)
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

    info: Dict[str, List] = {}
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

    for run in runs:
        if run.state == "running":
            history = run.metric_history(metric_name)
            if config["metric"]["goal"] == "maximize":
                history = list(map(lambda x: -x, history))

            closest_band = -1
            closest_threshold = 0
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
            run.optimizer_info = run_info

    return terminate_runs
