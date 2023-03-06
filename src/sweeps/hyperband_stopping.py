from copy import deepcopy
import pprint
from typing import Any, Dict, List, Union
import logging

import numpy as np

from .config import SweepConfig, fill_validate_early_terminate
from .config.schema import dereferenced_sweep_config_jsonschema, fill_validate_metric
from .run import RunState, SweepRun

_logger = logging.Logger("dev")


def hyperband_baseline_validate_and_fill(config: Dict) -> Dict:
    config = deepcopy(config)

    if "metric" not in config:
        raise ValueError('Hyperband stopping requires "metric" section')

    if "early_terminate" not in config:
        raise ValueError('Hyperband stopping requires "early_terminate" section.')

    if config["early_terminate"]["type"] != "hyperband":
        raise ValueError("Sweep config is not configured for hyperband stopping")

    config = fill_validate_metric(config)
    et_config = config["early_terminate"]

    # remove extra keys (that are already ignored anyway) from the early_terminate config dict to avoid triggering a
    # jsonschema validation error. Extra keys are disallowed by the schema but to maintain compatibility we
    # will allow them here.
    allowed_keys = list(
        dereferenced_sweep_config_jsonschema["definitions"]["hyperband_stopping"][
            "properties"
        ].keys()
    )

    _logger.debug(
        f"[hyperband_baseline_validate_and_fill]: allowed_keys={allowed_keys}"
    )

    to_delete = []
    for key in et_config:
        if key not in allowed_keys:
            to_delete.append(key)
    for key in to_delete:
        del et_config[key]

    # fill in defaults needed for hyperband
    config = fill_validate_early_terminate(config)

    return config


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

    _logger.debug(f"[hyperband_stop_runs] len-runs: {len(runs)}, validate: {validate}")
    _logger.debug(f"Passed config: {pprint.pformat(config)}")
    if len(runs) > 0:
        _logger.debug(f"last run full history: {runs[-1].history}")

    # validate config and fill in defaults
    if validate:
        # this fully validates the entire config
        config = SweepConfig(config)

    _logger.debug(f"Config after validation: {pprint.pformat(config)}")

    # this manually validates the parts of the config that are absolutely
    # required to do hyperband stopping, and fills in any associated defaults
    config = hyperband_baseline_validate_and_fill(config)

    _logger.debug(f"Config after val_and_fill: {pprint.pformat(config)}")

    et_config = config["early_terminate"]

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
        raise ValueError(
            "Bands must be an array of length at least 1. Try increasing s, decreasing eta, or increasing min_iter."
        )

    if r < 0 or r > 1:
        raise ValueError("r must be a float between 0 and 1")

    _logger.debug(f"bands: {bands}, r: {r}")

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
        band_values = [h[band - 1] for h in all_run_histories if len(h) >= band]
        if len(band_values) == 0:
            threshold = np.inf
        else:
            _logger.debug(f"(band: {band}) values: {band_values}")
            threshold = sorted(band_values)[int(r * len(band_values))]
        thresholds.append(threshold)

    _logger.debug(f"thresholds: {thresholds}")

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
                if band <= len(history):
                    closest_band = band
                    closest_threshold = threshold
                else:
                    break

            _logger.debug(
                f"({run.name}) closest_band: {closest_band}, closest_threshold: {closest_threshold}, len-history: {len(history)}"
            )

            if closest_band != -1:  # no bands apply yet
                # Conservative termination condition
                condition_val = min(history)
                if et_config.get("strict") is True:
                    # More aggresive, strict termination condition
                    condition_val = history[closest_band - 1]

                if condition_val > closest_threshold:
                    terminate_runs.append(run)
                    termstr = " STOP"

                _logger.debug(
                    f"({run.name}) metric_val: {condition_val}, use_strict?: {et_config.get('strict')}, terminated?: {condition_val > closest_threshold}"
                )

                bandstr = f" (Band: {closest_band} Metric: {condition_val} Threshold: {closest_threshold})"

            run_info = info.copy()
            run_info["lines"].append(
                f"Run: {run.name} Step: {len(history)} {bandstr} {termstr}"
            )
            run.early_terminate_info = run_info

    _logger.debug(f"runstates: {[f'{r.name}:{r.state}' for r in runs]}")

    return terminate_runs
