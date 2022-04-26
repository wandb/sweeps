import logging
from typing import Any, Dict, Union

from ray import tune
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray.tune.schedulers import (
    HyperBandScheduler,
    FIFOScheduler,
)

from .config import SweepConfig
from .params import HyperParameter, HyperParameterSet

_logger = logging.getLogger(__name__)

def convert_SweepRun_to_TuneRun(
    sweep_config : SweepConfig,
) -> Dict[str, Any]:
    """ Convert SweepConfig to a Ray Tune Search Space.

    Important keys in Ray Tune Run:

    metric: Metric to optimize. This metric should be reported
        with `tune.report()`. If set, will be passed to the search
        algorithm and scheduler.
    mode: Must be one of [min, max]. Determines whether objective is
        minimizing or maximizing the metric attribute. If set, will be
        passed to the search algorithm and scheduler.
    ...
    config: Algorithm-specific configuration for Tune variant
        generation (e.g. env, hyperparams). Defaults to empty dict.
        Custom search algorithms may ignore this.
    ...
    search_alg: Search algorithm for
        optimization. You can also use the name of the algorithm.
    scheduler: Scheduler for executing
        the experiment. Choose among FIFO (default), MedianStopping,
        AsyncHyperBand, HyperBand and PopulationBasedTraining. Refer to
        ray.tune.schedulers for more options. You can also use the
        name of the scheduler.
    """
    tune_run: Dict[str, Any] = {
        "metric" : None,
        "mode" : None,
        "config" : dict(),
        "search_alg" : None,
        "scheduler" : None,
    }

    _metric: Dict[str, Any] = sweep_config.get("metric", None)
    if _metric is None:
        raise ValueError(f"SweepConfig must have a 'metric' key.")
    else:
        if _metric.get("name", None) is None:
            raise ValueError(f"config['metric'] must have a 'name' key.")
        tune_run['metric'] = _metric["name"]

        if _metric.get("goal", None) is None:
            raise ValueError(f"config['metric'] must have a 'goal' key.")
        elif _metric['goal'].lower() in ["minimum", "min", "minimize"]:
            tune_run['mode'] = 'min'
        elif _metric['goal'].lower() in ["maximum", "max", "maximize"]:
            tune_run['mode'] = 'max'
        else:
            raise ValueError(f"Unsupported config['metric']['goal'] {_metric['goal']}")

    _early_terminate: Dict[str, Any] = sweep_config.get("early_terminate", None)
    if _early_terminate is None:
        _logger.warning(f'No early termination condition found, using FIFO scheduler.')
        tune_run['scheduler'] = FIFOScheduler()
    else:
        if _early_terminate.get("type", None) is None:
            raise ValueError(f"config['early_terminate'] must have a 'type' key.")
        if _early_terminate["type"] == "hyperband":
            tune_run['scheduler'] = HyperBandScheduler(
                metric=tune_run['metric'], mode=tune_run['mode'])

    _method: Dict[str, Any] = sweep_config.get("method", None)
    if _method is None:
        raise ValueError(f"SweepConfig must have a 'method' key.")
    elif _method == "random":
        tune_run['search_alg'] = BasicVariantGenerator()
    elif _method == "grid":
        tune_run['search_alg'] = BasicVariantGenerator()
    elif _method == "bayes":
        tune_run['search_alg'] = BayesOptSearch()
    else:
        raise ValueError(f"Unknown search method '{_method}'.")
    
    _parameters: Dict[str, Any] = sweep_config.get("parameters", None)
    if _parameters is None:
        raise ValueError(f"SweepConfig must have a 'parameters' key.")
    params = HyperParameterSet.from_config(_parameters)
    # Uses https://docs.ray.io/en/latest/tune/api_docs/search_space.html
    for param in params:
        if param.type == HyperParameter.CONSTANT:
            tune_run['config'][param.name] = param.value
        elif param.type == HyperParameter.CATEGORICAL:
            # Sample an option uniformly from the specified choices
            tune_run['config'][param.name] = tune.choice(param.config["values"])
        elif param.type == HyperParameter.CATEGORICAL_PROB:
            # Sample an option uniformly from the specified choices
            _logger.warning(f"CATEGORICAL_PROB not implemented.")
            tune_run['config'][param.name] = tune.choice(param.config["values"])
        elif param.type == HyperParameter.INT_UNIFORM:
            # Sample a float uniformly between 3.2 and 5.4,
            # rounding to increments of 0.2
            _logger.warning(f"CATEGORICAL_PROB not implemented.")
            tune_run['config'][param.name] = tune.quniform(3.2, 5.4, 0.2)
        elif param.type == HyperParameter.UNIFORM:
            # Sample a float uniformly between -5.0 and -1.0
            _logger.warning(f"CATEGORICAL_PROB not implemented.")
            tune_run['config'][param.name] = tune.uniform(-5, -1)
        elif param.type == HyperParameter.LOG_UNIFORM_V1:
            # Sample a float uniformly between 0.0001 and 0.01, while
            # sampling in log space
            _logger.warning(f"CATEGORICAL_PROB not implemented.")
            tune_run['config'][param.name] = tune.loguniform(1e-4, 1e-2)
        elif param.type == HyperParameter.LOG_UNIFORM_V2:
            # Sample a float uniformly between 0.0001 and 0.01, while
            # sampling in log space
            _logger.warning(f"CATEGORICAL_PROB not implemented.")
            tune_run['config'][param.name] = tune.loguniform(1e-4, 1e-2)
        elif param.type == HyperParameter.INV_LOG_UNIFORM_V1:
            _logger.warning(f"CATEGORICAL_PROB not implemented.")
            tune_run['config'][param.name] = tune.loguniform(1e-4, 1e-2)
        elif param.type == HyperParameter.INV_LOG_UNIFORM_V2:
            _logger.warning(f"CATEGORICAL_PROB not implemented.")
            tune_run['config'][param.name] = tune.loguniform(1e-4, 1e-2)
        elif param.type == HyperParameter.Q_UNIFORM:
            # Sample a float uniformly between 3.2 and 5.4,
            # rounding to increments of 0.2
            _logger.warning(f"CATEGORICAL_PROB not implemented.")
            tune_run['config'][param.name] = tune.quniform(3.2, 5.4, 0.2)
        elif param.type == HyperParameter.Q_LOG_UNIFORM_V1:
            _logger.warning(f"CATEGORICAL_PROB not implemented.")
            pass
        elif param.type == HyperParameter.Q_LOG_UNIFORM_V2:
            _logger.warning(f"CATEGORICAL_PROB not implemented.")
            pass
        elif param.type == HyperParameter.NORMAL:
            _logger.warning(f"CATEGORICAL_PROB not implemented.")
            pass
        elif param.type == HyperParameter.Q_NORMAL:
            _logger.warning(f"CATEGORICAL_PROB not implemented.")
            pass
        elif param.type == HyperParameter.LOG_NORMAL:
            _logger.warning(f"CATEGORICAL_PROB not implemented.")
            pass
        elif param.type == HyperParameter.Q_LOG_NORMAL:
            _logger.warning(f"CATEGORICAL_PROB not implemented.")
            pass
        elif param.type == HyperParameter.BETA:
            _logger.warning(f"CATEGORICAL_PROB not implemented.")
            pass
        elif param.type == HyperParameter.Q_BETA:
            _logger.warning(f"CATEGORICAL_PROB not implemented.")
            pass
        else:
            raise ValueError(f"Unsupported parameter type '{param.type}'.")

    return tune_run