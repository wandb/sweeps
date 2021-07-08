from typing import List, Optional, Union, Any, Dict
from enum import Enum
import numpy as np

from pydantic import BaseModel, Field
from .config import SweepConfig
from ._types import floating


class RunState(str, Enum):
    proposed = "proposed"
    running = "running"
    finished = "finished"
    killed = "killed"
    crashed = "crashed"
    failed = "failed"
    preempted = "preempted"
    preempting = "preempting"


class SweepRun(BaseModel):
    """A wandb Run that is part of a Sweep.

    >>> run = SweepRun(
    ...   name="my_run",
    ...   state=RunState.running,
    ...   config={"a": {"value": 1}},
    ... )

    Args:
        name: Name of the run.
        state: State of the run.
        config: `dict` representation of the run's wandb.config.
        summaryMetrics: `dict` of summary statistics for the run.
        history: List of dicts containing the arguments to calls of wandb.log made during the run.
        search_info: Dict containing information produced by the search algorithm.
        early_terminate_info: Dict containing information produced by the early terminate algorithm.
    """

    name: Optional[str] = None
    summary_metrics: dict = Field(default_factory=lambda: {}, alias="summaryMetrics")
    history: List[dict] = Field(default_factory=lambda: [])
    config: dict = Field(default_factory=lambda: {})
    state: RunState = RunState.proposed
    search_info: Optional[Dict] = None
    early_terminate_info: Optional[Dict] = None

    class Config:
        use_enum_values = True
        allow_population_by_field_name = True

    def metric_history(self, metric_name: str) -> List[floating]:
        return [d[metric_name] for d in self.history if metric_name in d]

    def summary_metric(self, metric_name: str) -> floating:
        if metric_name not in self.summary_metrics:
            raise KeyError(f"{metric_name} is not a summary metric of this run.")
        return self.summary_metrics[metric_name]

    def metric_extremum(self, metric_name: str, kind: str) -> floating:
        """Calculate the maximum or minimum value of a specified metric.

        >>> run = SweepRun(history=[{'a': 1}, {'b': 3}, {'a': 2, 'b': 4}], summary_metrics={'a': 50})
        >>> assert run.metric_extremum('a', 'maximum') == 50

        Args:
            metric_name: The name of the target metric.
            kind: What kind of extremum to get (either "maximum" or "minimum").

        Returns:
            The maximum or minimum metric.
        """

        cmp_func = np.max if kind == "maximum" else np.min
        try:
            summary_metric = [self.summary_metric(metric_name)]
        except KeyError:
            summary_metric = []
        all_metrics = self.metric_history(metric_name) + summary_metric

        if len(all_metrics) == 0:
            raise ValueError(f"Cannot extract metric {metric_name} from run")

        def filter_func(x: Any) -> bool:
            try:
                return np.isscalar(x) and np.isfinite(x)
            except TypeError:
                return False

        all_metrics = list(filter(filter_func, all_metrics))

        if len(all_metrics) == 0:
            raise ValueError("Run does not have any finite metric values")

        return cmp_func(all_metrics)


def next_run(
    sweep_config: Union[dict, SweepConfig],
    runs: List[SweepRun],
    validate: bool = False,
    **kwargs,
) -> Optional[SweepRun]:
    """Calculate the next run in a sweep.

    >>> suggested_run = next_run({
    ...    'method': 'grid',
    ...    'parameters': {'a': {'values': [1, 2, 3]}}
    ... }, [])
    >>> assert suggested_run.config['a']['value'] == 1

    Args:
        sweep_config: The config for the sweep.
        runs: List of runs in the sweep.
        validate: Whether to validate `sweep_config` against the SweepConfig JSONschema.
           If true, will raise a Validation error if `sweep_config` does not conform to
           the schema. If false, will attempt to run the sweep with an unvalidated schema.

    Returns:
        The suggested run.
    """

    from .grid_search import grid_search_next_run
    from .random_search import random_search_next_run
    from .bayes_search import bayes_search_next_run

    # validate the sweep config
    if validate:
        sweep_config = SweepConfig(sweep_config)

    # this access is safe due to the jsonschema
    method = sweep_config["method"]

    if method == "grid":
        return grid_search_next_run(runs, sweep_config, validate=validate, **kwargs)
    elif method == "random":
        return random_search_next_run(sweep_config, validate=validate)
    elif method == "bayes" or isinstance(method, dict) and "bayes" in method.keys():
        return bayes_search_next_run(runs, sweep_config, validate=validate, **kwargs)
    else:
        raise ValueError(
            f'Invalid search type {method}, must be one of ["grid", "random", "bayes"]'
        )


def stop_runs(
    sweep_config: Union[dict, SweepConfig],
    runs: List[SweepRun],
    validate: bool = False,
) -> List[SweepRun]:
    """Calculate the runs in a sweep to stop by early termination.

    >>> to_stop = stop_runs({
    ...    "method": "grid",
    ...    "metric": {"name": "loss", "goal": "minimize"},
    ...    "early_terminate": {
    ...        "type": "hyperband",
    ...        "max_iter": 5,
    ...        "eta": 2,
    ...        "s": 2,
    ...    },
    ...    "parameters": {"a": {"values": [1, 2, 3]}},
    ... }, [
    ...    SweepRun(
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
    ...])


    Args:
        sweep_config: The config for the sweep.
        runs: List of runs in the sweep.
        validate: Whether to validate `sweep_config` against the SweepConfig JSONschema.
           If true, will raise a Validation error if `sweep_config` does not conform to
           the schema. If false, will attempt to run the sweep with an unvalidated schema.


    Returns:
        A list of the runs to stop.
    """

    from .hyperband_stopping import hyperband_stop_runs

    # validate the sweep config
    if validate:
        sweep_config = SweepConfig(sweep_config)

    if "metric" not in sweep_config:
        raise ValueError('early terminate requires "metric" section')

    if "early_terminate" not in sweep_config:
        raise ValueError('early terminate requires "early_terminate" section.')
    et_type = sweep_config["early_terminate"]["type"]

    if et_type == "hyperband":
        return hyperband_stop_runs(runs, sweep_config, validate=validate)
    else:
        raise ValueError(
            f'Invalid early stopping type {et_type}, must be one of ["hyperband"]'
        )
