from typing import List, Optional, Union
from enum import Enum
import numpy as np

from dataclasses import dataclass, field

from .config import SweepConfig


RunState = Enum(
    "run_states",
    (
        "proposed",
        "running",
        "finished",
        "killed",
        "crashed",
        "failed",
        "preempted",
        "preempting",
    ),
)


@dataclass
class Run:
    """Minimal representation of a W&B Run for sweeps.

    Attributes
    ----------
    name : str
        Name of the run.
    state : {'running', 'finished', 'killed', 'crashed', 'failed', 'preempting', 'preempted', 'proposed'}
        State of the run.
    config : dict
        dict representation of the run's wandb.config. E.g.,
        `{'config_variable_1': 1, 'config_variable_2': 2, 'optimizer': 'sgd'}`
    summaryMetrics : dict
        dict of summary statistics for the run. E.g., `{'loss': 0.5, 'accuracy': 0.9}`.
    history : list of dict
        Iterable of dicts containing the arguments to calls of wandb.log
        made during the run. E.g., [{"loss": 10}, {"a": 9}, {"a": 8}, {"a": 7}]
    optimizer_info: dict
        For runs in the proposed state, information produced by the optimizer. E.g.,
        {'improvement_prob': 0.2}
    """

    name: Optional[str] = None
    summary_metrics: dict = field(default_factory=lambda: {})
    history: List[dict] = field(default_factory=lambda: [])
    config: dict = field(default_factory=lambda: {})
    state: RunState = RunState.proposed
    optimizer_info: Optional[dict] = None

    def metric_history(self, metric_name: str) -> List[np.floating]:
        return [
            d[metric_name]
            for d in self.history
            if d[metric_name] is not None and np.isfinite(d[metric_name])
        ]

    def summary_metric(self, metric_name: str) -> np.floating:
        if metric_name not in self.summary_metrics:
            raise KeyError(f"{metric_name} is not a summary metric of this run.")
        return self.summary_metrics[metric_name]

    def metric_extremum(self, metric_name: str, kind: str) -> np.floating:
        """Extract the value of the target optimization metric from a
         specified sweep run.

        Parameters
        ----------
        sweep_config: SweepConfig or dict
            The sweep configuration, where the name of the target metric
            is specified.
        run: Run
            The run to extract the value of the metric from.
        default: float, optional, default None
            The default value to use if no metric is found.

        Returns
        -------
        metric: float
            The run's metric.
        """

        cmp_func = np.max if kind == "maximum" else np.min
        all_metrics = self.metric_history(metric_name) + [
            self.summary_metric(metric_name)
        ]
        return cmp_func(all_metrics)


def next_run(
    sweep_config: Union[dict, SweepConfig], runs: List[Run], **kwargs
) -> Optional[Run]:
    """Calculate the next run in a sweep given the Sweep config and the list of runs already in progress or finished. Returns the next run, or None if the parameter space is exhausted."""

    from .grid_search import grid_search_next_run
    from .random_search import random_search_next_run

    # from .bayes_search import bayes_search_next_run

    # this access is safe due to the jsonschema
    method = sweep_config["method"]

    if method == "grid":
        return grid_search_next_run(runs, sweep_config, **kwargs)
    elif method == "random":
        return random_search_next_run(sweep_config)
    elif method == "bayes":
        # return bayes_search_next_run(runs, sweep_config, **kwargs)
        #  TODO: implement this
        raise NotImplementedError
    else:
        raise ValueError(
            f'Invalid search type {method}, must be one of ["grid", "random", "bayes"]'
        )
