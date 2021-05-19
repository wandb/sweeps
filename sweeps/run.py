from typing import List, Optional
from enum import Enum
import numpy as np

from dataclasses import dataclass, field


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
    """A W&B Run.

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
    """

    name: Optional[str] = None
    summary_metrics: dict = field(default_factory=lambda: {})
    history: List[dict] = field(default_factory=lambda: [])
    config: dict = field(default_factory=lambda: {})
    state: RunState = RunState.proposed

    def metric_history(self, metric_name: str) -> List[float]:
        # TODO: remove maxmimze from this func
        return [
            d[metric_name]
            for d in self.history
            if d[metric_name] is not None and np.isfinite(d[metric_name])
        ]

    def summary_metric(self, metric_name: str) -> float:
        if metric_name not in self.summary_metrics:
            raise KeyError(f"{metric_name} is not a summary metric of this run.")
        return self.summary_metrics[metric_name]

    def best_metric(self, metric_name: str, maximize: bool) -> float:
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

        cmp_func = max if maximize else min
        all_metrics = self.metric_history(metric_name) + [
            self.summary_metric(metric_name)
        ]
        return cmp_func(all_metrics)


@dataclass
class Suggestion:
    # run Can be none to suggest terminating the sweep.
    run: Optional[Run]
    reason: str = ""
