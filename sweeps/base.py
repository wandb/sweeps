from .util import is_nan_or_nan_string
from .config.cfg import SweepConfig
from typing import Iterable, Union, Optional
from enum import Enum

from dataclasses import dataclass


RunState = Enum(
    "run_states",
    (
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
    state : {'running', 'finished', 'killed', 'crashed', 'failed', 'preempting', 'preempted'}
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

    name: str
    state: RunState
    config: dict
    summaryMetrics: dict
    history: Iterable[dict]


@dataclass
class Sweep:
    """A W&B Hyperparameter Sweep.

    Attributes
    ----------
    runs : iterable of `Run`s
        The runs associated with the sweep.
    config : SweepConfig
        The configuration of the sweep.
    """

    runs: Iterable[Run]
    config: SweepConfig


class Search:
    """Base class for Hyperparameter sweep search methods."""

    def _metric_from_run(
        self,
        sweep_config: Union[SweepConfig, dict],
        run: Run,
        default: Optional[float] = None,
    ) -> float:
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
        metric = None
        metric_name = sweep_config["metric"]["name"]

        maximize = False
        if "goal" in sweep_config["metric"]:
            if sweep_config["metric"]["goal"] == "maximize":
                maximize = True

        # Use summary to find metric
        if metric_name in run.summaryMetrics:
            metric = run.summaryMetrics[metric_name]
            # Exclude None or NaN summary metrics.
            if metric is None or is_nan_or_nan_string(metric):
                metric = None
        if maximize and metric is not None:
            metric = -metric

        # Use history to find metric (if available)
        metric_history = []
        run_history = getattr(run, "history", [])
        for line in run_history:
            m = line.get(metric_name)
            if m is None:
                continue
            if is_nan_or_nan_string(m):
                continue
            metric_history.append(m)
        if maximize:
            metric_history = [-m for m in metric_history]

        # find minimum from summary or history
        if metric_history:
            if metric:
                metric_history.append(metric)
            metric = min(metric_history)

        # use default if specified
        if metric is None:
            if default is None:
                raise ValueError("Couldn't find summary metric {}".format(metric_name))
            metric = default
        return metric

    def next_run(self, sweep: Sweep) -> Optional[Run]:
        """Calculate the next run in the sweep, update the sweep's list of runs,
        then return the new run.

        Parameters
        ----------
        sweep: Sweep
            The sweep to calculate a run for.

        Returns
        -------
        next_run: Run or NoneType
            None if all work complete for this sweep. Otherwise, the next
            run in the sweep.
        """
        raise NotImplementedError


class EarlyTerminate:
    def _load_metric_name_and_goal(self, sweep_config):
        if "metric" not in sweep_config:
            raise ValueError("Key 'metric' required for early termination")

        self.metric_name = sweep_config["metric"]["name"]

        self.maximize = False
        if "goal" in sweep_config["metric"]:
            if sweep_config["metric"]["goal"] == "maximize":
                self.maximize = True

    def _load_run_metric_history(self, run):
        metric_history = []
        for line in run.history:
            if self.metric_name in line:
                m = line[self.metric_name]
                metric_history.append(m)

        # Filter out bad values
        metric_history = [
            x for x in metric_history if x is not None and not is_nan_or_nan_string(x)
        ]
        if self.maximize:
            metric_history = [-m for m in metric_history]

        return metric_history

    def stop_runs(self, sweep_config, runs):
        return [], {}
