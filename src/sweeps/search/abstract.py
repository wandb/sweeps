import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from ..config import SweepConfig, validate_search
from ..params import HyperParameter, HyperParameterSet
from ..run import SweepRun

_log = logging.getLogger(__name__)


class AbstractSearch(ABC):
    """Abstract base class for Search Algotithms."""

    def __init__(
        self,
        sweep_config: Union[Dict, SweepConfig],
        validate: bool = False,
        random_state: Union[np.random.RandomState, int] = 42,
    ) -> None:
        if validate:
            _log.info("Ensuring sweep config is properly formatted")
            sweep_config = SweepConfig(sweep_config)
        else:
            validate_search(sweep_config)
        self.sweep_config = sweep_config
        # Ensures repeatably random behavior
        self.random_state = random_state  # type: ignore
        # Create hyperparameters from SweepConfig
        _params: List[HyperParameter] = []
        for _name, _config in sorted(sweep_config["parameters"].items()):
            _params.append(HyperParameter(_name, _config))
        self.params: HyperParameterSet = HyperParameterSet(_params)

    @property
    def random_state(self) -> np.random.RandomState:
        return self._random_state

    @random_state.setter
    def random_state(self, random_state: Union[np.random.RandomState, int]) -> None:
        if type(random_state) == int:
            # TODO: Remove legacy "seed" style random state
            # np.random.seed(random_state)
            self._random_state = np.random.RandomState(random_state)
        else:
            self._random_state = random_state  # type: ignore
        _log.info(f"Search random state set to {self._random_state}")

    @abstractmethod
    def _next_runs(self, *args, **kwargs) -> Sequence[Optional[SweepRun]]:
        pass

    def next_runs(
        self,
        runs: List[SweepRun],
        *args,
        n: int = 1,
        **kwargs,
    ) -> Sequence[Optional[SweepRun]]:
        return self._next_runs(runs, *args, n=n, **kwargs)
