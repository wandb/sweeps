from abc import ABC, abstractmethod
import logging
from typing import List, Optional, Sequence, Union

import numpy as np

from ..config.cfg import SweepConfig
from ..params import HyperParameterSet
from ..run import SweepRun

_log = logging.getLogger(__name__)


class AbstractSearch(ABC):
    """Abstract base class for Search Algotithms."""

    def __init__(
        self,
        sweep_config: Union[dict, SweepConfig],
        runs: List[SweepRun],
        validate: bool = False,
        random_state: Union[np.random.RandomState, int] = 42,
    ) -> None:
        if validate:
            _log.info("Ensuring sweep config is properly formatted")
            sweep_config = SweepConfig(sweep_config)
        else:
            SweepConfig.quick_validate(sweep_config)
        self.params: HyperParameterSet = HyperParameterSet.from_config(
            sweep_config["parameters"]
        )
        self.random_state = random_state

    @property
    def random_state(self) -> np.random.RandomState:
        return self._random_state

    @random_state.setter
    def random_state(self, random_state: Union[np.random.RandomState, int]) -> None:
        if type(random_state) == int:
            self._random_state = np.random.RandomState(random_state)
        else:
            self._random_state = random_state
        
    @abstractmethod
    def _next_runs(self, *args, **kwargs) -> Sequence[Optional[SweepRun]]:
        pass

    def next_runs(
        self,
        *args,
        **kwargs,
    ) -> Sequence[Optional[SweepRun]]:
        return self._next_runs(*args, **kwargs)
