from abc import ABC, abstractmethod
import logging
from typing import List, Optional, Sequence, Union

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
    ) -> None:
        if validate:
            _log.info("Ensuring sweep config is properly formatted")
            sweep_config = SweepConfig(sweep_config)
        else:
            SweepConfig.quick_validate(sweep_config)
        self.params: HyperParameterSet = HyperParameterSet.from_config(
            sweep_config["parameters"]
        )

    @abstractmethod
    def _next_run(self) -> Optional[SweepRun]:
        pass

    @abstractmethod
    def _next_runs(self) -> Sequence[Optional[SweepRun]]:
        pass

    def next_run(
        self,
        *args,
        **kwargs,
    ) -> Optional[SweepRun]:
        self._next_run(*args, **kwargs)

    def next_runs(
        self,
        *args,
        **kwargs,
    ) -> Sequence[Optional[SweepRun]]:
        self._next_runs(*args, **kwargs)
