from abc import ABC, abstractmethod
import logging
from typing import List, Optional, Sequence, Union

from ..config.cfg import SweepConfig 
from ..run import SweepRun

_log = logging.getLogger(__name__)

class AbstractSearch(ABC):
    """Abstract base class for Search Algotithms."""

    def next_run(
        sweep_config: Union[dict, SweepConfig],
        runs: List[SweepRun],
        validate: bool = False,
        **kwargs,
    ) -> Optional[SweepRun]:
        pass

    def next_runs(
        sweep_config: Union[dict, SweepConfig],
        runs: List[SweepRun],
        validate: bool = False,
        n: int = 1,
        **kwargs,
    ) -> Sequence[Optional[SweepRun]]:
        pass