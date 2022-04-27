from typing import Union, List, Sequence, Optional

from .abstract import AbstractSearch
from ..config.cfg import SweepConfig
from ..run import SweepRun
from ..params import HyperParameterSet, make_run_config_from_params


class RandomSearch(AbstractSearch):
    """Suggest runs with Hyperparameters sampled randomly from specified distributions."""

    def _next_runs(
        self,
        *args,
        n: int = 1,
        **kwargs,
    ) -> Sequence[Optional[SweepRun]]:
        retval = []
        for _ in range(n):
            for param in self.params:
                param.value = param.sample()
            run = SweepRun(config=make_run_config_from_params(self.params))
            retval.append(run)
        return retval
