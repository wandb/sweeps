from typing import Optional, Sequence

from ..params import make_run_config_from_params
from ..run import SweepRun
from .abstract import AbstractSearch


class RandomSearch(AbstractSearch):
    """Suggest runs with Hyperparameters sampled randomly from specified distributions."""

    def _next_runs(
        self,
        *args,
        n: int = 1,
        **kwargs,
    ) -> Sequence[Optional[SweepRun]]:  # type: ignore
        retval = []
        for _ in range(n):
            for param in self.params:
                param.value = param.sample()
            run = SweepRun(config=make_run_config_from_params(self.params))
            retval.append(run)
        return retval
