""" Custom types for sweeps, mainly for type-checking. """

from typing import Any, List, Tuple, Union

import numpy as np


floating = Union[float, np.floating]
integer = Union[int, np.integer]
# TODO: Remove Any from here
#   mypy is very picky when it comes to supported operand types
ArrayLike = Union[Any, List, Tuple, np.ndarray]
