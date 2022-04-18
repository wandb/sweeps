""" Custom types for sweeps, mainly for type-checking. """

from typing import Any, List, Tuple, Union

import numpy as np


floating = Union[float, np.floating]
integer = Union[int, np.integer]
ArrayLike = Union[List, Tuple, np.ndarray]
