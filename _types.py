from typing import Union, TYPE_CHECKING, Any

import numpy as np
from distutils.version import LooseVersion


floating = Union[float, np.floating]
integer = Union[int, np.integer]

if TYPE_CHECKING:
    if LooseVersion(np.version.version) < LooseVersion("1.20"):
        ArrayLike: Any
    else:
        ArrayLike = np.typing.ArrayLike
