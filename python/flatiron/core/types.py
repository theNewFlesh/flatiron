from typing import Any, Optional, Union
from pathlib import Path

from tensorflow import keras  # noqa F401
from keras import models as tfmodels
import numpy as np
import torch.nn
# ------------------------------------------------------------------------------

AnyModel = Union[tfmodels.Model, torch.nn.Module]
Getter = dict[str, Any]
Compiled = dict[str, Any]
Filepath = Union[str, Path]

# optional
OptArray = Optional[np.ndarray]
OptBool = Optional[bool]
OptFloat = Optional[float]
OptInt = Optional[int]
OptLabels = Optional[Union[int, str, list[int], list[str]]]
OptStr = Optional[str]

# float
Floats = Union[float, list[float]]
OptFloats = Optional[Union[float, list[float]]]
OptListFloat = Optional[list[float]]
OptPairFloat = Optional[Union[float, tuple[float, float]]]
PairFloat = tuple[float, float]

# int
Ints = Union[int, list[int]]
OptInts = Optional[Union[int, tuple[int, ...]]]
PairInt = tuple[int, int]
