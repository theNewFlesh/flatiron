from typing import Any, Optional, Union
from pathlib import Path

from tensorflow import keras  # noqa F401
from keras import models as tfmodels
import numpy as np
import torch.nn
# ------------------------------------------------------------------------------

AnyModel = Union[tfmodels.Model, torch.nn.Module]
Compiled = dict[str, Any]
Filepath = Union[str, Path]
OptArray = Optional[np.ndarray]
OptFloat = Optional[int]
OptInt = Optional[int]
OptLabels = Optional[Union[int, str, list[int], list[str]]]
OptStr = Optional[str]
