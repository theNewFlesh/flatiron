from typing import Any, Callable, Optional, Union
from pathlib import Path

from tensorflow import keras  # noqa F401
from keras import callbacks as tfcb
from keras import models as tfmodels
import numpy as np
# ------------------------------------------------------------------------------

AnyModel = Union[tfmodels.Model, Any]
Callbacks = dict[str, Union[Callable, tfcb.TensorBoard, tfcb.ModelCheckpoint]]
Compiled = dict[str, Any]
Filepath = Union[str, Path]
OptArray = Optional[np.ndarray]
OptFloat = Optional[int]
OptInt = Optional[int]
OptLabels = Optional[Union[int, str, list[int], list[str]]]
OptStr = Optional[str]
