from typing import Callable, Optional, Union
from pathlib import Path

from tensorflow import keras  # noqa F401
from keras import callbacks as tfcb
from keras import models as tfmodels
import numpy as np
import torch.nn
# ------------------------------------------------------------------------------

AnyModel = Union[tfmodels.Model, torch.nn.Module]
Callbacks = list[Union[Callable, tfcb.TensorBoard, tfcb.ModelCheckpoint]]
Filepath = Union[str, Path]
OptArray = Optional[np.NDArray]
