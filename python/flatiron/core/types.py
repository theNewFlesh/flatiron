from typing import Callable, Optional, Union
from pathlib import Path

from keras import models as tfmodels
import numpy as np
import torch.nn
# ------------------------------------------------------------------------------

AnyModel = Union[tfmodels.Model, torch.nn.Module]
Callbacks = list[Callable]
Filepath = Union[str, Path]
OptArray = Optional[np.ndarray]
