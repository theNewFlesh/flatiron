from typing import Optional, Union
from pathlib import Path

from keras import models as tfmodels
import numpy as np
import torch.nn
# ------------------------------------------------------------------------------

Model = Union[tfmodels.Model, torch.nn.Module]
Filepath = Union[str, Path]
OptArray = Optional[np.ndarray]
