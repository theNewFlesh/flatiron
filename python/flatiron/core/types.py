from typing import Any, Callable, Optional, Union
from pathlib import Path
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from flatiron.core.dataset import Dataset  # noqa: F401

from tensorflow import keras  # noqa F401
from keras import callbacks as tfcb
from keras import models as tfmodels
import numpy as np
# ------------------------------------------------------------------------------

AnyModel = Union[tfmodels.Model, Any]
Callbacks = list[Union[Callable, tfcb.TensorBoard, tfcb.ModelCheckpoint]]
Filepath = Union[str, Path]
OptArray = Optional[np.ndarray]
OptLabels = Optional[Union[int, str, list[str]]]
OptDataset = Optional[Dataset]
