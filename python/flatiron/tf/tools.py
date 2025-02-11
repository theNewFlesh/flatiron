from flatiron.core.types import Filepath, OptArray  # noqa: F401
from keras import models as tfmodels  # noqa F401
from tensorflow import keras  # noqa F401
import numpy as np  # noqa F401

import math

from keras import callbacks as tfcallbacks

import flatiron.core.tools as fict
# ------------------------------------------------------------------------------


def get_callbacks(log_directory, checkpoint_pattern, checkpoint_params={}):
    # type: (Filepath, str, dict) -> list
    '''
    Create a list of callbacks for Tensoflow model.

    Args:
        log_directory (str or Path): Tensorboard project log directory.
        checkpoint_pattern (str): Filepath pattern for checkpoint callback.
        checkpoint_params (dict, optional): Params to be passed to checkpoint
            callback. Default: {}.

    Raises:
        EnforceError: If log directory does not exist.
        EnforeError: If checkpoint pattern does not contain '{epoch}'.

    Returns:
        list: Tensorboard and ModelCheckpoint callbacks.
    '''
    fict.enforce_callbacks(log_directory, checkpoint_pattern)
    callbacks = [
        tfcallbacks.TensorBoard(log_dir=log_directory, histogram_freq=1),
        tfcallbacks.ModelCheckpoint(checkpoint_pattern, **checkpoint_params),
    ]
    return callbacks


def train(
    model,           # type: tfmodels.Model
    x_train,         # type: np.ndarray
    y_train,         # type: np.ndarray
    x_test=None,     # type: OptArray
    y_test=None,     # type: OptArray
    callbacks=None,  # type: list
    batch_size=32,   # type: int
    **kwargs,
):
    # type: (...) -> None
    '''
    Train TensorFlow model.

    Args:
        model (tfmodels.Model): TensorFlow model.
        x_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        x_test (np.ndarray, optional): Test data. Default: None.
        y_test (np.ndarray, optional): Test labels. Default: None.
        callbacks (list, optional): List of callbacks. Default: None.
        batch_size (int, optional): Batch size. Default: 32.
        **kwargs: Other params to be passed to `model.fit`.
    '''
    n = x_train.shape[0]  # type: ignore
    val = None
    if x_test is not None and y_test is not None:
        val = (x_test, y_test)
    model.fit(
        x=x_train,
        y=y_train,
        callbacks=callbacks,
        validation_data=val,
        steps_per_epoch=math.ceil(n / batch_size),
        **kwargs,
    )
