from tensorflow import keras  # noqa F401
from keras import models as tfmodels  # noqa F401
from flatiron.core.types import Callbacks, Filepath, OptDataset  # noqa: F401
from flatiron.core.dataset import Dataset  # noqa F401

import math

from keras import callbacks as tfcallbacks

import flatiron.core.tools as fict
# ------------------------------------------------------------------------------


def get_callbacks(log_directory, checkpoint_pattern, checkpoint_params={}):
    # type: (Filepath, str, dict) -> Callbacks
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
    callbacks,       # type: Callbacks
    train_data,      # type: Dataset
    test_data,       # type: OptDataset
    batch_size=32,   # type: int
    **kwargs,
):
    # type: (...) -> None
    '''
    Train TensorFlow model.

    Args:
        model (tfmodels.Model): TensorFlow model.
        callbacks (list): List of callbacks.
        train_data (Dataset): Training dataset.
        test_data (Dataset): Test dataset.
        batch_size (int, optional): Batch size. Default: 32.
        **kwargs: Other params to be passed to `model.fit`.
    '''
    x_train, y_train = train_data.load().xy_split()
    steps = math.ceil(x_train.shape[0] / batch_size)

    val = None
    if test_data is not None:
        val = test_data.load().xy_split()

    model.fit(
        x=x_train,
        y=y_train,
        callbacks=callbacks,
        validation_data=val,
        steps_per_epoch=steps,
        **kwargs,
    )
