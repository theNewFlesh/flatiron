from typing import Any, Optional  # noqa F401
from flatiron.core.dataset import Dataset  # noqa F401
from flatiron.core.types import Callbacks, Compiled, Filepath  # noqa: F401

import math

from tensorflow import keras  # noqa F401
from keras import callbacks as tfcallbacks
import tensorflow as tf

import flatiron
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
        dict: dict with Tensorboard and ModelCheckpoint callbacks.
    '''
    fict.enforce_callbacks(log_directory, checkpoint_pattern)
    return dict(
        tensorboard=tfcallbacks.TensorBoard(
            log_dir=log_directory, histogram_freq=1, update_freq=1
        ),
        checkpoint=tfcallbacks.ModelCheckpoint(checkpoint_pattern, **checkpoint_params),
    )


def pre_build(device):
    # type: (str) -> None
    '''
    Sets hardware device.

    Args:
        device (str): Hardware device.
    '''
    if device == 'cpu':
        tf.config.set_visible_devices([], 'GPU')


def compile(model, optimizer, loss, metrics, device, kwargs={}):
    # type: (Any, dict[str, Any], str, list[str], str, dict[str, Any]) -> dict[str, Any]
    '''
    Call `modile.compile` on given model with kwargs.

    Args:
        model (Any): Model to be compiled.
        optimizer (dict): Optimizer settings.
        loss (str): Loss to be compiled.
        metrics (list[str]): Metrics function to be compiled.
        device (str): Hardware device to compile to.
        kwargs: Other params to be passed to `model.compile`.

    Returns:
        dict: Dict of compiled objects.
    '''
    model.compile(
        optimizer=flatiron.tf.optimizer.get(optimizer),
        loss=flatiron.tf.loss.get(loss),
        metrics=[flatiron.tf.metric.get(m) for m in metrics],
        **kwargs,
    )
    return dict(model=model)


def train(
    compiled,       # type: Compiled
    callbacks,       # type: Callbacks
    train_data,      # type: Dataset
    test_data,       # type: Optional[Dataset]
    batch_size=32,   # type: int
    **kwargs,
):
    # type: (...) -> None
    '''
    Train TensorFlow model.

    Args:
        compiled (dict): Compiled objects.
        callbacks (dict): Dict of callbacks.
        train_data (Dataset): Training dataset.
        test_data (Dataset): Test dataset.
        batch_size (int, optional): Batch size. Default: 32.
        **kwargs: Other params to be passed to `model.fit`.
    '''
    model = compiled['model']
    x_train, y_train = train_data.xy_split()
    steps = math.ceil(x_train.shape[0] / batch_size)

    val = None
    if test_data is not None:
        val = test_data.xy_split()

    model.fit(
        x=x_train,
        y=y_train,
        callbacks=list(callbacks.values()),
        validation_data=val,
        steps_per_epoch=steps,
        **kwargs,
    )
