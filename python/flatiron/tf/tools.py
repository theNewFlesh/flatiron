from typing import Any, Optional  # noqa F401
from flatiron.core.dataset import Dataset  # noqa F401
from flatiron.core.types import Compiled, Filepath, Getter  # noqa: F401

from copy import deepcopy
import math

from tensorflow import keras  # noqa F401
from keras import callbacks as tfcallbacks
import tensorflow as tf

import flatiron.core.tools as fict
import flatiron.tf.loss as fi_tfloss
import flatiron.tf.metric as fi_tfmetric
import flatiron.tf.optimizer as fi_tfoptim

Callbacks = dict[str, tfcallbacks.TensorBoard | tfcallbacks.ModelCheckpoint]
# ------------------------------------------------------------------------------


def get(config, module, fallback_module):
    # type: (Getter, str, Any) -> Any
    '''
    Given a config and set of modules return an instance or function.

    Args:
        config (dict): Instance config.
        module (str): Always __name__.
        fallback_module (Any): Fallback module, either a tf or torch module.

    Raises:
        AssertionError: If config is not a dict.

    Returns:
        object: Instance or function.
    '''
    assert isinstance(config, dict), f'Config must be a dict. Given value: {config}.'

    config = deepcopy(config)
    name = config.pop('name')
    try:
        return fict.get_module_function(name, module)
    except NotImplementedError:
        return fallback_module.get(dict(class_name=name, config=config))


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
    # type: (Any, Getter, Getter, list[Getter], str, Getter) -> Getter
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
        optimizer=fi_tfoptim.get(optimizer),
        loss=fi_tfloss.get(loss),
        metrics=[fi_tfmetric.get(m) for m in metrics],
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
