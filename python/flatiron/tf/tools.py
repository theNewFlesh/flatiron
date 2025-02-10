from flatiron.core.types import Filepath  # noqa: F401

from tensorflow import keras  # noqa: F401
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
