from typing import Any  # noqa F401

from tensorflow import keras  # noqa: F401
from keras import optimizers as tfoptim

import flatiron.tf.tools as fi_tftools
# ------------------------------------------------------------------------------


def get(config):
    # type: (dict) -> Any
    '''
    Get function from this module.

    Args:
        config (dict): Optimizer config.

    Returns:
        function: Module function.
    '''
    return fi_tftools.get(config, __name__, tfoptim.__name__)
