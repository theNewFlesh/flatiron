from typing import Any, Callable, Union  # noqa F401
import numpy

import tensorflow as tf
import tensorflow.keras.backend as tfb

import flatiron.core.tools as fict

Arraylike = Union[numpy.ndarray, tf.Tensor]
# ------------------------------------------------------------------------------


def get(name):
    # type: (str) -> Callable[[Any], Any]
    '''
    Get function from this module.

    Args:
        name (str): Function name.

    Returns:
        function: Module function.
    '''
    return fict.get_module_function(name, __name__)
# ------------------------------------------------------------------------------


def identity(data):
    return data
