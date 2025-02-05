from typing import Any, Callable  # noqa F401

from tensorflow import keras  # noqa: F401
from keras import optimizers as tfo

import flatiron.core.tools as fict
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
    try:
        return fict.get_module_function(name, __name__)
    except NotImplementedError:
        return tfo.get(name)
