from typing import Any, Callable  # noqa F401

from tensorflow import keras  # noqa F401
from keras import optimizers as tfoptim

import flatiron.core.tools as fict
# ------------------------------------------------------------------------------


def get(params):
    # type: (dict) -> Callable[[Any], Any]
    '''
    Get function from this module.

    Args:
        params (dict): Optimizer params.

    Returns:
        function: Module function.
    '''
    try:
        return fict.get_module_function(params['class_name'], __name__)
    except NotImplementedError:
        return tfoptim.get(params)
