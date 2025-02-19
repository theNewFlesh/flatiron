from copy import deepcopy
from typing import Any, Callable  # noqa F401

from tensorflow import keras  # noqa: F401
from keras import optimizers as tfoptim

import flatiron.core.tools as fict
# ------------------------------------------------------------------------------


def get(config):
    # type: (dict) -> Callable[[Any], Any]
    '''
    Get function from this module.

    Args:
        name (dict): Optimizer config.

    Returns:
        function: Module function.
    '''
    kwargs = deepcopy(config)
    name = kwargs.pop('name')
    try:
        return fict.get_module_function(name, __name__)
    except NotImplementedError:
        return tfoptim.get(dict(class_name=name, config=kwargs))
