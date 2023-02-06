from typing import Any, Callable # noqa F401

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
    return fict.get_module_function(name, __name__)
# ------------------------------------------------------------------------------
