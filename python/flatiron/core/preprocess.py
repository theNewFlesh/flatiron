from typing import Any, Callable, Generator, Tuple  # noqa F401

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


def identity(data, **kwargs):
    # type: (Tuple[Any, Any], Any) -> Generator[Tuple[Any, Any], None, None]
    r'''
    Converts training data to generator function.

    Args:
        data (tuple): X, Y training data.
        \*\*kwargs (optional): Extra keyword args.

    Yields:
        tuple: (x, y) data
    '''
    yield data
