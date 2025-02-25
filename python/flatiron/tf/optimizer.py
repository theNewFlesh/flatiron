from typing import Any  # noqa F401

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
    return fi_tftools.get(config, __name__, 'keras.api.optimizers')
