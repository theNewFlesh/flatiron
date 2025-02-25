from flatiron.core.types import Getter  # noqa F401

from torch.nn import Module  # noqa: F401
import torchmetrics  # noqa F401

import flatiron.torch.tools as fi_torchtools
# ------------------------------------------------------------------------------


def get(config):
    # type: (Getter) -> Module
    '''
    Get function from this module.

    Args:
        config (dict): Loss config.

    Returns:
        function: Module function.
    '''
    return fi_torchtools.get(config, __name__, 'torchmetrics')
