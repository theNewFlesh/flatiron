from flatiron.core.types import Getter  # noqa F401

from torch.nn import Module  # noqa: F401

import flatiron.torch.tools as fi_torchtools
# ------------------------------------------------------------------------------


def get(config, model):
    # type: (Getter, Module) -> Module
    '''
    Get function from this module.

    Args:
        config (dict): Optimizer config.
        model (Module): Torch model.

    Returns:
        function: Module function.
    '''
    config['params'] = model.parameters()
    return fi_torchtools.get(config, __name__, 'torch.optim')
