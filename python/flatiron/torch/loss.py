from torch.nn import Module  # noqa: F401

import torch.nn.modules.loss as torchloss

import flatiron.core.tools as fict
# ------------------------------------------------------------------------------


def get(name):
    # type: (str) -> Module
    '''
    Get function from this module.

    Args:
        name (str): Function name.

    Returns:
        function: Module function.
    '''
    try:
        return fict.get_module_class(name, __name__)
    except NotImplementedError:
        return getattr(torchloss, name)
