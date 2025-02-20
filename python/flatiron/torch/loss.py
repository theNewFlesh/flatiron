from typing import Any  # noqa F401

from copy import deepcopy
from torch.nn import Module  # noqa: F401

import torch.nn.modules.loss as torchloss

import flatiron.core.tools as fict
# ------------------------------------------------------------------------------


def get(config):
    # type: (dict[str, Any]) -> Module
    '''
    Get function from this module.

    Args:
        config (dict): Loss config.

    Returns:
        function: Module function.
    '''
    config = deepcopy(config)
    name = config.pop('name')
    try:
        return fict.get_module_class(name, __name__)
    except NotImplementedError:
        return getattr(torchloss, name)(**config)
