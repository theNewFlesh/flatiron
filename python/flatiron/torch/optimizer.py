from torch.nn import Module  # noqa: F401

from copy import deepcopy

import torch.optim as torchoptim

import flatiron.core.tools as fict
# ------------------------------------------------------------------------------


def get(config, model):
    # type: (dict, Module) -> Module
    '''
    Get function from this module.

    Args:
        config (dict): Optimizer config.
        model (Module): Torch model.

    Returns:
        function: Module function.
    '''
    kwargs = deepcopy(config)
    name = kwargs.pop('name').capitalize()
    if name == 'Sgd':
        name = 'SGD'

    lut = dict(
        learning_rate='lr',
        epsilon='eps',
    )
    for old_key, new_key in lut.items():
        if old_key in config:
            kwargs[new_key] = kwargs.pop(old_key)

    if 'adam_beta_1' in kwargs or 'adam_beta_2' in kwargs:
        a1 = kwargs.pop('adam_beta_1', 0.9)
        a2 = kwargs.pop('adam_beta_2', 0.999)
        kwargs['betas'] = [a1, a2]

    try:
        return fict.get_module_class(name, __name__)
    except NotImplementedError:
        return getattr(torchoptim, name)(model.parameters(), **kwargs)
