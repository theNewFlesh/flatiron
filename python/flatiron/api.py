'''
This moudle is meant as a convenience for programmers who want a clean namespace
to sift through.
'''

import argparse as __argparse
import inspect as __inspect
import re as __re

from flatiron.core.config import PipelineConfig  # noqa F401
from flatiron.core.dataset import Dataset  # noqa F401
from flatiron.core.multidataset import MultiDataset  # noqa F401
from flatiron.core.logging import SlackLogger  # noqa F401
from flatiron.core.pipeline import PipelineBase  # noqa F401
# ------------------------------------------------------------------------------


def __create_namespace(module, mode='funcs+classes'):
    # type: (object, str) -> __argparse.Namespace
    '''
    Creates a clean namespace from a module.
    Only grabs public functions from the module.

    Args:
        module (object): module.
        mode (str, optional): Restric object to public functions and/or classes.
            Default: 'funcs+classes'.

    Returns:
        argparse.Namespace: Clean namespace.
    '''
    params = {}
    for key, val in __inspect.getmembers(module):
        func = __inspect.isfunction(val)
        clazz = __inspect.isclass(val)

        legal = [func, clazz]
        if mode == 'funcs':
            legal = [func]
        elif mode == 'classes':
            legal = [clazz]

        if any(legal) and __re.search('^[a-zA-Z]', key):
            params[key] = val
    return __argparse.Namespace(**params)


try:
    import flatiron.tf.config as __fi_tfconfig  # noqa F401
    import flatiron.tf.loss as __fi_tfloss  # noqa F401
    import flatiron.tf.metric as __fi_tfmetric  # noqa F401
    import flatiron.tf.optimizer as __fi_tfopt  # noqa F401
    import flatiron.tf.tools as __fi_tftools  # noqa F401
    import flatiron.tf.models as __fi_tfmodels  # noqa F401

    tf = __argparse.Namespace(
        config=__create_namespace(__fi_tfconfig, 'classes'),
        loss=__create_namespace(__fi_tfloss),
        metric=__create_namespace(__fi_tfmetric),
        models=__fi_tfmodels,
        optimizer=__create_namespace(__fi_tfopt),
        tools=__create_namespace(__fi_tftools, 'funcs'),
    )
except ImportError:
    pass

try:
    import flatiron.torch.config as __fi_torchconfig  # noqa F401
    import flatiron.torch.loss as __fi_torchloss  # noqa F401
    import flatiron.torch.metric as __fi_torchmetric  # noqa F401
    import flatiron.torch.optimizer as __fi_torchopt  # noqa F401
    import flatiron.torch.tools as __fi_torchtools  # noqa F401
    import flatiron.torch.models as __fi_torchmodels  # noqa F401

    torch = __argparse.Namespace(
        config=__create_namespace(__fi_torchconfig, 'classes'),
        loss=__create_namespace(__fi_torchloss),
        metric=__create_namespace(__fi_torchmetric),
        models=__fi_torchmodels,
        optimizer=__create_namespace(__fi_torchopt),
        tools=__create_namespace(__fi_torchtools, 'funcs'),
    )
except ImportError:
    pass
