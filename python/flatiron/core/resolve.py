from typing import Type  # noqa F401
from flatiron.core.types import OptStr, Getter  # noqa F401
from pydantic import BaseModel  # noqa F401

from copy import deepcopy

import flatiron.core.config as cfg
import flatiron.core.tools as fict
# ------------------------------------------------------------------------------


def resolve_config(config, model):
    # type: (dict, Type[BaseModel]) -> dict
    '''
    Resolves given Pipeline config.
    Config fields include:

        * framework
        * model
        * dataset
        * optimizer
        * loss
        * metrics
        * callbacks
        * train
        * logger

    Args:
        config (dict): Config dict.
        model (BaseModel): Model config class.

    Returns:
        dict: Resolved config.
    '''
    config = deepcopy(config)
    config = _resolve_model(config, model)
    config = _resolve_pipeline(config)
    config = _resolve_field(config, 'framework')
    config = _resolve_field(config, 'optimizer')
    config = _resolve_field(config, 'loss')
    config = _resolve_field(config, 'metrics')
    return config


def _resolve_model(config, model):
    # type: (dict, Type[BaseModel]) -> dict
    '''
    Resolve and validate given model config.

    Args:
        config (dict): Model config.
        model (BaseModel): Model config class.

    Returns:
        dict: Validated model config.
    '''
    config['model'] = model \
        .model_validate(config['model'], strict=True) \
        .model_dump()
    return config


def _resolve_pipeline(config):
    # type: (dict) -> dict
    '''
    Resolve and validate given pipeline config.

    Args:
        config (dict): Pipeline config.

    Returns:
        dict: Validated pipeline config.
    '''
    model = config.pop('model', {})
    config = cfg.PipelineConfig \
        .model_validate(config, strict=True) \
        .model_dump()
    config['model'] = model
    return config


def _resolve_field(config, field):
    # type: (dict, str) -> dict
    '''
    Resolve and validate given pipeline config field.

    Args:
        config (dict): Pipeline config.
        field (str): Config field name.

    Returns:
        dict: Updated pipeline config.
    '''
    prefix = config['framework']['name']
    if prefix == 'tensorflow':
        prefix = 'TF'
    else:
        prefix = prefix.capitalize()

    pkg = f'flatiron.{prefix.lower()}'
    lut = dict(
        framework=(f'{prefix}Framework', False, f'{pkg}.config', None              ),  # noqa E202
        optimizer=(f'{prefix}Opt',       True,  f'{pkg}.config', f'{pkg}.optimizer'),  # noqa E202
        loss     =(f'{prefix}Loss',      True,  f'{pkg}.config', f'{pkg}.loss'     ),  # noqa E202
        metrics  =(f'{prefix}Metric',    True,  f'{pkg}.config', f'{pkg}.metric'   ),  # noqa E202
    )
    keys = ['class_prefix', 'prepend', 'config_module', 'other_module']
    kwargs = dict(zip(keys, lut[field]))  # type: Getter

    subconfig = config[field]
    if isinstance(subconfig, list):
        config[field] = [_resolve_subconfig(x, **kwargs) for x in subconfig]
    else:
        config[field] = _resolve_subconfig(subconfig, **kwargs)

    return config


def _resolve_subconfig(
    subconfig, class_prefix, prepend, config_module, other_module
):
    # type: (dict, str, bool, str, OptStr) -> dict
    '''
    For use in _resolve_field. Resolves and validates given subconfig.
    If class is not custom definition found in config module or
    other module, a standard definition will be resolved from config module.
    class prefix and prepend are used to modify the config name field in
    order to make it a valid class name.

    Args:
        subconfig (dict): Subconfig.
        class_prefix (str): Class prefix.
        prepend (bool): Prepend class prefix.
        config_module (str): Module name.
        other_module (str): Module name.

    Returns:
        dict: Validated subconfig.
    '''
    if config_module is not None:
        if fict.is_custom_definition(subconfig, config_module):
            return subconfig
    if other_module is not None:
        if fict.is_custom_definition(subconfig, other_module):
            return subconfig

    name = subconfig['name']
    output = deepcopy(subconfig)
    output['name'] = class_prefix
    if prepend:
        output['name'] += name

    output = fict.resolve_module_config(output, config_module)
    output['name'] = name
    return output
