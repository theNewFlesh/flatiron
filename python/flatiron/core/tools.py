from typing import Any, Callable, Optional, Union  # noqa F401
from http.client import HTTPResponse  # noqa F401
from lunchbox.stopwatch import StopWatch  # noqa F401
from flatiron.core.types import Filepath, OptInt, OptFloat, Getter  # noqa F401
import pandas as pd  # noqa F401

from datetime import datetime
from pathlib import Path
import inspect
import os
import random
import re
import sys

from lunchbox.enforce import Enforce
import lunchbox.tools as lbt
import pytz
import yaml
# ------------------------------------------------------------------------------


def get_tensorboard_project(
    project, root='/mnt/storage', timezone='UTC', extension='keras'
):
    # type: (Filepath, Filepath, str, str) -> dict[str, str]
    '''
    Creates directory structure for Tensorboard project.

    Args:
        project (str): Name of project.
        root (str or Path): Tensorboard parent directory. Default: /mnt/storage
        timezone (str, optional): Timezone. Default: UTC.
        extension (str, optional): File extension.
            Options: [keras, safetensors]. Default: keras.

    Raises:
        EnforceError: If extension is not keras, pth or safetensors.

    Returns:
        dict: Project details.
    '''
    msg = 'Extension must be keras or safetensors. Given value: {a}.'
    Enforce(extension, 'in', ['keras', 'pth', 'safetensors'], message=msg)
    # --------------------------------------------------------------------------

    # create timestamp
    timestamp = datetime \
        .now(tz=pytz.timezone(timezone)) \
        .strftime('d-%Y-%m-%d_t-%H-%M-%S')

    # create directories
    root_dir = Path(root, project, 'tensorboard').as_posix()
    log_dir = Path(root_dir, timestamp).as_posix()
    model_dir = Path(log_dir, 'models').as_posix()
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # checkpoint pattern
    epoch = '{epoch:03d}'
    target = f'p-{project}_{timestamp}_e-{epoch}.{extension}'
    target = Path(model_dir, target).as_posix()

    output = dict(
        root_dir=root_dir,
        log_dir=log_dir,
        model_dir=model_dir,
        checkpoint_pattern=target,
    )
    return output


def enforce_callbacks(log_directory, checkpoint_pattern):
    # type: (Filepath, str) -> None
    '''
    Enforces callback parameters.

    Args:
        log_directory (str or Path): Tensorboard project log directory.
        checkpoint_pattern (str): Filepath pattern for checkpoint callback.

    Raises:
        EnforceError: If log directory does not exist.
        EnforeError: If checkpoint pattern does not contain '{epoch}'.
    '''
    log_dir = Path(log_directory)
    msg = f'Log directory: {log_dir} does not exist.'
    Enforce(log_dir.is_dir(), '==', True, message=msg)

    match = re.search(r'\{epoch.*?\}', checkpoint_pattern)
    msg = "Checkpoint pattern must contain '{epoch}'. "
    msg += f'Given value: {checkpoint_pattern}'
    msg = msg.replace('{', '{{').replace('}', '}}')
    Enforce(match, '!=', None, message=msg)


def enforce_getter(value):
    # type: (Getter) -> None
    '''
    Enforces value is a dict with a name key.

    Args:
        value (dict): Dict..

    Raises:
        EnforceError: Is not a dict with a name key.
    '''
    msg = 'Value must be a dict with a name key.'
    Enforce(value, 'instance of', dict, message=msg)
    Enforce('name', 'in', value, message=msg)


# MISC--------------------------------------------------------------------------
def pad_layer_name(name, length=18):
    # type: (str, int) -> str
    '''
    Pads underscores in a given layer name to make the string achieve a given
    length.

    Args:
        name (str): Layer name to be padded.
        length (int): Length of output string. Default: 18.

    Returns:
        str: Padded layer name.
    '''
    if length == 0:
        return name

    if '_' not in name:
        name += '_'
    delta = length - len(re.sub('_', '', name))
    return re.sub('_+', '_' * delta, name)


def unindent(text, spaces=4):
    # type: (str, int) -> str
    '''
    Unindents given block of text according to given number of spaces.

    Args:
        text (str): Text block to unindent.
        spaces (int, optional): Number of spaces to remove. Default: 4.

    Returns:
        str: Unindented text.
    '''
    output = text.split('\n')  # type: Any
    regex = re.compile('^ {' + str(spaces) + '}')
    output = [regex.sub('', x) for x in output]
    output = '\n'.join(output)
    return output


def slack_it(
    title,  # type: str
    channel,  # type: str
    url,  # type: str
    config=None,  # type: Optional[dict]
    stopwatch=None,  # type: Optional[StopWatch]
    timezone='UTC',  # type: str
    suppress=False,  # type: bool
):
    # type: (...) -> Union[str, HTTPResponse]
    '''
    Compose a message from given arguments and post it to slack.

    Args:
        title (str): Post title.
        channel (str): Slack channel.
        url (str): Slack URL.
        config (dict, optional): Parameter dict. Default: None.
        stopwatch (StopWatch, optional): StopWatch instance. Default: None.
        timezone (str, optional): Timezone. Default: UTC.
        suppress (bool, optional): Return message, rather than post it to Slack.
            Default: False.

    Returns:
        HTTPResponse: Slack response.
    '''
    now = datetime.now(tz=pytz.timezone(timezone)).isoformat()
    cfg = config or {}
    delta = 'none'
    hdelta = 'none'
    if stopwatch is not None:
        hdelta = stopwatch.human_readable_delta
        delta = str(stopwatch.delta)

    config_ = yaml.safe_dump(cfg, indent=4)
    message = f'''
        {title.upper()}

        RUN TIME:
        ```{hdelta} ({delta})```
        POST TIME:
        ```{now}```
        CONFIG:
        ```{config_}```
    '''[1:-1]
    message = unindent(message, spaces=8)

    if suppress:
        return message
    return lbt.post_to_slack(url, channel, message)  # pragma: no cover


def resolve_kwargs(kwargs, engine, optimizer, return_type='both'):
    # type: (dict, str, str, str) -> dict
    '''
    Filter keyword arguments base on prefix and return them minus the prefix.

    Args:
        kwargs (dict): Kwargs dict.
        engine (str): Deep learning framework.
        optimizer (str): Optimizer name.
        return_type (str, optional): Which kind of keys to return.
            Options: [prefixed, unprefixed, both]. Default: both.

    Returns:
        dict: Resolved kwargs.
    '''
    prefixed = {}
    unprefixed = {}
    for key, val in kwargs.items():
        if not re.search('__', key):
            unprefixed[key] = val
            continue

        head, tail = re.split('__', key, maxsplit=1)
        if not re.search(f'{engine}|{optimizer}', head):
            continue

        cond = [
            head.startswith(optimizer),
            head == engine,
            f'{engine}_{optimizer}' == head,
        ]
        if any(cond):
            prefixed[tail] = val

    if return_type == 'prefixed':
        return prefixed
    elif return_type == 'unprefixed':
        return unprefixed
    prefixed.update(unprefixed)
    return prefixed


def train_test_split(data, test_size=0.2, shuffle=True, seed=None, limit=None):
    # type: (pd.DataFrame, float, bool, OptFloat, OptInt) -> tuple[pd.DataFrame, pd.DataFrame]
    '''
    Split DataFrame into train and test DataFrames.

    Args:
        data (pd.DataFrame): DataFrame.
        test_size (float, optional): Test set size as a proportion.
            Default: 0.2.
        shuffle (bool, optional): Randomize data before splitting.
            Default: True.
        seed (int, optional): Seed number. Default: None.
        limit (int, optional): Limit the total length of train and test.
            Default: None.

    Raises:
        EnforceError: If data is not a DataFrame.
        EnforceError: If test_size is not between 0 and 1.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Train and test DataFrames.
    '''
    Enforce(data, 'instance of', pd.DataFrame)
    Enforce(test_size, '>=', 0)
    Enforce(test_size, '<=', 1)
    # --------------------------------------------------------------------------

    index = data.index.tolist()
    if shuffle:
        if seed is not None:
            rand = random.Random()
            rand.seed(seed)
            rand.shuffle(index)
        else:
            random.shuffle(index)

    if limit is not None:
        index = index[:limit]

    k = int(len(index) * (1 - test_size))
    return data.loc[index[:k]].copy(), data.loc[index[k:]].copy()


# MODULE-FUNCS------------------------------------------------------------------
def get_module(name):
    # type: (str) -> Any
    '''
    Get a module from a given name.

    Args:
        name (str): Module name.

    Raises:
        NotImplementedError: If module is not found.

    Returns:
        object: Module.
    '''
    try:
        return sys.modules[name]
    except KeyError:
        raise NotImplementedError(f'Module not found: {name}')


def get_module_function(name, module):
    # type: (str, str) -> Callable[[Any], Any]
    '''
    Get a function from a given module.

    Args:
        name (str): Function name.
        module (str): Module name.

    Raises:
        NotImplementedError: If function is not found in module.

    Returns:
        function: Module function.
    '''
    members = inspect.getmembers(get_module(module))
    funcs = dict(filter(lambda x: inspect.isfunction(x[1]), members))
    if name in funcs:
        return funcs[name]
    raise NotImplementedError(f'Function not found: {name}')


def get_module_class(name, module):
    # type: (str, str) -> Any
    '''
    Get a class from a given module.

    Args:
        name (str): Class name.
        module (str): Module name.

    Raises:
        NotImplementedError: If class is not found in module.

    Returns:
        class: Module class.
    '''
    members = inspect.getmembers(get_module(module))
    classes = dict(filter(lambda x: inspect.isclass(x[1]), members))
    if name in classes:
        return classes[name]
    raise NotImplementedError(f'Class not found: {name}')


def resolve_module_config(config, module):
    # type: (Getter, str) -> Getter
    '''
    Given a config and set of modules return a validated dict.

    Args:
        config (dict): Instance config.
        module (str): Always __name__.

    Raises:
        EnforceError: If config is not a dict with a name key.

    Returns:
        dict: Resolved config dict.
    '''
    enforce_getter(config)
    # --------------------------------------------------------------------------

    model = get_module_class(config['name'], module)
    return model.model_validate(config).model_dump()


def is_custom_definition(config, module):
    # type: (Getter, str) -> bool
    '''
    Determine whether config is of custom defined code.

    Args:
        config (dict): Instance config.
        module (str): Always __name__.

    Raises:
        EnforceError: If config is not a dict with a name key.

    Returns:
        bool: True if config is of custom defined code.
    '''
    enforce_getter(config)
    # --------------------------------------------------------------------------

    try:
        get_module_function(config['name'], module)
        return True
    except NotImplementedError:
        pass

    try:
        get_module_class(config['name'], module)
        return True
    except NotImplementedError:
        pass
    return False
