from typing import Any, Callable, Optional, Union  # noqa F401
from http.client import HTTPResponse  # noqa F401
from lunchbox.stopwatch import StopWatch  # noqa F401
from flatiron.core.types import Filepath, OptInt, OptFloat  # noqa F401
import pandas as pd  # noqa F401

from copy import deepcopy
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


def get_tensorboard_project(project, root='/mnt/storage', timezone='UTC'):
    # type: (Filepath, Filepath, str) -> dict[str, str]
    '''
    Creates directory structure for Tensorboard project.

    Args:
        project (str): Name of project.
        root (str or Path): Tensorboard parent directory. Default: /mnt/storage
        timezone (str, optional): Timezone. Default: UTC.

    Returns:
        dict: Project details.
    '''
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
    target = f'p-{project}_{timestamp}' + '_e-{epoch:03d}.keras'
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


def resolve_kwargs(prefix, kwargs, return_keys='both'):
    # type: (str, dict, str) -> dict
    '''
    Filter keyword arguments base on prefix and return them minus the prefix.

    Args:
        prefix (str): Prefix name.
        kwargs (dict): Kwargs dict.
        return_keys (str, optional): Which kind of keys to return.
            Options: [prefix, non-prefix, both]. Default: both.

    Returns:
        dict: Resolved kwargs.
    '''
    kwargs = deepcopy(kwargs)

    if prefix == 'tensorflow':
        prefix = 'tf'

    legal = ['tf', 'torch', 'sgd', 'adam']
    assert prefix in legal, f'Illegal prefix: {prefix}. Legal prefixes: {legal}.'

    # non prefix
    regex = '|'.join(legal)
    regex = f'^({regex})_'
    output = dict(filter(lambda x: not re.search(regex, x[0]), kwargs.items()))
    if return_keys == 'non-prefix':
        return output

    # prefix
    extra = dict(filter(lambda x: x[0].startswith(prefix), kwargs.items()))
    extra = {re.sub(regex, '', k): v for k, v in extra.items()}
    if return_keys == 'prefix':
        return extra

    output.update(extra)
    return output


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
