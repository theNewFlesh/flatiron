from typing import Any, Callable, Optional, Union  # noqa F401
from http.client import HTTPResponse  # noqa F401
from lunchbox.stopwatch import StopWatch  # noqa F401

from datetime import datetime
from pathlib import Path
import inspect
import os
import re
import sys

from lunchbox.enforce import Enforce
import lunchbox.tools as lbt
import pytz
import yaml

import tensorflow as tf
import tensorflow.keras.callbacks as tfc

Filepath = Union[str, Path]
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
    target = f'p-{project}_{timestamp}' + '_e-{epoch:03d}'
    target = Path(model_dir, target).as_posix()

    output = dict(
        root_dir=root_dir,
        log_dir=log_dir,
        model_dir=model_dir,
        checkpoint_pattern=target,
    )
    return output


def get_callbacks(log_directory, checkpoint_pattern, checkpoint_params={}):
    # type: (Filepath, str, dict) -> list
    '''
    Create a list of callbacks for Tensoflow model.

    Args:
        log_directory (str or Path): Tensorboard project log directory.
        checkpoint_pattern (str): Filepath pattern for checkpoint callback.
        checkpoint_params (dict, optional): Params to be passed to checkpoint
            callback. Default: {}.

    Raises:
        EnforceError: If log directory does not exist.
        EnforceError: If checkpoint pattern does not contain '{epoch}'.

    Returns:
        list: Tensorboard and ModelCheckpoint callbacks.
    '''
    log_dir = Path(log_directory)
    msg = f'Log directory: {log_dir} does not exist.'
    Enforce(log_dir.is_dir(), '==', True, message=msg)

    match = re.search(r'\{epoch.*?\}', checkpoint_pattern)
    msg = "Checkpoint pattern must contain '{epoch}'. "
    msg += f'Given value: {checkpoint_pattern}'
    msg = msg.replace('{', '{{').replace('}', '}}')
    Enforce(match, '!=', None, message=msg)
    # --------------------------------------------------------------------------

    callbacks = [
        tfc.TensorBoard(log_dir=log_directory, histogram_freq=1),
        tfc.ModelCheckpoint(checkpoint_pattern, **checkpoint_params),
    ]
    return callbacks


def split_tensor(tensor, index, axis=-1, output='left'):
    # type: (tf.Tensor, int, int, str) -> tf.Tensor
    '''
    Splits a given tensor according along the index of a given axis.

    Args:
        tensor (tf.Tensor): Tensor to be split.
        index (int): Index of axis to split tensor at.
        axis (int, optional): Axis to split on. Default: -1.
        output (str, optional): Return left or right tensor. Default: left.

    Raises:
        EnforceError: If tensor is not an instance of tf.Tensor.
        EnforceError: If axis is not an integer within the legal range.
        EnforceError: If index is not an integer within the legal range.
        EnforceError: If output is not 'left' or 'right'.

    Returns:
        tf.Tensor: Left or right tensor.
    '''
    # tensor
    Enforce(tensor, 'instance of', tf.Tensor)
    shape = tensor.shape

    # axis type
    msg = f'Axis must be an integer. Given value: {axis}.'
    Enforce(axis, 'instance of', int, message=msg)

    # axis value
    ndim = len(shape)
    axis_min = -ndim
    axis_max = ndim - 1
    msg = f'Illegal axis: {axis}. Given tensor of shape: {shape} dictates that '
    msg += f'axis must lie within the range: {axis_min} to {axis_max}.'
    Enforce(axis, '>=', axis_min, message=msg)
    Enforce(axis, '<=', axis_max, message=msg)

    # index type
    msg = f'Index must be an integer. Given value: {index}.'
    Enforce(index, 'instance of', int, message=msg)

    # index vaule
    nind = shape[axis]  # type: Any
    ind_max = nind - 1
    ind_min = -ind_max
    msg = f'Illegal index: {index}. Axis {axis} has a size of {nind} which '
    msg += 'dictates that index must lie within the the range: '
    msg += f'{ind_min} to {ind_max} and cannot be 0.'
    Enforce(index, '>=', ind_min, message=msg)
    Enforce(index, '<=', ind_max, message=msg)
    Enforce(index, '!=', 0, message=msg)

    # output
    output = output.lower()
    msg = 'Illegal output given: {a}. Legal outputs: {b}.'
    Enforce(output, 'in', ['left', 'right'], message=msg)
    # --------------------------------------------------------------------------

    full = slice(None)
    slices = [full] * len(tensor.shape)  # type: Any
    if output == 'left':
        slices[axis] = slice(None, index)
    else:
        slices[axis] = slice(index, None)
    return tensor[slices]


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
    config = config or {}
    delta = 'none'
    hdelta = 'none'
    if stopwatch is not None:
        hdelta = stopwatch.human_readable_delta
        delta = str(stopwatch.delta)

    config = yaml.safe_dump(config, indent=4)
    message = f'''
        {title.upper()}

        RUN TIME:
        ```{hdelta} ({delta})```
        POST TIME:
        ```{now}```
        CONFIG:
        ```{config}```
    '''[1:-1]
    message = unindent(message, spaces=8)

    if suppress:
        return message
    return lbt.post_to_slack(url, channel, message)  # pragma: no cover


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
    members = inspect.getmembers(sys.modules[module])
    funcs = dict(filter(lambda x: inspect.isfunction(x[1]), members))
    if name in funcs:
        return funcs[name]
    raise NotImplementedError(f'Function not found: {name}')
