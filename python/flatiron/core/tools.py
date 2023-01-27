from typing import Any, Optional, Union  # noqa F401
from http.client import HTTPResponse  # noqa F401
from lunchbox.stopwatch import StopWatch  # noqa F401

from datetime import datetime
from pathlib import Path
import json
import os
import re

from lunchbox.enforce import Enforce
import lunchbox.tools as lbt
import pytz
import yaml

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


def get_callbacks(log_directory, checkpoint_pattern, checkpoint_params):
    # type: (Filepath, str, dict) -> list
    '''
    Create a list of callbacks for Tensoflow model.

    Args:
        log_directory (str or Path): Tensorboard project log directory.
        checkpoint_pattern (str): Filepath pattern for checkpoint callback.
        checkpoint_params (dict): Params to be passed to checkpoint callback.

    Raises:
        EnforceError: If log directory does not exist.
        EnforeError: If checkpoint pattern does not contain '{epoch}'.

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


# MISC--------------------------------------------------------------------------
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
    source=None,  # type: Optional[str]
    target=None,  # type: Optional[str]
    params=None,  # type: Optional[dict]
    stopwatch=None,  # type: Optional[StopWatch]
    timezone='UTC',  # type: str
    testing=False,  # type: bool
):
    # type: (...) -> Union[str, HTTPResponse]
    '''
    Compose a message from given arguments and post it to slack.

    Args:
        title (str): Post title.
        channel (str): Slack channel.
        url (str): Slack URL.
        source (str, optional): Source data. Default: None.
        target (str, optional): Target data. Default: None.
        params (dict, optional): Parameter dict. Default: None.
        stopwatch (StopWatch, optional): StopWatch instance. Default: None.
        timezone (str, optional): Timezone. Default: UTC.
        testing (bool, optional): Test mode: Default: False.

    Returns:
        HTTPResponse: Slack response.
    '''
    now = datetime.now(tz=pytz.timezone(timezone)).isoformat()
    source = source or 'none'
    target = target or 'none'
    params = params or {}
    delta = 'none'
    hdelta = 'none'
    if stopwatch is not None:
        hdelta = stopwatch.human_readable_delta
        delta = str(stopwatch.delta)

    params = yaml.safe_dump(params, indent=4)
    if isinstance(target, dict):
        target = json.dumps(target, indent=4)
    elif not isinstance(target, str):
        target = str(target)

    message = f'''
        {title.upper()}

        RUNTIME:
        ```{hdelta} ({delta})```
        SOURCE:
        ```{source}```
        TARGET:
        ```{target}```
        PARAMS:
        ```{params}```
        TIME:
        ```{now}```
    '''[1:-1]
    message = unindent(message, spaces=8)

    if testing:
        return message
    return lbt.post_to_slack(url, channel, message)
