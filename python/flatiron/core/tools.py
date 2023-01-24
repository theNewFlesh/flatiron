from typing import Any, Optional, Union  # noqa F401
from http.client import HTTPResponse  # noqa F401
from lunchbox.stopwatch import StopWatch  # noqa F401

from datetime import datetime
import json
import re

import lunchbox.tools as lbt
import pytz
import yaml
# ------------------------------------------------------------------------------


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
