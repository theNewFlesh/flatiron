from typing import Any, Optional  # noqa F401

import lunchbox.tools as lbt

import flatiron.core.tools as fict
# ------------------------------------------------------------------------------


class SlackLogger(lbt.LogRuntime):
    '''
    SlackLogger is a class for logging information to stdout and Slack.
    '''
    def __init__(
        self,
        message,  # type: str
        config,  # type: dict
        slack_channel=None,  # type: Optional[str]
        slack_url=None,  # type: Optional[str]
        timezone='UTC',  # type: str
        level='warn',
        **kwargs,  # type: Any
    ):
        # type: (...) -> None
        '''
        SlackLogger is a class for logging information to stdout and Slack.

        If slack_url and slack_channel are specified, SlackLogger will
        attempt to log custom formatted output to Slack.

        Args:
            message (str): Log message or Slack title.
            config (dict): Config dict.
            slack_channel (str, optional): Slack channel name. Default: None.
            slack_url (str, optional): Slack URL name. Default: None.
            timezone (str, optional): Timezone. Default: UTC.
            level (str or int, optional): Log level. Default: warn.
            **kwargs (optional): LogRuntime kwargs.
        '''
        super().__init__(message=message, level=level, **kwargs)

        if slack_channel is not None and slack_url is not None:
            self._message_func = lambda _, stp: fict.slack_it(
                title=message,
                channel=str(slack_channel),
                url=str(slack_url),
                config=config,
                stopwatch=stp,
                timezone=timezone,
                suppress=True,
            )  # type: Any
            self._callback = lambda msg: lbt.post_to_slack(
                slack_url,
                slack_channel,
                msg,
            )
