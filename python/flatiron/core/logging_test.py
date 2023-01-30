import logging
import time
import unittest

import flatiron.core.logging as ficl
# ------------------------------------------------------------------------------


class LoggingTests(unittest.TestCase):
    def test_slacklogger(self):
        kwargs = dict(
            message='title',
            config=dict(a=[1, 2, 3]),
            slack_channel='foobar,',
            slack_url='https://hooks.slack.com/services/test',
        )
        with self.assertLogs(level=logging.WARNING) as result:
            with ficl.SlackLogger(**kwargs):
                time.sleep(0.01)
            expected = 'TITLE\n\n.*RUN TIME(\n|.)*POST TIME(\n|.)*CONFIG'
            self.assertRegex(result.output[0], expected)
