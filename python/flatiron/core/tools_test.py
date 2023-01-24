import unittest

from lunchbox.stopwatch import StopWatch

import flatiron.core.tools as fict
# ------------------------------------------------------------------------------


class ToolsTests(unittest.TestCase):
    def test_unindent(self):
        # 4 spaces
        text = '''
    a
    b
        c
        d'''[1:]

        expected = '''
a
b
    c
    d'''[1:]
        result = fict.unindent(text)
        self.assertEqual(result, expected)

        # 2 spaces
        text = '''
  a
  b
    c
    d'''[1:]

        expected = '''
a
b
  c
  d'''[1:]
        result = fict.unindent(text, spaces=2)
        self.assertEqual(result, expected)

    def test_slack_it(self):
        stopwatch = StopWatch()
        stopwatch.start()
        stopwatch.stop()

        kwargs = dict(
            title='title',
            channel='channel',
            url='url',
            source='source',
            target='target',
            params=dict(a=dict(b=1)),
            stopwatch=stopwatch,
            timezone='America/Los_Angeles',
            testing=True,
        )
        result = fict.slack_it(**kwargs)

        # keys
        expected = 'TITLE(\n|.)*RUNTIME(\n|.)*SOURCE(\n|.)*TARGET(\n|.)*'
        expected += 'PARAMS(\n|.)*TIME'
        self.assertRegex(result, expected)

        # title
        expected = 'TITLE\n'
        self.assertRegex(result, expected)

        # runtime
        expected = r'RUNTIME:\n```0 seconds \(0:00:00\.\d+\)```'
        self.assertRegex(result, expected)

        # source
        expected = 'SOURCE:\n```source```'
        self.assertRegex(result, expected)

        # target
        expected = 'TARGET:\n```target```'
        self.assertRegex(result, expected)

        # params
        expected = '''
            PARAMS:
            ```a:
                b: 1
            ```'''[1:]
        expected = fict.unindent(expected, 12)
        self.assertRegex(result, expected)

        # time
        expected = r'TIME:\n```\d\d\d\d-\d\d-\d\dT.*```'
        self.assertRegex(result, expected)
