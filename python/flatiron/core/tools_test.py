from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from lunchbox.enforce import EnforceError
from lunchbox.stopwatch import StopWatch
import tensorflow.keras.callbacks as tfc

import flatiron.core.tools as fict
# ------------------------------------------------------------------------------


class ToolsTests(unittest.TestCase):
    def test_get_tensorboard_project(self):
        with TemporaryDirectory() as root:
            result = fict.get_tensorboard_project(
                'foo', root, timezone='America/Los_Angeles'
            )

            # root dir
            root_re = f'{root}/foo/tensorboard'
            self.assertRegex(result['root_dir'], root_re)

            # log dir
            time_re = r'd-\d\d\d\d-\d\d-\d\d_t-\d\d-\d\d-\d\d'
            self.assertRegex(result['log_dir'], f'{root_re}/{time_re}')

            # model dir
            self.assertRegex(result['model_dir'], f'{root_re}/{time_re}/models')

            # checkpoint pattern
            model_re = f'p-foo_{time_re}_e-{{epoch:03d}}'
            self.assertRegex(
                result['checkpoint_pattern'],
                f'{root_re}/{time_re}/models/{model_re}'
            )

            self.assertTrue(Path(result['root_dir']).is_dir())
            self.assertTrue(Path(result['log_dir']).is_dir())
            self.assertTrue(Path(result['model_dir']).is_dir())

    def test_get_callbacks(self):
        with TemporaryDirectory() as root:
            proj = fict.get_tensorboard_project('proj', root)
            result = fict.get_callbacks(
                proj['log_dir'], proj['checkpoint_pattern'], {}
            )
            self.assertIsInstance(result[0], tfc.TensorBoard)
            self.assertIsInstance(result[1], tfc.ModelCheckpoint)

    def test_get_callbacks_errors(self):
        # log dir
        expected = 'Log directory: /tmp/foobar does not exist.'
        with self.assertRaisesRegex(EnforceError, expected):
            fict.get_callbacks('/tmp/foobar', 'pattern', {})

        # checkpoint pattern
        with TemporaryDirectory() as root:
            expected = r"Checkpoint pattern must contain '\{epoch\}'\. "
            expected += 'Given value: foobar'
            with self.assertRaisesRegex(EnforceError, expected):
                fict.get_callbacks(root, 'foobar', {})

    def test_pad_layer_name(self):
        expected = 'foo____bar'
        result = fict.pad_layer_name('foo_bar', 10)
        self.assertEqual(len(result), 10)
        self.assertEqual(result, expected)

        result = fict.pad_layer_name('foo___bar', 10)
        self.assertEqual(len(result), 10)
        self.assertEqual(result, expected)

        result = fict.pad_layer_name('foo________bar', 10)
        self.assertEqual(len(result), 10)
        self.assertEqual(result, expected)

        expected = 'foo_______'
        result = fict.pad_layer_name('foo_', 10)
        self.assertEqual(len(result), 10)
        self.assertEqual(result, expected)

        result = fict.pad_layer_name('foo', 10)
        self.assertEqual(len(result), 10)
        self.assertEqual(result, expected)

        result = fict.pad_layer_name('foo__', 0)
        self.assertEqual(len(result), 5)
        self.assertEqual(result, 'foo__')

        result = fict.pad_layer_name('foo__bar', 0)
        self.assertEqual(len(result), 8)
        self.assertEqual(result, 'foo__bar')

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
            suppress=True,
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

    def test_slack_it_target(self):
        stopwatch = StopWatch()
        stopwatch.start()
        stopwatch.stop()

        kwargs = dict(
            title='title',
            channel='channel',
            url='url',
            target=dict(foo='bar'),
            suppress=True,
        )
        result = fict.slack_it(**kwargs)

        # dict
        expected = 'TARGET:\n```foo: bar\n```'
        self.assertRegex(result, expected)

        # None
        kwargs['target'] = None
        result = fict.slack_it(**kwargs)
        expected = 'TARGET:\n```none```'
        self.assertRegex(result, expected)
