from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from lunchbox.enforce import EnforceError
from lunchbox.stopwatch import StopWatch
from tensorflow import keras  # noqa: F401
from keras import callbacks as tfc

import flatiron.core.tools as fict
# ------------------------------------------------------------------------------


def fake_func(foo):
    return foo + 'bar'


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

    def test_enforce_callbacks(self):
        with TemporaryDirectory() as root:
            proj = fict.get_tensorboard_project('proj', root)
            fict.enforce_callbacks(proj['log_dir'], proj['checkpoint_pattern'])

    def test_enforce_callbacks_errors(self):
        # log dir
        expected = 'Log directory: /tmp/foobar does not exist.'
        with self.assertRaisesRegex(EnforceError, expected):
            fict.enforce_callbacks('/tmp/foobar', 'pattern')

        # checkpoint pattern
        with TemporaryDirectory() as root:
            expected = r"Checkpoint pattern must contain '\{epoch\}'\. "
            expected += 'Given value: foobar'
            with self.assertRaisesRegex(EnforceError, expected):
                fict.enforce_callbacks(root, 'foobar')

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
            config=dict(a=dict(b=1)),
            stopwatch=stopwatch,
            timezone='America/Los_Angeles',
            suppress=True,
        )
        result = fict.slack_it(**kwargs)

        # keys
        expected = 'TITLE\n\nRUN TIME(\n|.)*POST TIME(\n|.)*CONFIG'
        self.assertRegex(result, expected)

        # title
        expected = 'TITLE\n'
        self.assertRegex(result, expected)

        # post time
        expected = r'TIME:\n```\d\d\d\d-\d\d-\d\dT.*```'
        self.assertRegex(result, expected)

        # run time
        expected = r'RUN TIME:\n```0 seconds \(0:00:00\.\d+\)```'
        self.assertRegex(result, expected)

        # config
        expected = '''
            CONFIG:
            ```a:
                b: 1
            ```'''[1:]
        expected = fict.unindent(expected, 12)
        self.assertRegex(result, expected)

    def test_get_module_function(self):
        func = fict.get_module_function('fake_func', __name__)
        self.assertIs(func, fake_func)
        self.assertEqual(func('foo'), 'foobar')

        expected = 'Function not found: nonexistent_func'
        with self.assertRaisesRegex(NotImplementedError, expected):
            fict.get_module_function('nonexistent_func', __name__)
