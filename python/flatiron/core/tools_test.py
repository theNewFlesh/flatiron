from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from lunchbox.enforce import EnforceError
from lunchbox.stopwatch import StopWatch
import pandas as pd

import flatiron.core.tools as fict
# ------------------------------------------------------------------------------


def fake_func(foo):
    return foo + 'bar'


class FakeClass:
    pass


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
                f'{root_re}/{time_re}/models/{model_re}.keras'
            )

            self.assertTrue(Path(result['root_dir']).is_dir())
            self.assertTrue(Path(result['log_dir']).is_dir())
            self.assertTrue(Path(result['model_dir']).is_dir())

    def test_get_tensorboard_project_extension(self):
        with TemporaryDirectory() as root:
            result = fict.get_tensorboard_project(
                'foo', root, timezone='America/Los_Angeles',
                extension='safetensors'
            )

            self.assertRegex(
                result['checkpoint_pattern'],
                f'{root}/foo/tensorboard/.*/models/p-foo_.*_e-{{epoch:03d}}.safetensors'
            )

    def test_get_tensorboard_project_errors(self):
        expected = 'Extension must be keras or safetensors. Given value: pth.'
        with TemporaryDirectory() as root:
            with self.assertRaisesRegex(EnforceError, expected):
                fict.get_tensorboard_project('foo', root, extension='pth')

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

    def test_resolve_kwargs(self):
        kwargs = dict(
            model=0, optimizer=0, loss=0,
            tf_foo=0, tf_bar=0,
            torch_taco=0, torch_pizza=0,
            adam_kiwi=10,
            sgd_egg=20,
        )

        # prefix
        result = fict.resolve_kwargs('tensorflow', kwargs, return_keys='prefix')
        self.assertEqual(result, dict(foo=0, bar=0))

        result = fict.resolve_kwargs('torch', kwargs, return_keys='prefix')
        self.assertEqual(result, dict(taco=0, pizza=0))

        result = fict.resolve_kwargs('adam', kwargs, return_keys='prefix')
        self.assertEqual(result, dict(kiwi=10))

        result = fict.resolve_kwargs('sgd', kwargs, return_keys='prefix')
        self.assertEqual(result, dict(egg=20))

        # non-prefix
        expected = dict(model=0, optimizer=0, loss=0)
        result = fict.resolve_kwargs('tensorflow', kwargs, return_keys='non-prefix')
        self.assertEqual(result, expected)

        result = fict.resolve_kwargs('torch', kwargs, return_keys='non-prefix')
        self.assertEqual(result, expected)

        # both
        expected = dict(model=0, optimizer=0, loss=0, foo=0, bar=0)
        result = fict.resolve_kwargs('tensorflow', kwargs)
        self.assertEqual(result, expected)

    def test_resolve_kwargs_errors(self):
        expected = 'Illegal prefix: wrong. Legal prefixes: .*tf.*torch.*sgd.*adam'
        with self.assertRaisesRegex(AssertionError, expected):
            fict.resolve_kwargs('wrong', {})

    def test_train_test_split(self):
        data = pd.DataFrame()
        data['x'] = list(range(100))
        data['y'] = list(range(100))

        # regular use
        result = fict.train_test_split(data, test_size=0.2, shuffle=True)
        self.assertIsInstance(result[0], pd.DataFrame)
        self.assertIsInstance(result[1], pd.DataFrame)
        self.assertEqual(result[0].columns.tolist(), list('xy'))
        self.assertEqual(result[1].columns.tolist(), list('xy'))
        self.assertEqual(len(result[0]), 80)
        self.assertEqual(len(result[1]), 20)

        # shuffle
        result = fict.train_test_split(data, test_size=0.1, shuffle=False)
        self.assertEqual(result[0].index.tolist(), list(range(0, 90)))
        self.assertEqual(result[1].index.tolist(), list(range(90, 100)))

        # seed
        x0, y0 = fict.train_test_split(data, shuffle=True, seed=0.1)
        x1, y1 = fict.train_test_split(data, shuffle=True, seed=0.1)
        x2, y2 = fict.train_test_split(data, shuffle=True, seed=0.1)
        self.assertTrue(x0.equals(x0))
        self.assertTrue(x0.equals(x1))
        self.assertTrue(x0.equals(x2))
        self.assertTrue(y0.equals(y0))
        self.assertTrue(y0.equals(y1))
        self.assertTrue(y0.equals(y2))

        # limit
        x, y = fict.train_test_split(data, limit=10, test_size=0.2)
        self.assertEqual(len(x), 8)
        self.assertEqual(len(y), 2)

    def test_train_test_split_errors(self):
        with self.assertRaises(EnforceError):
            fict.train_test_split('foobar')

        with self.assertRaises(EnforceError):
            fict.train_test_split(pd.DataFrame(), test_size=1.1)

    def test_get_module(self):
        module = fict.get_module(__name__)
        self.assertEqual(module.__name__, __name__)

        expected = 'Module not found: foobar'
        with self.assertRaisesRegex(NotImplementedError, expected):
            fict.get_module('foobar')

    def test_get_module_function(self):
        func = fict.get_module_function('fake_func', __name__)
        self.assertIs(func, fake_func)
        self.assertEqual(func('foo'), 'foobar')

        expected = 'Function not found: nonexistent_func'
        with self.assertRaisesRegex(NotImplementedError, expected):
            fict.get_module_function('nonexistent_func', __name__)

    def test_get_module_class(self):
        func = fict.get_module_class('FakeClass', __name__)
        self.assertIs(func, FakeClass)

        expected = 'Class not found: NonClass'
        with self.assertRaisesRegex(NotImplementedError, expected):
            fict.get_module_class('NonClass', __name__)
