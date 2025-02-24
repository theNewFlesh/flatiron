from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tensorflow import keras  # noqa F401
import pydantic_core._pydantic_core as pydc

import flatiron.core.config as ficc
# ------------------------------------------------------------------------------


class DatasetConfigTests(unittest.TestCase):
    def get_config(self, root):
        return dict(
            source=Path(root).as_posix(),
            ext_regex='npy|exr|png|jpeg|jpg|tiff',
            labels=None,
            label_axis=-1,
            test_size=0.2,
            limit=None,
            reshape=True,
            shuffle=True,
            seed=None,
        )

    def test_base_config(self):
        class Foo(ficc.BaseConfig):
            bar: str  # type: ignore

        Foo.model_validate(dict(bar='taco'))

        with self.assertRaises(pydc.ValidationError):
            Foo.model_validate(dict(bar='taco', pizza='kiwi'))

    def test_validate(self):
        with TemporaryDirectory() as root:
            ficc.DatasetConfig(**self.get_config(root))

    def test_model_dump(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            keys = [
                'ext_regex',
                'labels',
                'label_axis',
                'test_size',
                'limit',
                'reshape',
                'seed',
                'shuffle',
            ]
            for key in keys:
                del config[key]

            result = ficc.DatasetConfig(**config).model_dump()
            self.assertEqual(result, self.get_config(root))

    def test_split_test_size(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            config['test_size'] = -0.2
            with self.assertRaises(ValueError):
                ficc.DatasetConfig(**config)

    def test_split_test_limit(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            config['limit'] = -10
            with self.assertRaises(ValueError):
                ficc.DatasetConfig(**config)


class FrameworkConfigTests(unittest.TestCase):
    def get_config(self):
        return dict(name='torch', foo='bar')

    def test_validate(self):
        ficc.FrameworkConfig.model_validate(dict(name='tensorflow'))
        ficc.FrameworkConfig.model_validate(dict(name='torch'))
        with self.assertRaises(ValueError):
            ficc.FrameworkConfig.model_validate(dict(name='foo'))

    def test_defaults(self):
        expected = dict(name='tensorflow')
        result = ficc.FrameworkConfig().model_dump()
        self.assertEqual(result, expected)


class OptimizerConfigTests(unittest.TestCase):
    def get_config(self):
        return dict(
            name='SGD',
            learning_rate=0.001,
            momentum=0.0,
            nesterov=False,
        )

    def test_validate(self):
        ficc.OptimizerConfig(**self.get_config())

    def test_defaults(self):
        expected = dict(name='SGD')
        result = ficc.OptimizerConfig().model_dump()
        self.assertEqual(result, expected)


class LossConfigTests(unittest.TestCase):
    def get_config(self):
        return dict(
            name='MeanSquaredError',
            foo='bar',
        )

    def test_validate(self):
        ficc.LossConfig(**self.get_config())

    def test_defaults(self):
        expected = dict(name='MeanSquaredError')
        result = ficc.LossConfig().model_dump()
        self.assertEqual(result, expected)


class CallbacksConfigTests(unittest.TestCase):
    def get_config(self):
        return dict(
            project='project',
            root='root',
            monitor='val_loss',
            verbose=0,
            save_best_only=False,
            save_weights_only=False,
            mode='auto',
            save_freq='epoch',
            initial_value_threshold=None,
        )

    def test_validate(self):
        ficc.CallbacksConfig(**self.get_config())

        config = self.get_config()
        config['mode'] = None
        with self.assertRaises(ValueError):
            ficc.CallbacksConfig(**config)

    def test_defaults(self):
        expected = self.get_config()
        config = dict(project='project', root='root')
        result = ficc.CallbacksConfig(**config).model_dump()
        self.assertEqual(result, expected)


class TrainConfigTests(unittest.TestCase):
    def get_config(self):
        return dict(
            batch_size=32,
            epochs=30,
            verbose='auto',
            validation_split=0.0,
            shuffle=True,
            initial_epoch=1,
            validation_freq=1,
        )

    def test_validate(self):
        ficc.TrainConfig(**self.get_config())

    def test_defaults(self):
        expected = self.get_config()
        result = ficc.TrainConfig().model_dump()
        self.assertEqual(result, expected)


class LoggerConfigTests(unittest.TestCase):
    def get_config(self):
        return dict(
            slack_channel=None,
            slack_url=None,
            slack_methods=['load', 'compile', 'train'],
            timezone='UTC',
            level='warn',
        )

    def test_validate(self):
        ficc.LoggerConfig(**self.get_config())

    def test_slack_methods(self):
        config = self.get_config()
        config['slack_methods'] = ['load', 'foo', 'bar']

        expected = 'foo is not a legal pipeline method'
        with self.assertRaisesRegex(ValueError, expected):
            ficc.LoggerConfig(**config)

    def test_defaults(self):
        expected = self.get_config()
        result = ficc.LoggerConfig().model_dump()
        self.assertEqual(result, expected)


class PipelineConfigTests(unittest.TestCase):
    def get_config(self):
        return dict(
            framework=dict(name='tensorflow'),
            dataset=dict(
                source='/tmp/foobar/info.csv',
                label_axis=-1,
            ),
            optimizer=dict(
                name='adam',
            ),
            loss=dict(
                name='jaccard_loss',
            ),
            metrics=[
                dict(name='Mean'),
            ],
            callbacks=dict(
                project='project',
                root='root',
            ),
            logger=dict(),
            train=dict(),
        )

    def test_validate(self):
        result = ficc.PipelineConfig \
            .model_validate(self.get_config(), strict=True) \
            .model_dump()
        self.assertEqual(result['callbacks']['mode'], 'auto')
        self.assertEqual(result['framework']['name'], 'tensorflow')

    def test_validate_metrics(self):
        config = self.get_config()
        config['metrics'] = [dict(metrics=[dict(name='Mean'), {}])]
        expected = 'All dicts must contain name key. Given value:.*{}.'
        with self.assertRaisesRegex(ValueError, expected):
            ficc.PipelineConfig.model_validate(config, strict=True)

    def test_errors(self):
        config = self.get_config()
        config['framework']['name'] = None
        with self.assertRaises(ValueError):
            ficc.PipelineConfig(**config)

        del config['framework']
        with self.assertRaises(ValueError):
            ficc.PipelineConfig(**config)

        config = self.get_config()
        config['logger'] = 123
        with self.assertRaises(ValueError):
            ficc.PipelineConfig(**config)

        config = self.get_config()
        config['callbacks']['mode'] = 'foobar'
        with self.assertRaises(ValueError):
            ficc.PipelineConfig(**config)
