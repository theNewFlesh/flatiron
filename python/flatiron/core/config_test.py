from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tensorflow import keras  # noqa F401
from keras import optimizers as tfoptim

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
            shuffle=True,
            seed=None,
        )

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


class OptimizerConfigTests(unittest.TestCase):
    def get_config(self):
        return dict(
            class_name='sgd',
            learning_rate=0.001,
            momentum=0,
            nesterov=False,
            weight_decay=0.0,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            use_ema=False,
            ema_momentum=0.99,
            ema_overwrite_frequency=None,
            jit_compile=True,
        )

    def test_validate(self):
        ficc.OptimizerConfig(**self.get_config())

    def test_defaults(self):
        expected = self.get_config()
        result = ficc.OptimizerConfig(class_name='sgd').model_dump()
        self.assertEqual(result, expected)
        tfoptim.get(result)


class CompileConfigTests(unittest.TestCase):
    def get_config(self):
        return dict(
            loss='dice_loss',
            metrics=[],
            tf_loss_weights=None,
            tf_weighted_metrics=None,
            tf_run_eagerly=False,
            tf_steps_per_execution=1,
            tf_jit_compile=False,
        )

    def test_validate(self):
        ficc.CompileConfig(**self.get_config())

    def test_defaults(self):
        expected = self.get_config()
        result = ficc.CompileConfig(loss='dice_loss').model_dump()
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
            engine='tensorflow',
            dataset=dict(
                source='/tmp/foobar/info.csv',
                split_index=-1,
            ),
            optimizer=dict(),
            compile=dict(
                loss='jaccard_loss',
            ),
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
        self.assertEqual(result['engine'], 'tensorflow')

    def test_errors(self):
        config = self.get_config()
        config['engine'] = None
        with self.assertRaises(ValueError):
            ficc.PipelineConfig(**config)

        del config['engine']
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
