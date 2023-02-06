from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from schematics.exceptions import DataError
import tensorflow.keras.optimizers as tfo

import flatiron.core.config as ficc
# ------------------------------------------------------------------------------


class DatasetConfigTests(unittest.TestCase):
    def get_config(self, root):
        return dict(
            source=Path(root).as_posix(),
            load_limit=None,
            load_shuffle=False,
            split_index=-1,
            split_axis=-1,
            split_test_size=0.2,
            split_train_size=None,
            split_random_state=42,
            split_shuffle=True,
        )

    def test_validate(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            ficc.DatasetConfig(config).validate()

    def test_to_native(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            keys = [
                'load_limit',
                'load_shuffle',
                'split_axis',
                'split_test_size',
                'split_train_size',
                'split_random_state',
                'split_shuffle',
            ]
            for key in keys:
                del config[key]

            result = ficc.DatasetConfig(config).to_native()
            self.assertEqual(result, self.get_config(root))

    def test_split_test_size(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            config['split_test_size'] = -0.2
            with self.assertRaises(DataError):
                ficc.DatasetConfig(config).validate()

    def test_split_train_size(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            config['split_train_size'] = -0.2
            with self.assertRaises(DataError):
                ficc.DatasetConfig(config).validate()


class OptimizerConfigTests(unittest.TestCase):
    def get_config(self):
        return dict(
            name='sgd',
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
        config = self.get_config()
        ficc.OptimizerConfig(config).validate()

    def test_defaults(self):
        expected = self.get_config()
        result = ficc.OptimizerConfig(dict(name='sgd')).to_native()
        self.assertEqual(result, expected)

        del result['name']
        tfo.get('sgd', **result)


class CompileConfigTests(unittest.TestCase):
    def get_config(self):
        return dict(
            loss='dice_loss',
            metrics=[],
            loss_weights=None,
            weighted_metrics=None,
            run_eagerly=False,
            steps_per_execution=1,
            jit_compile=False,
        )

    def test_validate(self):
        config = self.get_config()
        ficc.CompileConfig(config).validate()

    def test_defaults(self):
        expected = self.get_config()
        config = dict(loss='dice_loss')
        result = ficc.CompileConfig(config).to_native()
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
            experimental_io_device=None,
            experimental_enable_async_checkpoint=False,
        )

    def test_validate(self):
        config = self.get_config()
        ficc.CallbacksConfig(config).validate()

    def test_defaults(self):
        expected = self.get_config()
        config = dict(project='project', root='root')
        result = ficc.CallbacksConfig(config).to_native()
        self.assertEqual(result, expected)


class FitConfigTests(unittest.TestCase):
    def get_config(self):
        return dict(
            batch_size=32,
            epochs=30,
            verbose='auto',
            validation_split=0.0,
            shuffle=True,
            initial_epoch=1,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
        )

    def test_validate(self):
        config = self.get_config()
        ficc.FitConfig(config).validate()

    def test_defaults(self):
        expected = self.get_config()
        result = ficc.FitConfig({}).to_native()
        self.assertEqual(result, expected)


class LoggerConfigTests(unittest.TestCase):
    def get_config(self):
        return dict(
            slack_channel=None,
            slack_url=None,
            slack_methods=['load', 'compile', 'fit'],
            timezone='UTC',
            level='warn',
        )

    def test_validate(self):
        config = self.get_config()
        ficc.LoggerConfig(config).validate()

    def test_slack_methods(self):
        config = self.get_config()
        config['slack_methods'] = ['load', 'foo', 'bar']

        expected = 'foo is not a legal pipeline method.*'
        expected += 'bar is not a legal pipeline method'
        with self.assertRaisesRegex(DataError, expected):
            ficc.LoggerConfig(config).validate()

    def test_defaults(self):
        expected = self.get_config()
        result = ficc.LoggerConfig({}).to_native()
        self.assertEqual(result, expected)


class PipelineConfigTests(unittest.TestCase):
    def get_config(self):
        return dict(
            model={},
            dataset=dict(source='/tmp/foobar/info.csv', split_index=-1),
            preprocess=dict(name='identity'),
            optimizer=dict(),
            compile=dict(loss='jaccard_loss'),
            callbacks=dict(project='project', root='root'),
            fit=dict(),
            logger=dict(),
        )

    def test_validate(self):
        config = self.get_config()
        ficc.PipelineConfig(config).validate()
