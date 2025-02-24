import os
from tempfile import TemporaryDirectory

import tensorflow as tf
from tensorflow import keras  # noqa F401
from keras import callbacks as tfcallbacks

import flatiron
import flatiron.core.tools as fict
from flatiron.core.dataset import Dataset
from flatiron.core.dataset_test import DatasetTestBase
import flatiron.tf.models.dummy as fi_tfdummy
import flatiron.tf.tools as fi_tftools

# disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# ------------------------------------------------------------------------------


class MockModel:
    def fit(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def compile(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class TFToolsTests(DatasetTestBase):
    def test_get_callbacks(self):
        with TemporaryDirectory() as root:
            proj = fict.get_tensorboard_project('proj', root)
            result = fi_tftools.get_callbacks(
                proj['log_dir'], proj['checkpoint_pattern']
            )
            self.assertIsInstance(result['tensorboard'], tfcallbacks.TensorBoard)
            self.assertIsInstance(result['checkpoint'], tfcallbacks.ModelCheckpoint)

    def test_pre_build(self):
        fi_tftools.pre_build('cpu')
        result = tf.config.get_visible_devices('GPU')
        self.assertEqual(result, [])

    def test_compile(self):
        model = MockModel()
        result = fi_tftools.compile(
            model=model,
            optimizer=dict(name='Adam', learning_rate=0.01),
            loss=dict(name='MeanSquaredError'),
            metrics=[dict(name='dice')],
            device='cpu',
            kwargs=dict(jit_compile=True),
        )
        self.assertEqual(result, dict(model=model))
        self.assertTrue(model.kwargs['jit_compile'])
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES', 'xxx'), '-1')

        expected = dict(name='Adam', learning_rate=0.01)
        expected = flatiron.tf.optimizer.get(expected).__class__
        self.assertIsInstance(model.kwargs['optimizer'], expected)

        expected = flatiron.tf.loss.get(dict(name='MeanSquaredError')).__class__
        self.assertIsInstance(model.kwargs['loss'], expected)

        expected = flatiron.tf.metric.get(dict(name='dice')).__class__
        self.assertIsInstance(model.kwargs['metrics'][0], expected)

    def test_compile_device(self):
        model = MockModel()
        result = fi_tftools.compile(
            model=model,
            optimizer=dict(name='Adam', learning_rate=0.01),
            loss=dict(name='MeanSquaredError'),
            metrics=[dict(name='dice')],
            device='1',
            kwargs=dict(jit_compile=True),
        )
        self.assertEqual(result, dict(model=model))
        self.assertTrue(model.kwargs['jit_compile'])

    def test_train(self):
        model = fi_tfdummy.get_dummy_model((10, 10, 3))
        model.compile(
            loss=dict(name='MeanSquaredError'),
            optimizer=dict(name='Adam'),
        )
        compiled = dict(model=model)

        with TemporaryDirectory() as root:
            self.create_png_dataset_files(root)
            train, test = Dataset \
                .read_directory(root, labels=[3]) \
                .train_test_split()
            train.load()
            test.load()

            fi_tftools.train(
                compiled=compiled,
                callbacks={},
                train_data=train,
                test_data=test,
                batch_size=1,
            )
