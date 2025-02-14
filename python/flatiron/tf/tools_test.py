from tempfile import TemporaryDirectory
import unittest

import numpy as np
from keras import callbacks as tfcallbacks

import flatiron
import flatiron.core.tools as fict
import flatiron.tf.tools as fi_tftools
# ------------------------------------------------------------------------------


class MockModel:
    def fit(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def compile(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class TFToolsTests(unittest.TestCase):
    def test_get_callbacks(self):
        with TemporaryDirectory() as root:
            proj = fict.get_tensorboard_project('proj', root)
            result = fi_tftools.get_callbacks(
                proj['log_dir'], proj['checkpoint_pattern']
            )
            self.assertIsInstance(result['tensorboard'], tfcallbacks.TensorBoard)
            self.assertIsInstance(result['checkpoint'], tfcallbacks.ModelCheckpoint)

    def test_compile(self):
        model = MockModel()
        result = fi_tftools.compile(
            model=model, optimizer='adam', loss='mse', metrics=['dice'],
            kwargs=dict(jit_compile=True)
        )
        self.assertEqual(result, dict(model=model))
        self.assertTrue(model.kwargs['jit_compile'])

        expected = flatiron.tf.optimizer.get('adam').__class__
        self.assertIsInstance(model.kwargs['optimizer'], expected)

        expected = flatiron.tf.loss.get('mse').__class__
        self.assertIsInstance(model.kwargs['loss'], expected)

        expected = flatiron.tf.metric.get('dice').__class__
        self.assertIsInstance(model.kwargs['metrics'][0], expected)

    def test_train(self):
        model = MockModel()
        expected = dict(
            x=np.ones(100),
            y=np.ones(100),
            callbacks=[],
            validation_data=(np.ones(10), np.ones(10)),
            steps_per_epoch=4,
            foobar=True,
        )
        fi_tftools.train(
            model=model,
            callbacks=[],
            x_train=np.ones(100),
            y_train=np.ones(100),
            x_test=np.ones(10),
            y_test=np.ones(10),
            batch_size=25,
            foobar=True,
        )
        for key, result in model.kwargs.items():
            if key in ['x', 'y']:
                self.assertEqual(str(result), str(expected[key]))
            elif key == 'validation_data':
                self.assertEqual(str(result[0]), str(expected[key][0]))
                self.assertEqual(str(result[1]), str(expected[key][1]))
            else:
                self.assertEqual(result, expected[key])
