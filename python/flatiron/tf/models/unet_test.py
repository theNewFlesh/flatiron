import unittest

from lunchbox.enforce import EnforceError
from tensorflow import keras  # noqa: F401
from keras import models as tfm

import flatiron.tf.models.unet as ftfu
# ------------------------------------------------------------------------------


class UNetTests(unittest.TestCase):
    def get_kwargs(self):
        return dict(
            input_width=208,
            input_height=208,
            input_channels=3,
            classes=1,
            filters=16,
            layers=9,
            activation='relu',
            batch_norm=True,
            output_activation='sigmoid',
            kernel_initializer='he_normal',
            attention_gates=False,
            attention_activation_1='relu',
            attention_activation_2='sigmoid',
            attention_kernel_size=1,
            attention_strides=1,
            attention_padding='same',
            attention_kernel_initializer='he_normal',
        )

    def test_get_unet_model(self):
        result = ftfu.get_unet_model(**self.get_kwargs())
        self.assertIsInstance(result, tfm.Model)

    def test_get_unet_model_errors(self):
        kwargs = self.get_kwargs()
        exp = 'Layers must be an odd integer greater than 2. Given value: '

        # float
        layers = 9.0
        kwargs['layers'] = layers
        expected = exp + str(layers)
        with self.assertRaisesRegex(EnforceError, expected):
            ftfu.get_unet_model(**kwargs)

        # < 3
        layers = 2
        kwargs['layers'] = layers
        expected = exp + str(layers)
        with self.assertRaisesRegex(EnforceError, expected):
            ftfu.get_unet_model(**kwargs)

        # even
        layers = 8
        kwargs['layers'] = layers
        expected = exp + str(layers)
        with self.assertRaisesRegex(EnforceError, expected):
            ftfu.get_unet_model(**kwargs)

        # bad width and layers
        layers = 9
        kwargs['layers'] = layers
        kwargs['input_height'] = 100
        kwargs['input_width'] = 100
        expected = 'Given input_width and layers are not compatible. '
        expected += 'Input_width: 100. Layers: 9.'
        with self.assertRaisesRegex(EnforceError, expected):
            ftfu.get_unet_model(**kwargs)

    def test_unet_width_and_layers_are_valid(self):
        result = ftfu.unet_width_and_layers_are_valid(128, 9)
        self.assertTrue(result)

        result = ftfu.unet_width_and_layers_are_valid(130, 9)
        self.assertFalse(result)
