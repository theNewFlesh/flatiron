# from pathlib import Path
# import json
# import math
# import os
# import re

# from keras_unet.models.custom_unet import *
# from lunchbox.stopwatch import StopWatch
# from tensorflow.keras.callbacks import TensorBoard
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import keras_unet
# import keras_unet.metrics as unmet
# import keras_unet.models as unmod
# import numpy as np
# import pandas as pd
# import tensorflow_addons as tfa
# import tensorflow.keras.backend as tfkb
# import tensorflow.keras.optimizers as tfko
# import tensorflow.keras.models as tfkm

from lunchbox.enforce import Enforce
import numpy as np
import tensorflow as tf
import sklearn.model_selection as skm
import tensorflow.keras.layers as tfkl
import tensorflow.keras.models as tfkm
import tensorflow.keras.optimizers as tfko
import tensorflow.keras.preprocessing.image as tfkpp

import flatiron.core.loss as ficl
import flatiron.core.metric as ficm
import flatiron.core.tools as fict
# ------------------------------------------------------------------------------


def conv_2d_block(
    input_,
    filters=16,
    activation='relu',
    batch_norm=True,
    kernel_initializer='he_normal',
):
    # type: (tf.Tensor, int, str, bool, str) -> tf.Tensor
    r'''
    2D Convolution block without padding.

    .. math::
        :nowrap:

            \begin{align}
                architecture & \rightarrow Conv2D + ReLU + BatchNorm + Conv2D + ReLU + BatchNorm \\
                kernel & \rightarrow (3, 3) \\
                strides & \rightarrow (1, 1) \\
                padding & \rightarrow same \\
            \end{align}

    .. image:: images/conv_2d_block.svg
      :width: 800

    Args:
        input_ (tf.Tensor): Input tensor.
        filters (int, optional): Default: 16.
        activation (str, optional): Activation function. Default: relu.
        batch_norm (str, bool): Default: True.
        kernel_initializer (str, optional): Default: he_normal.

    Returns:
        tf.Tensor: Conv2D Block
    '''
    kwargs = dict(
        filters=filters,
        kernel=(3, 3),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding='same',
        use_bias=not batch_norm,
    )

    conv_1 = tfkl.Conv2D(**kwargs)(input_)
    if batch_norm:
        conv_1 = tfkl.BatchNormalization()(conv_1)

    conv_2 = tfkl.Conv2D(**kwargs)(conv_1)
    if batch_norm:
        conv_2 = tfkl.BatchNormalization()(conv_2)

    return conv_2


def attention_gate_2d(query, skip_connection):
    # type: (tf.Tensor, tf.Tensor) -> tf.Tensor
    '''
    Attention gate for 2D inputs.
    See: https://arxiv.org/abs/1804.03999

    Args:
        query (tf.Tensor): 2D Tensor of query.
        skip_connection (tf.Tensor): 2D Tensor of features.

    Returns:
        tf.Tensor: 2D Attention Gate.
    '''
    filters = skip_connection, query.get_shape().as_list()[-1]
    kwargs = dict(
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer='he_normal',
    )
    conv_1 = tfkl.Conv2D(filters=filters, **kwargs)(skip_connection)
    conv_2 = tfkl.Conv2D(filters=filters, **kwargs)(query)
    gate = tfkl.add([conv_1, conv_2])
    gate = tfkl.Activation('relu')(gate)
    gate = tfkl.Conv2D(filters=1, **kwargs)(gate)
    gate = tfkl.Activation('sigmoid')(gate)
    output = tfkl.multiply([skip_connection, gate])
    return output


def unet(
    input_shape,
    filters=16,
    layers=9,
    classes=1,
    activation='relu',
    batch_norm=True,
    attention_gates=False,
    output_activation='sigmoid',
    kernel_initializer='he_normal',
):
    '''
    UNet model for 2D semantic segmentation.

    see: https://arxiv.org/abs/1505.04597
    see: https://arxiv.org/pdf/1411.4280.pdf
    see: https://arxiv.org/abs/1804.03999

    Args:
        input_shape (tf.Tensor, optional): Tensor of shape (width, height,
            channels).
        filters (int, optional): Number of filters for initial con 2d block.
            Default: 16.
        layers (int, optional): Total number of layers. Default: 9.
        classes (int, optional): Number of output classes. Default: 1.
        activation (tf.Tensor, optional): Activation function to be used.
            Default: relu.
        batch_norm (tf.Tensor, optional): Use batch normalization.
            Default: True.
        attention_gates (tf.Tensor, optional): Use attention gates.
            Default: False.
        output_activation (tf.Tensor, optional): Output activation function.
            Default: sigmoid.
        kernel_initializer (tf.Tensor, optional): Default: he_normal.

    Raises:
        EnforceError: If layers is not an odd integer greater than 2.

    Returns:
        tfkm.Model: UNet model.
    '''
    msg = 'Layers must be an odd integer greater than 2. Given value: {a}.'
    Enforce(layers, 'instance of', int, message=msg)
    Enforce(layers, '>=', 3, message=msg)
    Enforce(layers % 2 == 1, '==', True, message=msg)
    # --------------------------------------------------------------------------

    n = (layers - 1) / 2
    down_layers = []

    # input layer
    input_ = tfkl.Input(input_shape)

    # down layers
    x = input_
    for i in range(n):
        # conv backend of layer
        x = conv_2d_block(
            input_=x,
            filters=filters,
            batch_norm=batch_norm,
            activation=activation,
            kernel_initializer=kernel_initializer,
        )
        down_layers.append(x)

        # downsample
        x = tfkl.MaxPooling2D((2, 2))(x)
        filters *= 2

    # middle layer
    x = conv_2d_block(
        input_=x,
        filters=filters,
        batch_norm=batch_norm,
        activation=activation,
        kernel_initializer=kernel_initializer,
    )

    # up layers
    up_layers = list(reversed(down_layers))
    for layer in up_layers[:n]:
        filters /= 2

        # upsample
        x = tfkl.Conv2DTranspose(
            filters, (2, 2), strides=(2, 2), padding='same'
        )(x)

        # attention gate
        if attention_gates:
            x = tfkl.concatenate([attention_gate_2d(x, layer), x])
        else:
            x = tfkl.concatenate([layer, x])

        # conv backend of layer
        x = conv_2d_block(
            input_=x,
            filters=filters,
            batch_norm=batch_norm,
            activation=activation,
            kernel_initializer=kernel_initializer,
        )

    output = tfkl.Conv2D(classes, (1, 1), activation=output_activation)(x)
    model = tfkm.Model(inputs=[input_], outputs=[output])
    return model


def get_config():
    return dict(
        model=dict(
            input_shape=(208, 208, 3),
            classes=1,
            activation='leaky_relu',
            batch_norm=True,
            upsample_mode='deconv',
            attention_gates=True,
            filters=64,
            layers=4,
            output_activation='sigmoid',
        ),
        split=dict(
            test_size=0.1,
            random_state=42,
        ),
        batch_size=32,
        # optimizer=dict(
        #     learning_rate=0.015,
        #     momentum=0.99,
        # ),
        compile_params=dict(
            optimizer=tfko.SGD(**dict(
                learning_rate=0.015,
                momentum=0.99,
            )),
            loss=ficl.jaccard_loss,
            metrics=[
                ficm.jaccard, ficm.dice, ficm.intersection_over_union
            ],
        ),
        preprocess=dict(
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='constant',
        ),
        fit=dict(
            verbose='auto',
            epochs=50,
            callbacks=[fict.get_callbacks()],
            validation_data=(x_test, y_test),
            shuffle=True,
            initial_epoch=0,
            steps_per_epoch=np.ceil(x_train.shape[0] / batch_size),
            use_multiprocessing=True,
        )
    )


def setup(x, y, model, config):
    # type: (np.ndarray, np.ndarray, tfkm.Model, dict) -> dict
    '''
    '''
    # train test split
    x_train, x_test, y_train, y_test = skm \
        .train_test_split(x, y, **config['split'])
    if config['input_shape'] != x_train.shape[1:]:
        raise ValueError('Bad input shape')

    # preprocessing
    data = tfkpp.ImageDataGenerator(**config['preprocess'])
    data.fit(x_train)

    # compile model
    model.compile(**config['compile'])

    output = dict(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        model=model,
        data=data,
    )
    return output
