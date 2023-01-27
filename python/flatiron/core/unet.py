# from pathlib import Path
# import json
import math
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
# import sklearn.model_selection as skms
# import tensorflow_addons as tfa
# import tensorflow.keras.backend as tfkb
# import tensorflow.keras.optimizers as tfko
# import tensorflow.keras.models as tfkm

from lunchbox.enforce import Enforce
import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow.keras.models as tfkm

# import flatiron.core.loss as ficl
# import flatiron.core.tools as fict
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


# def get_config():
#     return dict(
#         input_shape=(208, 208, 3),
#         classes=1,
#         activation='leaky_relu',
#         batch_norm=True,
#         upsample_mode='deconv',
#         dropout=0.0,
#         dropout_change_per_layer=0.0,
#         dropout_type='spatial',
#         use_dropout_on_upsampling=False,  # best False
#         attention_gates=True,               # best False
#         filters=64,                       # best 64       above 64 exceeds single GPU memory
#         layers=4,                     # best 4
#         output_activation='sigmoid',      # best sigmoid
#     )

# params = {}

# split_params = dict(
#     test_size=0.1,
#     random_state=42,
# )
# params.update(split_params)
# x_train, x_test, y_train, y_test = skms.train_test_split(x, y, **split_params)
# x_train.shape

# model_params = get_config()
# if model_params['input_shape'] != x_train.shape[1:]:
#     raise ValueError('Bad input shape')

# params.update(model_params)

# opt_params = dict(
#     learning_rate=0.015,               # best 0.01
#     momentum=0.99,                    # best 0.99
# )
# optimizer = tfko.SGD(**opt_params)     # best
# # optimizer = tfko.Adam(**opt_params)
# params.update(opt_params)
# # loss = 'binary_crossentropy'
# # loss = jaccards_loss
# loss = ficl.dice_loss
# compile_params = dict(
#     optimizer=optimizer,
#     loss=loss,  # best jaccard
#     metrics=[unmet.iou, unmet.iou_thresholded, unmet.dice_coef, unmet.jaccard_coef],
# )
# params.update(compile_params)

# model = unet(**model_params)
# model.compile(**compile_params)

# batch_size = 32   # best 32
# fit_params = dict(
#     verbose='auto',
#     epochs=50,
#     callbacks=[fict.get_callbacks()],
#     validation_data=(x_test, y_test),
#     shuffle=True,
#     initial_epoch=0,
#     steps_per_epoch=np.ceil(x_train.shape[0] / batch_size),
#     # validation_split=0.0,
#     # sample_weight=None,
#     # validation_steps=None,
#     # validation_batch_size=None,
#     # validation_freq=1,
#     # max_queue_size=10,
#     # workers=14,
#     use_multiprocessing=True,
# )
# params.update(fit_params)
# del params['validation_data']

# datagen_params = dict(
#     # featurewise_center=True,
#     # featurewise_std_normalization=True,
#     # rotation_range=10,
#     # width_shift_range=0.1,
#     # height_shift_range=0.1,
#     # zoom_range=0.15,
#     # shear_range=0.1,
#     horizontal_flip=True,
#     vertical_flip=True,
#     fill_mode='constant',
# )
# datagen = keras_unet.utils.get_augmented(x_train, y_train, batch_size=batch_size, data_gen_args=datagen_params)

# params.update(datagen_params)
# params['batch_size'] = batch_size
# # datagen = ImageDataGenerator(**datagen_params)
# # datagen.fit(x_train)

# for k, v in params.items():
#     if type(v) in [bool, float, int, str]:
#         continue
#     if isinstance(v, tuple):
#         params[k] = list(v)
#     if isinstance(v, list):
#         params[k] = list(map(str, v))
#     else:
#         params[k] = str(v)
# slack_it('start training', params=params)

# model.fit(datagen, **fit_params)


# src = '/mnt/storage/tensorboard/date-2021-10-11_time-16-35-11/models/date-2021-10-11_time-16-35-11_epoch-050'
# # mdl = tf.keras.models.load_model(
# #     src,
# #     custom_objects=dict(
# #         jaccards_loss=jaccards_loss,
# #         dice_loss=dice_loss,
# #         iou=unmet.iou,
# #         iou_thresholded=unmet.iou_thresholded,
# #         dice_coef=unmet.dice_coef,
# #         jaccard_coef=unmet.jaccard_coef,
# #         get_config=get_config,
# #         loss=dice_loss,
# #     )
# # )
