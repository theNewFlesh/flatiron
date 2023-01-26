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
# import sklearn.model_selection as skms
# import tensorflow_addons as tfa
# import tensorflow.keras.backend as tfkb
# import tensorflow.keras.optimizers as tfko
# import tensorflow.keras.models as tfkm

import tensorflow as tf
import tensorflow.keras.layers as tfkl

# import flatiron.core.loss as ficl
# import flatiron.core.tools as fict
# ------------------------------------------------------------------------------


def upsample_conv():
    pass


def upsample_simple():
    pass


def downsample_conv2d(
    input_,
    filters=16,
    activation='relu',
    batch_norm=True,
    kernel_initializer='he_normal',
):
    # type: (tf.Tensor, int, str, bool, str) -> tf.Tensor
    r'''
    Downsample 2D Convolution block.

    .. math::
        :nowrap:

            \begin{align}
                architecture & \rightarrow Conv2D + ReLU + BatchNorm + Conv2D + ReLU + BatchNorm \\
                kernel & \rightarrow (3, 3) \\
                strides & \rightarrow (1, 1) \\
                padding & \rightarrow same \\
            \end{align}

    .. image:: images/downsample_conv2d.svg
      :width: 400

    Args:
        input_ (tf.Tensor): Input tensor.
        filters (int, optional): Default: 16.
        activation (str, optional): Activation function. Default: relu.
        batch_norm (str, bool): Default: True.
        kernel_initializer (str, optional): Default: he_normal.

    Returns:
        tf.Tensor: Conv2D Block
    '''
    kernel = (3, 3)
    strides = (1, 1)
    padding = 'same'
    bias = not batch_norm

    output = tfkl.Conv2D(
        filters,
        kernel,
        strides=strides,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=bias,
    )(input_)

    if batch_norm:
        output = tfkl.BatchNormalization()(output)

    output = tfkl.Conv2D(
        filters,
        kernel,
        strides=strides,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=bias,
    )(output)

    if batch_norm:
        output = tfkl.BatchNormalization()(output)
    return output


# def attention_concat():
#     pass


# def concatenate():
#     pass


# def unet(
#     input_shape,
#     num_classes=1,
#     activation='relu',
#     use_batch_norm=True,
#     upsample_mode='deconv',  # 'deconv' or 'simple'
#     dropout=0.0,
#     dropout_change_per_layer=0.0,
#     dropout_type='spatial',
#     use_dropout_on_upsampling=False,
#     use_attention=False,
#     filters=16,
#     num_layers=4,
#     output_activation='sigmoid',
# ):  # 'sigmoid' or 'softmax'
#     """
#     Customisable UNet architecture (Ronneberger et al. 2015 [1]).
#     Arguments:
#     input_shape: 3D Tensor of shape (x, y, num_channels)
#     num_classes (int): Unique classes in the output mask. Should be set to 1 for binary segmentation
#     activation (str): A keras.activations.Activation to use. ReLu by default.
#     use_batch_norm (bool): Whether to use Batch Normalisation across the channel axis between convolutional layers
#     upsample_mode (one of "deconv" or "simple"): Whether to use transposed convolutions or simple upsampling in the decoder part
#     dropout (float between 0. and 1.): Amount of dropout after the initial convolutional block. Set to 0. to turn Dropout off
#     dropout_change_per_layer (float between 0. and 1.): Factor to add to the Dropout after each convolutional block
#     dropout_type (one of "spatial" or "standard"): Type of Dropout to apply. Spatial is recommended for CNNs [2]
#     use_dropout_on_upsampling (bool): Whether to use dropout in the decoder part of the network
#     use_attention (bool): Whether to use an attention dynamic when concatenating with the skip-connection, implemented as proposed by Oktay et al. [3]
#     filters (int): Convolutional filters in the initial convolutional block. Will be doubled every block
#     num_layers (int): Number of total layers in the encoder not including the bottleneck layer
#     output_activation (str): A keras.activations.Activation to use. Sigmoid by default for binary segmentation
#     Returns:
#     model (keras.models.Model): The built U-Net
#     Raises:
#     ValueError: If dropout_type is not one of "spatial" or "standard"
#     [1]: https://arxiv.org/abs/1505.04597
#     [2]: https://arxiv.org/pdf/1411.4280.pdf
#     [3]: https://arxiv.org/abs/1804.03999
#     """
#     down_layers = []
#     n = math.ceil(num_layers / 2)

#     if upsample_mode == "deconv":
#         upsample = upsample_conv
#     else:
#         upsample = upsample_simple

#     # Build U-Net model
#     inputs = tfkl.Input(input_shape)
#     x = inputs

#     for i in range(n):
#         x = downsample_conv2d(
#             inputs=x,
#             filters=filters,
#             use_batch_norm=use_batch_norm,
#             dropout=dropout,
#             dropout_type=dropout_type,
#             activation=activation,
#         )
#         down_layers.append(x)
#         x = tfkl.MaxPooling2D((2, 2))(x)
#         dropout += dropout_change_per_layer
#         filters *= 2 # double the number of filters with each layer

#     for i in range(n):
#         x = downsample_conv2d(
#             inputs=x,
#             filters=filters,
#             use_batch_norm=use_batch_norm,
#             dropout=dropout,
#             dropout_type=dropout_type,
#             activation=activation,
#         )
#         down_layers.append(x)
#         x = tfkl.MaxPooling2D((2, 2))(x)
#         dropout += dropout_change_per_layer
#         filters *= 2 # double the number of filters with each layer

#     x = downsample_conv2d(
#         inputs=x,
#         filters=filters,
#         use_batch_norm=use_batch_norm,
#         dropout=dropout,
#         dropout_type=dropout_type,
#         activation=activation,
#     )

#     if not use_dropout_on_upsampling:
#         dropout = 0.0
#         dropout_change_per_layer = 0.0

#     up_layers = list(reversed(down_layers))
#     for layer in up_layers[:n]:
#         filters //= 2  # decreasing number of filters with each layer
#         dropout -= dropout_change_per_layer
#         x = upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
#         if use_attention:
#             x = attention_concat(conv_below=x, skip_connection=layer)
#         else:
#             x = concatenate([x, layer])
#         x = downsample_conv2d(
#             inputs=x,
#             filters=filters,
#             use_batch_norm=use_batch_norm,
#             dropout=dropout,
#             dropout_type=dropout_type,
#             activation=activation,
#         )

#     for layer in up_layers[n:]:
#         filters //= 2  # decreasing number of filters with each layer
#         dropout -= dropout_change_per_layer
#         x = upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
#         if use_attention:
#             x = attention_concat(conv_below=x, skip_connection=layer)
#         else:
#             x = concatenate([x, layer])
#         x = downsample_conv2d(
#             inputs=x,
#             filters=filters,
#             use_batch_norm=use_batch_norm,
#             dropout=dropout,
#             dropout_type=dropout_type,
#             activation=activation,
#         )

#     outputs = tfkl.Conv2D(num_classes, (1, 1), activation=output_activation)(x)
#     model = tfkm.Model(inputs=[inputs], outputs=[outputs])
#     return model


# def get_config():
#     return dict(
#         input_shape=(208, 208, 3),
#         num_classes=1,
#         activation='leaky_relu',
#         use_batch_norm=True,
#         upsample_mode='deconv',
#         dropout=0.0,
#         dropout_change_per_layer=0.0,
#         dropout_type='spatial',
#         use_dropout_on_upsampling=False,  # best False
#         use_attention=True,               # best False
#         filters=64,                       # best 64       above 64 exceeds single GPU memory
#         num_layers=4,                     # best 4
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
# # model = unmod.custom_unet(**model_params)
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
