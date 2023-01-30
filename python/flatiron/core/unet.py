from keras.engine.keras_tensor import KerasTensor

from lunchbox.enforce import Enforce
import numpy as np
import sklearn.model_selection as skm
import tensorflow as tf
import tensorflow.keras.layers as tfl
import tensorflow.keras.models as tfm
import tensorflow.keras.optimizers as tfo
import tensorflow.keras.preprocessing.image as tfpp

import flatiron.core.loss as ficl
import flatiron.core.metric as ficm
import flatiron.core.tools as fict
# ------------------------------------------------------------------------------


PAD = 18


def conv_2d_block(
    input_,  # type: KerasTensor
    filters=16,  # type: int
    activation='relu',  # type: str
    batch_norm=True,  # type: bool
    kernel_initializer='he_normal',  # type: str
    name='conv-2d-block',  # type: str
):
    # type: (...) -> KerasTensor
    r'''
    2D Convolution block without padding.

    .. math::
        :nowrap:

            \begin{align}
                architecture & \rightarrow Conv2D + ReLU + BatchNorm + Conv2D
                + ReLU + BatchNorm \\
                kernel & \rightarrow (3, 3) \\
                strides & \rightarrow (1, 1) \\
                padding & \rightarrow same \\
            \end{align}

    .. image:: images/conv_2d_block.svg
      :width: 800

    Args:
        input_ (KerasTensor): Input tensor.
        filters (int, optional): Default: 16.
        activation (str, optional): Activation function. Default: relu.
        batch_norm (str, bool): Default: True.
        kernel_initializer (str, optional): Default: he_normal.
        name (str, optional): Layer name. Default: conv-2d-block

    Returns:
        KerasTensor: Conv2D Block
    '''
    name = fict.pad_layer_name(name, length=PAD)
    kwargs = dict(
        filters=filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding='same',
        use_bias=not batch_norm,
    )

    name2 = f'{name}-1'
    conv_1 = tfl.Conv2D(**kwargs, name=f'{name}-0')(input_)
    if batch_norm:
        conv_1 = tfl.BatchNormalization(name=f'{name}-1')(conv_1)
        name2 = f'{name}-2'

    conv_2 = tfl.Conv2D(**kwargs, name=name2)(conv_1)
    if batch_norm:
        conv_2 = tfl.BatchNormalization(name=f'{name}-3')(conv_2)

    return conv_2


def attention_gate_2d(
    query,  # type: KerasTensor
    skip_connection,  # type: KerasTensor
    activation_1='relu',  # type: str
    activation_2='sigmoid',  # type: str
    kernel_size=1,  # type: int
    strides=1,  # type: int
    padding='same',  # type: str
    kernel_initializer='he_normal',  # type: str
    name='attention-gate',  # type: str
):
    # type: (...) -> KerasTensor
    '''
    Attention gate for 2D inputs.
    See: https://arxiv.org/abs/1804.03999

    Args:
        query (KerasTensor): 2D Tensor of query.
        skip_connection (KerasTensor): 2D Tensor of features.
        activation_1 (str, optional): First activation. Default: 'relu'
        activation_2 (str, optional): Second activation. Default: 'sigmoid'
        kernel_size (int, optional): Kernel_size. Default: 1
        strides (int, optional): Strides. Default: 1
        padding (str, optional): Padding. Default: 'same'
        kernel_initializer (str, optional): Kernel initializer.
            Default: 'he_normal'
        name (str, optional): Layer name. Default: attention-gate

    Returns:
        KerasTensor: 2D Attention Gate.
    '''
    name = fict.pad_layer_name(name, length=PAD)
    filters = query.get_shape().as_list()[-1]
    kwargs = dict(
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
    )
    conv_0 = tfl.Conv2D(
        filters=filters, **kwargs, name=f'{name}-0'
    )(skip_connection)
    conv_1 = tfl.Conv2D(filters=filters, **kwargs, name=f'{name}-1')(query)
    gate = tfl.add([conv_0, conv_1], name=f'{name}-2')
    gate = tfl.Activation(activation_1, name=f'{name}-3')(gate)
    gate = tfl.Conv2D(filters=1, **kwargs, name=f'{name}-4')(gate)
    gate = tfl.Activation(activation_2, name=f'{name}-5')(gate)
    gate = tfl.multiply([skip_connection, gate], name=f'{name}-6')
    output = tfl.concatenate([gate, query], name=f'{name}-7')
    return output


def get_unet_model(
    input_shape,  # type: tuple[int]
    classes=1,  # type: int
    filters=16,  # type: int
    layers=9,  # type: int
    activation='relu',  # type: str
    batch_norm=True,  # type: bool
    output_activation='sigmoid',  # type: str
    kernel_initializer='he_normal',  # type: str
    attention_gates=False,  # type: bool
    attention_activation_1='relu',  # type: str
    attention_activation_2='sigmoid',  # type: str
    attention_kernel_size=1,  # type: int
    attention_strides=1,  # type: int
    attention_padding='same',  # type: str
    attention_kernel_initializer='he_normal',  # type: str
):
    # type: (...) -> tfm.Model
    '''
    UNet model for 2D semantic segmentation.

    see: https://arxiv.org/abs/1505.04597
    see: https://arxiv.org/pdf/1411.4280.pdf
    see: https://arxiv.org/abs/1804.03999

    Args:
        input_shape (KerasTensor, optional): Tensor of shape (width, height,
            channels).
        classes (int, optional): Number of output classes. Default: 1.
        filters (int, optional): Number of filters for initial con 2d block.
            Default: 16.
        layers (int, optional): Total number of layers. Default: 9.
        activation (KerasTensor, optional): Activation function to be used.
            Default: relu.
        batch_norm (KerasTensor, optional): Use batch normalization.
            Default: True.
        output_activation (KerasTensor, optional): Output activation function.
            Default: sigmoid.
        kernel_initializer (KerasTensor, optional): Default: he_normal.
        attention_gates (KerasTensor, optional): Use attention gates.
            Default: False.
        attention_activation_1 (str, optional): First activation.
            Default: 'relu'
        attention_activation_2 (str, optional): Second activation.
            Default: 'sigmoid'
        attention_kernel_size (int, optional): Kernel_size. Default: 1
        attention_strides (int, optional): Strides. Default: 1
        attention_padding (str, optional): Padding. Default: 'same'
        attention_kernel_initializer (str, optional): Kernel initializer.
            Default: 'he_normal'

    Raises:
        EnforceError: If layers is not an odd integer greater than 2.

    Returns:
        tfm.Model: UNet model.
    '''
    msg = 'Layers must be an odd integer greater than 2. Given value: {a}.'
    Enforce(layers, 'instance of', int, message=msg)
    Enforce(layers, '>=', 3, message=msg)
    Enforce(layers % 2 == 1, '==', True, message=msg)
    # --------------------------------------------------------------------------

    n = int((layers - 1) / 2)
    encode_layers = []

    # input layer
    input_ = tfl.Input(input_shape, name='input')

    # encode layers
    x = input_
    for i in range(n):
        # conv backend of layer
        x = conv_2d_block(
            input_=x,
            filters=filters,
            batch_norm=batch_norm,
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f'encode-block_{i:02d}',
        )
        encode_layers.append(x)

        # downsample
        name = fict.pad_layer_name(f'downsample_{i:02d}', length=PAD)
        x = tfl.MaxPooling2D((2, 2), name=name)(x)
        filters *= 2

    # middle layer
    name = fict.pad_layer_name('middle-block_00', length=PAD)
    x = conv_2d_block(
        input_=x,
        filters=filters,
        batch_norm=batch_norm,
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name,
    )

    # decode layers
    decode_layers = list(reversed(encode_layers))
    for i, layer in enumerate(decode_layers[:n]):
        filters = int(filters / 2)

        # upsample
        name = fict.pad_layer_name(f'upsample_{i:02d}', length=PAD)
        x = tfl.Conv2DTranspose(
            filters=filters,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding='same',
            name=name,
        )(x)

        # attention gate
        if attention_gates:
            name = fict.pad_layer_name(f'attention-gate_{i:02d}', length=PAD)
            x = attention_gate_2d(
                x,
                layer,
                activation_1=attention_activation_1,
                activation_2=attention_activation_2,
                kernel_size=attention_kernel_size,
                strides=attention_strides,
                padding=attention_padding,
                kernel_initializer=attention_kernel_initializer,
                name=name,
            )
        else:
            name = fict.pad_layer_name(f'concat_{i:02d}', length=PAD)
            x = tfl.concatenate([layer, x], name=name)

        # conv backend of layer
        x = conv_2d_block(
            input_=x,
            filters=filters,
            batch_norm=batch_norm,
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f'decode-block_{i:02d}',
        )

    output = tfl.Conv2D(
        classes, (1, 1), activation=output_activation, name='output'
    )(x)
    model = tfm.Model(inputs=[input_], outputs=[output])
    return model
