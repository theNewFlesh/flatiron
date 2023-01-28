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


def conv_2d_block(
    input_,
    filters=16,
    activation='relu',
    batch_norm=True,
    kernel_initializer='he_normal',
    name='conv-2d-block',
):
    # type: (KerasTensor, int, str, bool, str, str) -> KerasTensor
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
    name = fict.pad_layer_name(name)
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


def attention_gate_2d(query, skip_connection, name='attention-gate'):
    # type: (KerasTensor, KerasTensor, str) -> KerasTensor
    '''
    Attention gate for 2D inputs.
    See: https://arxiv.org/abs/1804.03999

    Args:
        query (KerasTensor): 2D Tensor of query.
        skip_connection (KerasTensor): 2D Tensor of features.
        name (str, optional): Layer name. Default: attention-gate

    Returns:
        KerasTensor: 2D Attention Gate.
    '''
    name = fict.pad_layer_name(name)
    filters = query.get_shape().as_list()[-1]
    kwargs = dict(
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer='he_normal',
    )
    conv_0 = tfl.Conv2D(
        filters=filters, **kwargs, name=f'{name}-0'
    )(skip_connection)
    conv_1 = tfl.Conv2D(filters=filters, **kwargs, name=f'{name}-1')(query)
    gate = tfl.add([conv_0, conv_1], name=f'{name}-2')
    gate = tfl.Activation('relu', name=f'{name}-3')(gate)
    gate = tfl.Conv2D(filters=1, **kwargs, name=f'{name}-4')(gate)
    gate = tfl.Activation('sigmoid', name=f'{name}-5')(gate)
    gate = tfl.multiply([skip_connection, gate], name=f'{name}-6')
    output = tfl.concatenate([gate, query], name=f'{name}-7')
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
        input_shape (KerasTensor, optional): Tensor of shape (width, height,
            channels).
        filters (int, optional): Number of filters for initial con 2d block.
            Default: 16.
        layers (int, optional): Total number of layers. Default: 9.
        classes (int, optional): Number of output classes. Default: 1.
        activation (KerasTensor, optional): Activation function to be used.
            Default: relu.
        batch_norm (KerasTensor, optional): Use batch normalization.
            Default: True.
        attention_gates (KerasTensor, optional): Use attention gates.
            Default: False.
        output_activation (KerasTensor, optional): Output activation function.
            Default: sigmoid.
        kernel_initializer (KerasTensor, optional): Default: he_normal.

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
        name = fict.pad_layer_name(f'downsample_{i:02d}')
        x = tfl.MaxPooling2D((2, 2), name=name)(x)
        filters *= 2

    # middle layer
    name = fict.pad_layer_name('middle-block_00')
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
        filters /= 2

        # upsample
        name = fict.pad_layer_name(f'upsample_{i:02d}')
        x = tfl.Conv2DTranspose(
            filters=filters,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding='same',
            name=name,
        )(x)

        # attention gate
        if attention_gates:
            name = fict.pad_layer_name(f'attention-gate_{i:02d}')
            x = attention_gate_2d(x, layer, name=name)
        else:
            name = fict.pad_layer_name(f'concat_{i:02d}')
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


def get_config():
    params = dict(
        info=dict(
            project='proj0121',
            root='/mnt/storage/projects',
            spec='dset001',
            desc='bg-detection',
            version=4,
        ),
        data=dict(
            limit=15000,
        ),
        model=dict(
            input_shape=(192, 192, 3),
            num_classes=1,
            activation='leaky_relu',
            use_batch_norm=True,
            upsample_mode='deconv',
            use_attention=True,
            filters=32,
            num_layers=4,
            output_activation='sigmoid',
        ),
        misc=dict(
            optimizer_class=tfo.SGD,
            batch_size=16,
        ),
        optimizer=dict(
            learning_rate=0.008,
            momentum=0.99,
        ),
        compile_=dict(
            loss=ficl.jaccard_loss,
            metrics=[ficm.jaccard, ficm.dice],
        ),
        callback=dict(
            save_best_only=True,
            metric='jaccard',
            mode='auto',
            save_freq='epoch',
            update_freq='batch',
        ),
        fit=dict(
            epochs=35,
            verbose='auto',
            shuffle=True,
            initial_epoch=0,
            use_multiprocessing=True,
        ),
        datagen=dict(
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='constant',
        )
    )

    params['data']['source'] = get_info_path(**params['info'])
    opt = params['misc']['optimizer_class'](**params['optimizer'])
    params['compile_']['optimizer'] = opt
    root = params['info']['root']
    proj = params['info']['project']
    cb = params['callback']
    cbs, log_dir = fict.get_callbacks(root, proj, cb)
    params['misc']['log_dir'] = log_dir
    params['fit']['callbacks'] = [cbs]

    return params


def setup(x, y, model, config):
    # type: (np.ndarray, np.ndarray, tfm.Model, dict) -> dict
    '''
    '''
    # train test split
    x_train, x_test, y_train, y_test = skm \
        .train_test_split(x, y, **config['split'])
    if config['input_shape'] != x_train.shape[1:]:
        raise ValueError('Bad input shape')

    # preprocessing
    data = tfpp.ImageDataGenerator(**config['preprocess'])
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
