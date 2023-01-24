
from collections import Counter
import json
import re
import random
from datetime import datetime
from pathlib import Path
import shutil
import os
import warnings
import math
import pytz

os.environ['LD_LIBRARY_PATH'] = '/home/ubuntu/.local/lib/python3.7/site-packages'

from lunchbox.stopwatch import StopWatch
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import hidebound.core.tools as hbt
import keras_unet.metrics as unmet
import keras_unet.models as unmod
import lunchbox.tools as lbt
import pandas as pd
import sklearn.model_selection as skms
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as tfb
import tensorflow.keras.optimizers as opt
import keras_unet
import numpy as np
import cufflinks as cf
cf.go_offline()

from vision.image.image import Image
from vision.image.sequence import ImageSequence
import vision.image.sequence_tools as sqt
import vision.image.image_tools as imt
from vision.image.image import BitDepth
from vision.image.color import BasicColor
# ------------------------------------------------------------------------------


def get_callbacks():
    root = Path('/mnt/storage', 'tensorboard')
    os.makedirs(root, exist_ok=True)

    timestamp = datetime \
        .now(tz=pytz.timezone('America/Los_Angeles')) \
        .strftime('date-%Y-%m-%d_time-%H-%M-%S')
    log_dir = Path(root, timestamp).as_posix()
    os.makedirs(log_dir, exist_ok=True)

    target = Path(log_dir, 'models')
    os.makedirs(target, exist_ok=True)
    target = Path(target, timestamp + '_epoch-{epoch:03d}').as_posix()

    callbacks = [
        TensorBoard(log_dir=log_dir, histogram_freq=1),
        tf.keras.callbacks.ModelCheckpoint(
            target,
            metric='val_iou',
            mode='auto',
            save_freq='epoch',
            update_freq='batch',
            write_steps_per_second=True,
            write_images=True,
        )
    ]
    return callbacks


def dset001_to_arrays(source, class_balance=0.2, binarize=False):
    data = pd.read_hdf(source, 'metadata')
    mask = data.mean_alpha.cumsum() / data.index
    mask = mask >= class_balance
    data = data[mask]

    if len(data) < 1000:
        msg = 'Class balance too low'
        raise ValueError(msg)

    rgb = list('rgb')
    data['content'] = data.filepath.apply(Image.read)

    seq = ImageSequence.from_images(data.content.tolist())
    x = seq[:, :, :, rgb].to_array()
    y = seq[:, :, :, 'a'].to_array()[..., np.newaxis]

    if binarize:
        y = np.where(y >= 0.5, np.float16(1.0), np.float16(0.0))

    return x, y

params = {}
ver = 6
src = f'/mnt/storage/projects/data001/assets/dset001/p-data001_s-dset001_d-set0169-ground-truth-ds01_v{ver:03d}/p-data001_s-dset001_d-set0169-ground-truth-ds01_v{ver:03d}.hdf'
conform_params = dict(
    source=src,
    class_balance=0.15,  # best 0.15
)
params.update(conform_params)

stopwatch = StopWatch()
stopwatch.start()
x, y = dset001_to_arrays(**conform_params)
stopwatch.stop()
# slack_it('dset001_to_arrays', params['source'], '', params, stopwatch)

split_params = dict(
    test_size=0.1,
    random_state=42,
)
params.update(split_params)
x_train, x_test, y_train, y_test = skms.train_test_split(x, y, **split_params)
x_train.shape

from keras_unet.models.custom_unet import *


def custom_unet_four_gpu(
    input_shape,
    num_classes=1,
    activation='relu',
    use_batch_norm=True,
    upsample_mode='deconv',  # 'deconv' or 'simple'
    dropout=0.0,
    dropout_change_per_layer=0.0,
    dropout_type='spatial',
    use_dropout_on_upsampling=False,
    use_attention=False,
    filters=16,
    num_layers=4,
    output_activation='sigmoid',
):  # 'sigmoid' or 'softmax'

    \"\"\"
    Customisable UNet architecture (Ronneberger et al. 2015 [1]).
    Arguments:
    input_shape: 3D Tensor of shape (x, y, num_channels)
    num_classes (int): Unique classes in the output mask. Should be set to 1 for binary segmentation
    activation (str): A keras.activations.Activation to use. ReLu by default.
    use_batch_norm (bool): Whether to use Batch Normalisation across the channel axis between convolutional layers
    upsample_mode (one of \"deconv\" or \"simple\"): Whether to use transposed convolutions or simple upsampling in the decoder part
    dropout (float between 0. and 1.): Amount of dropout after the initial convolutional block. Set to 0. to turn Dropout off
    dropout_change_per_layer (float between 0. and 1.): Factor to add to the Dropout after each convolutional block
    dropout_type (one of \"spatial\" or \"standard\"): Type of Dropout to apply. Spatial is recommended for CNNs [2]
    use_dropout_on_upsampling (bool): Whether to use dropout in the decoder part of the network
    use_attention (bool): Whether to use an attention dynamic when concatenating with the skip-connection, implemented as proposed by Oktay et al. [3]
    filters (int): Convolutional filters in the initial convolutional block. Will be doubled every block
    num_layers (int): Number of total layers in the encoder not including the bottleneck layer
    output_activation (str): A keras.activations.Activation to use. Sigmoid by default for binary segmentation
    Returns:
    model (keras.models.Model): The built U-Net
    Raises:
    ValueError: If dropout_type is not one of \"spatial\" or \"standard\"
    [1]: https://arxiv.org/abs/1505.04597
    [2]: https://arxiv.org/pdf/1411.4280.pdf
    [3]: https://arxiv.org/abs/1804.03999
    \"\"\"
    down_layers = []
    n = math.ceil(num_layers / 2)
    # with tf.device(tf.DeviceSpec(job='localhost', device_type='GPU', device_index=0)):
    with tf.device('/device:gpu:0'):
        if upsample_mode == \"deconv\":
            upsample = upsample_conv
        else:
            upsample = upsample_simple

        # Build U-Net model
        inputs = Input(input_shape)
        x = inputs

        for i in range(n):
            x = conv2d_block(
                inputs=x,
                filters=filters,
                use_batch_norm=use_batch_norm,
                dropout=dropout,
                dropout_type=dropout_type,
                activation=activation,
            )
            down_layers.append(x)
            x = MaxPooling2D((2, 2))(x)
            dropout += dropout_change_per_layer
            filters *= 2 # double the number of filters with each layer

    # with tf.device(tf.DeviceSpec(job='localhost', device_type='GPU', device_index=1)):
    with tf.device('/device:gpu:1'):
        for i in range(n):
            x = conv2d_block(
                inputs=x,
                filters=filters,
                use_batch_norm=use_batch_norm,
                dropout=dropout,
                dropout_type=dropout_type,
                activation=activation,
            )
            down_layers.append(x)
            x = MaxPooling2D((2, 2))(x)
            dropout += dropout_change_per_layer
            filters *= 2 # double the number of filters with each layer

        x = conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            activation=activation,
        )

    # with tf.device(tf.DeviceSpec(job='localhost', device_type='GPU', device_index=2)):
    with tf.device('/device:gpu:2'):
        if not use_dropout_on_upsampling:
            dropout = 0.0
            dropout_change_per_layer = 0.0

        up_layers = list(reversed(down_layers))
        for layer in up_layers[:n]:
            filters //= 2  # decreasing number of filters with each layer
            dropout -= dropout_change_per_layer
            x = upsample(filters, (2, 2), strides=(2, 2), padding=\"same\")(x)
            if use_attention:
                x = attention_concat(conv_below=x, skip_connection=layer)
            else:
                x = concatenate([x, layer])
            x = conv2d_block(
                inputs=x,
                filters=filters,
                use_batch_norm=use_batch_norm,
                dropout=dropout,
                dropout_type=dropout_type,
                activation=activation,
            )

    # with tf.device(tf.DeviceSpec(job='localhost', device_type='GPU', device_index=3)):
    with tf.device('/device:gpu:3'):
        for layer in up_layers[n:]:
            filters //= 2  # decreasing number of filters with each layer
            dropout -= dropout_change_per_layer
            x = upsample(filters, (2, 2), strides=(2, 2), padding=\"same\")(x)
            if use_attention:
                x = attention_concat(conv_below=x, skip_connection=layer)
            else:
                x = concatenate([x, layer])
            x = conv2d_block(
                inputs=x,
                filters=filters,
                use_batch_norm=use_batch_norm,
                dropout=dropout,
                dropout_type=dropout_type,
                activation=activation,
            )

        outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)
        model = Model(inputs=[inputs], outputs=[outputs])
    return model


def get_config():
    return dict(
        input_shape=(208, 208, 3),
        num_classes=1,
        activation='leaky_relu',
        use_batch_norm=True,
        upsample_mode='deconv',
        dropout=0.0,
        dropout_change_per_layer=0.0,
        dropout_type='spatial',
        use_dropout_on_upsampling=False,  # best False
        use_attention=True,               # best False
        filters=64,                       # best 64       above 64 exceeds single GPU memory
        num_layers=4,                     # best 4
        output_activation='sigmoid',      # best sigmoid
    )

model_params = get_config()
if model_params['input_shape'] != x_train.shape[1:]:
    raise ValueError('Bad input shape')

params.update(model_params)

opt_params = dict(
    learning_rate=0.015,               # best 0.01
    momentum=0.99,                    # best 0.99
)
optimizer = opt.SGD(**opt_params)     # best
# optimizer = opt.Adam(**opt_params)
params.update(opt_params)
# loss = 'binary_crossentropy'
# loss = jaccard_distance_loss
loss = dice_loss
compile_params = dict(
    optimizer=optimizer,
    loss=loss,  # best jaccard
    metrics=[unmet.iou, unmet.iou_thresholded, unmet.dice_coef, unmet.jaccard_coef],
)
params.update(compile_params)

model = custom_unet_four_gpu(**model_params)
# model = unmod.custom_unet(**model_params)
model.compile(**compile_params)

batch_size = 32   # best 32
fit_params = dict(
    verbose='auto',
    epochs=50,
    callbacks=[get_callbacks()],
    validation_data=(x_test, y_test),
    shuffle=True,
    initial_epoch=0,
    steps_per_epoch=np.ceil(x_train.shape[0] / batch_size),
    # validation_split=0.0,
    # sample_weight=None,
    # validation_steps=None,
    # validation_batch_size=None,
    # validation_freq=1,
    # max_queue_size=10,
    # workers=14,
    use_multiprocessing=True,
)
params.update(fit_params)
del params['validation_data']

datagen_params = dict(
    # featurewise_center=True,
    # featurewise_std_normalization=True,
    # rotation_range=10,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # zoom_range=0.15,
    # shear_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='constant',
)
datagen = keras_unet.utils.get_augmented(x_train, y_train, batch_size=batch_size, data_gen_args=datagen_params)

params.update(datagen_params)
params['batch_size'] = batch_size
# datagen = ImageDataGenerator(**datagen_params)
# datagen.fit(x_train)

def get_datagen_contact_sheet(datagen):
    sample = next(datagen)
    xs = []
    ys= []
    for i in range(sample[0].shape[0]):
        x = Image.from_array(sample[0][i])
        y = Image.from_array(sample[1][i])
        xs.append(x)
        ys.append(y)

    xs = ImageSequence.from_images(xs)
    xs = sqt.get_contact_sheet(xs, width=1)
    ys = ImageSequence.from_images(ys)
    ys = sqt.get_contact_sheet(ys, width=1)
    cs = imt.staple(xs, ys)
    return cs

# get_datagen_contact_sheet(datagen)

# src = '/mnt/storage/tensorboard/date-2021-10-11_time-15-27-55/models/date-2021-10-11_time-15-27-55_epoch-012'
# model = tf.keras.models.load_model(
#     src,
#     custom_objects=dict(
#         jaccard_distance_loss=jaccard_distance_loss,
#         dice_loss=dice_loss,
#         iou=unmet.iou,
#         iou_thresholded=unmet.iou_thresholded,
#         dice_coef=unmet.dice_coef,
#         jaccard_coef=unmet.jaccard_coef,
#         get_config=get_config,
#     )
# )

for k, v in params.items():
    if type(v) in [bool, float, int, str]:
        continue
    if isinstance(v, tuple):
        params[k] = list(v)
    if isinstance(v, list):
        params[k] = list(map(str, v))
    else:
        params[k] = str(v)
slack_it('start training', params=params)

model.fit(datagen, **fit_params)

# clear GPU memory
from numba import cuda
for device in cuda.devices.gpus.lst:
    device.reset()

def floor_mult(a, b, floor=0.25):
    return Image.from_array(a * np.clip(b + floor, 0, 1))

src = '/mnt/storage/tensorboard/date-2021-10-11_time-16-35-11/models/date-2021-10-11_time-16-35-11_epoch-050'
# mdl = tf.keras.models.load_model(
#     src,
#     custom_objects=dict(
#         jaccard_distance_loss=jaccard_distance_loss,
#         dice_loss=dice_loss,
#         iou=unmet.iou,
#         iou_thresholded=unmet.iou_thresholded,
#         dice_coef=unmet.dice_coef,
#         jaccard_coef=unmet.jaccard_coef,
#         get_config=get_config,
#         loss=dice_loss,
#     )
# )

floor = 0.5
index = random.randint(0, x.shape[0])
yt = floor_mult(x[index], y[index], floor=floor)
yhat_a = mdl.predict(x[index][np.newaxis, ...])[0]
yhat = floor_mult(x[index], yhat_a, floor=floor)

y_a = y[index]
y_a = np.concatenate([y_a] * 3, axis=2)
y_a = Image.from_array(y_a)
yhat_a = np.concatenate([yhat_a] * 3, axis=2)
yhat_a = Image.from_array(yhat_a)

imgs = [yt, yhat, y_a, yhat_a]
imgs = [img.to_bit_depth(BitDepth.FLOAT16) for img in imgs]
imgs = ImageSequence.from_images(imgs)
sqt.get_contact_sheet(imgs, width=2, gap=5, color=BasicColor.BLUE2)

src = '/mnt/storage/tensorboard/date-2021-10-01_time-14-19-53/models/date-2021-10-01_time-14-19-53_epoch-050'
mdl = tf.keras.models.model_from_config(
    json.dumps(get_config()), custom_objects=dict(
        jaccard_distance_loss=jaccard_distance_loss,
        dice_loss=dice_loss,
    )
)

import hidebound.core.tools as hbt

foo = '/mnt/storage/tensorboard'
foo = hbt.directory_to_dataframe(foo)
foo['timestamp'] = foo.filepath.apply(lambda x: Path(*Path(x).parts[:5]).name)
regex = re.compile('e(\d\d)')
mask = foo.filename.apply(regex.search).astype(bool)
foo = foo[mask]
foo['epoch'] = foo.filename.apply(lambda x: regex.search(x).group(1)).astype(int)
foo = foo.groupby('timestamp', as_index=False).max()
foo['target'] = foo.filepath \
    .apply(lambda x: re.sub('weights-e', 'epoch-', x)) \
    .apply(lambda x: re.sub('weights', 'models', x)) \
    .apply(lambda x: os.path.splitext(x)[0])

keys = foo.filepath.tolist()
vals = foo.target.tolist()
for src, tgt in zip(keys, vals):
    print(src, tgt)
    mdl = tf.keras.models.load_model(src, custom_objects=dict(iou=unmet.iou, iou_thresholded=unmet.iou_thresholded, jaccard_distance_loss=jaccard_distance_loss))
    break
    mdl.save(tgt, save_format='tf', custom_objects)
