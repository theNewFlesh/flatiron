import schematics as scm
import schematics.types as scmt
# ------------------------------------------------------------------------------


class ImagePreprocessConfig(scm.Model):
    '''
    Config for parameters passed to ImageDataGenerator.

    Attributes:
        featurewise_center (bool, optional): Set input mean to 0 over the
            dataset. Default: False.
        samplewise_center (bool, optional): Set each sample mean to 0.
            Default: False.
        featurewise_std_normalization (bool, optional): Divide inputs by std of
            the dataset. Default: False.
        samplewise_std_normalization (bool, optional): Divide each input by its
            std. Default: False.
        zca_whitening (bool, optional): Apply ZCA whitening. Default: False.
        zca_epsilon (float, optional): Epsilon for ZCA whitening. Default: 1e-6.
        rotation_range (int, optional): Degree range for random rotations.
            Default: 0.
        width_shift_range (float, optional): Width shift. Default: 0.
        height_shift_range (float, optional): Height shift. Default: 0.
        brightness_range (list[float]): Brightness range. Default: None.
        shear_range (float, optional): Shear angle range. Default: 0.
        zoom_range (list[float]): Range for random zoom. Default: 0.
        channel_shift_range (float, optional): Range for random channel shifts.
            Default: 0.
        fill_mode (str, optional): Empty pixel value fill. Options include:
            constant, nearest, reflect, wrap. Default: nearest.
        cval (float, optional): Fill value. Default: 0.
        horizontal_flip (bool, optional): Randomly flip inputs horizontally.
            Default: False.
        vertical_flip (bool, optional): Randomly flip inputs vertically.
            Default: False.
        rescale (float, optional): Rescaling factor. Defaul: None.
        preprocessing_function (object, optional): Custom preprocessing
            function. Default: None.
        data_format (str, optional): Channel placement. Options include:
            channels_first, channels_last. Default: channels_last.
        validation_split (float, optional): Fraction of images reserved for
            validation. Default: 0.
        interpolation_order (int, optional): Interpolation order. Default: 1.
        dtype (object, optional): Dtype to use for the generated arrays.
            Default: None.
    '''
    featurewise_center = scmt.BooleanType(default=False)
    samplewise_center = scmt.BooleanType(default=False)
    featurewise_std_normalization = scmt.BooleanType(default=False)
    samplewise_std_normalization = scmt.BooleanType(default=False)
    zca_whitening = scmt.BooleanType(default=False)
    zca_epsilon = scmt.FloatType(default=1e-06)
    rotation_range = scmt.IntType(default=0)
    width_shift_range = scmt.FloatType(default=0.0)
    height_shift_range = scmt.FloatType(default=0.0)
    brightness_range = scmt.ListType(scmt.FloatType, default=None)
    shear_range = scmt.FloatType(default=0)
    zoom_range = scmt.ListType(scmt.FloatType, default=None)
    channel_shift_range = scmt.FloatType(default=0.0)
    fill_mode = scmt.StringType(default='nearest')
    cval = scmt.FloatType(default=0.0)
    horizontal_flip = scmt.BooleanType(default=False)
    vertical_flip = scmt.BooleanType(default=False)
    rescale = scmt.FloatType(default=None)
    # preprocessing_function = scmt.Object(default=None)
    data_format = scmt.StringType(default='channels_last')
    validation_split = scmt.FloatType(default=0.0)
    interpolation_order = scmt.IntType(default=1)
    # dtype = scmt.Object(default=None)
