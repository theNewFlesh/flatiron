import schematics as scm
import schematics.types as scmt

import flatiron.core.validators as vd
# ------------------------------------------------------------------------------


class DatasetConfig(scm.Model):
    '''
    Configuration for Dataset.

    See: https://thenewflesh.github.io/flatiron/core.html#module-flatiron.core.dataset

    Attributes:
        source (str): Dataset directory or CSV filepath.
        load_limit (str or int): Limit data by number of samples or memory size.
            Default: None.
        load_shuffle (bool): Shuffle chunks before loading. Default: False.
        split_index (int): Index of axis to split on.
        split_axis (int): Axis to split data on. Default: -1.
        split_test_size (float): Test size. Default: 0.2
        split_train_size (float): Train size. Default: None
        split_random_state (int): Seed for shuffling randomness. Default: 42.
        split_shuffle (bool): Shuffle data rows. Default: True.
    '''
    source = scmt.StringType(required=True)
    load_limit = scmt.UnionType([scmt.IntType, scmt.StringType], default=None)
    load_shuffle = scmt.BooleanType(required=True, default=False)
    split_index = scmt.IntType(required=True)
    split_axis = scmt.IntType(required=True, default=-1)
    split_test_size = scmt.FloatType(
        required=True, default=0.2, validators=[lambda x: vd.is_gte(x, 0)]
    )
    split_train_size = scmt.FloatType(validators=[lambda x: vd.is_gte(x, 0)])
    split_random_state = scmt.IntType(required=True, default=42)
    split_shuffle = scmt.BooleanType(required=True, default=True)


class OptimizerConfig(scm.Model):
    '''
    Configuration for keras optimizer.

    See: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer

    Attributes:
        name: (string, optional): Name of optimizer. Default='sgd'.
        learning_rate: (float, optional): Learning rate. Default=0.001.
        momentum: (float, optional): Momentum. Default=0.
        nesterov: (boolean, optional): User Nesterov updates. Default=False.
        weight_decay: (string, optional): Decay weights. Default: None.
        clipnorm: (float, optional): Clip individual weights so norm is not
            higher than this. Default: None.
        clipvalue: (float, optional): Clip weights at this max value.
            Default: None
        global_clipnorm: (float, optional): Clip all weights so norm is not
            higher than this. Default: None.
        use_ema: (boolean, optional): Exponential moving average. Default=False.
        ema_momentum: (float, optional): Exponential moving average momentum.
            Default=0.99.
        ema_overwrite_frequency: (int, optional): Frequency of EMA overwrites.
            Default: None.
        jit_compile: (boolean, optional): Use XLA. Default=True.
    '''
    name = scmt.StringType(default='sgd')
    learning_rate = scmt.FloatType(default=0.001)
    momentum = scmt.FloatType(default=0)
    nesterov = scmt.BooleanType(default=False)
    weight_decay = scmt.FloatType(default=0)
    clipnorm = scmt.FloatType(default=None)
    clipvalue = scmt.FloatType(default=None)
    global_clipnorm = scmt.FloatType(default=None)
    use_ema = scmt.BooleanType(default=False)
    ema_momentum = scmt.FloatType(default=0.99)
    ema_overwrite_frequency = scmt.IntType(default=None)
    jit_compile = scmt.BooleanType(default=True)


class CompileConfig(scm.Model):
    '''
    Configuration for calls to model.compile.

    See: https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile

    Attributes:
        loss (string): Loss metric name.
        metrics (list[str], optional): List of metrics. Default: [].
        loss_weights (list[float], optional): List of loss weights.
            Default: None.
        weighted_metrics (list[float], optional): List of metric weights.
            Default: None.
        run_eagerly (boolean, optional): Leave as False. Default: False.
        steps_per_execution (int, optional): Number of batches per function
            call. Default: 1.
        jit_compile (boolean, optional): Use XLA. Default: False.
    '''
    loss = scmt.StringType(required=True)
    metrics = scmt.ListType(scmt.StringType, default=[])
    loss_weights = scmt.ListType(scmt.FloatType, default=None)
    weighted_metrics = scmt.ListType(scmt.FloatType, default=None)
    run_eagerly = scmt.BooleanType(default=False)
    steps_per_execution = scmt.IntType(default=1)
    jit_compile = scmt.BooleanType(default=False)


class CallbacksConfig(scm.Model):
    '''
    Configuration for tensorflow callbacks.

    See: https://thenewflesh.github.io/flatiron/core.html#module-flatiron.core.tools
    See: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint

    Attributes:
        project (str): Name of project.
        root (str or Path): Tensorboard parent directory. Default: /mnt/storage.
        monitor (str, optional): Metric to monitor. Default: 'val_loss'.
        verbose (int, optional): Log callback actions. Default: 0.
        save_best_only (bool, optional): Save only best model. Default: False.
        mode (str, optional): Overwrite best model via
            `mode(old metric, new metric)`. Options: [auto, min, max].
            Default: 'auto'.
        save_weights_only (bool, optional): Only save model weights.
            Default: False.
        save_freq (union, optional): Save after each epoch or N batches.
            Options: 'epoch' or int. Default: 'epoch'.
        initial_value_threshold (float, optional): Initial best value of metric.
            Default: None.
        experimental_io_device (str, optional): IO device name.
            Default: None.
        experimental_enable_async_checkpoint (bool, optional): Use async
            checkpoint. Default: False.
    '''
    project = scmt.StringType(required=True)
    root = scmt.StringType(required=True)
    monitor = scmt.StringType(default='val_loss')
    verbose = scmt.IntType(default=0)
    save_best_only = scmt.BooleanType(default=False)
    save_weights_only = scmt.BooleanType(default=False)
    mode = scmt.StringType(default='auto', validators=[vd.is_callback_mode])
    save_freq = scmt.UnionType([scmt.StringType, scmt.IntType], default='epoch')
    initial_value_threshold = scmt.FloatType()
    experimental_io_device = scmt.StringType()
    experimental_enable_async_checkpoint = scmt.BooleanType(default=False)


class FitConfig(scm.Model):
    '''
    Configuration for calls to model.fit.

    See: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit

    Attributes:
        batch_size (int, optional): Number of samples per update. Default: 32.
        epochs (int, optional): Number of epochs to train model. Default: 30.
        verbose (str or int, optional): Verbosity of model logging.
            Options: 'auto', 0, 1, 2.
            0 is silent. 1 is progress bar. 2 is one line per epoch.
            Auto is usually 1. Default: auto.
        validation_split (float, optional): Fraction of training data to use for
            validation. Default: 0.
        shuffle (bool, optional): Shuffle training data per epoch.
            Default: True.
        initial_epoch (int, optional): Epoch at which to start training
            (useful for resuming a previous training run). Default: 1.
        validation_freq (int, optional): Number of training epochs before new
            validation. Default: 1.
        max_queue_size (int, optional): Max size of generator queue.
            Default: 10.
        workers (int, optional): Max processes used by generator. Default: 1.
        use_multiprocessing (bool, optional): Use multiprocessing for
            generators. Default: False.
    '''
    batch_size = scmt.IntType(default=32)
    epochs = scmt.IntType(default=30)
    verbose = scmt.UnionType([scmt.StringType, scmt.IntType], default='auto')
    validation_split = scmt.FloatType(default=0.0)
    shuffle = scmt.BooleanType(default=True)
    initial_epoch = scmt.IntType(default=1)
    validation_freq = scmt.IntType(default=1)
    max_queue_size = scmt.IntType(default=10)
    workers = scmt.IntType(default=1)
    use_multiprocessing = scmt.BooleanType(default=False)
    # class_weight
    # sample_weight
    # initial_epoch
    # validation_steps
    # validation_batch_size


class LoggerConfig(scm.Model):
    '''
    Configuration for logger.

    See: https://thenewflesh.github.io/flatiron/core.html#module-flatiron.core.logging

    Attributes:
        slack_channel (str, optional): Slack channel name. Default: None.
        slack_url (str, optional): Slack URL name. Default: None.
        slack_methods (list[str], optional): Pipeline methods to be logged to
            Slack. Default: [load, compile, fit].
        timezone (str, optional): Timezone. Default: UTC.
        level (str or int, optional): Log level. Default: warn.
    '''
    slack_channel = scmt.StringType(default=None)
    slack_url = scmt.StringType(default=None)
    slack_methods = scmt.ListType(
        scmt.StringType(validators=[vd.is_pipeline_method]),
        default=['load', 'compile', 'fit']
    )
    timezone = scmt.StringType(default='UTC')
    level = scmt.StringType(default='warn')


class PreprocessConfig(scm.Model):
    '''
    Base class for PreprocessConfig classes.

    Attributes:
        name (str, optional): Name of preprocess function. Default: identity.
    '''
    name = scmt.StringType(required=True, default='identity')


class PipelineConfig(scm.Model):
    '''
    Configuration for PipelineBase classes.

    See: https://thenewflesh.github.io/flatiron/core.html#module-flatiron.core.pipeline

    Attributes:
        dataset (dict): Dataset configuration.
        optimizer (dict): Optimizer configuration.
        compile (dict): Compile configuration.
        callbacks (dict): Callbacks configuration.
        fit (dict): Fit configuration.
        logger (dict): Logger configuration.
    '''
    dataset = scmt.ModelType(DatasetConfig, required=True)
    preprocess = scmt.ModelType(PreprocessConfig, required=True)
    optimizer = scmt.ModelType(OptimizerConfig, required=True)
    compile = scmt.ModelType(CompileConfig, required=True)
    callbacks = scmt.ModelType(CallbacksConfig, required=True)
    fit = scmt.ModelType(FitConfig, required=True)
    logger = scmt.ModelType(LoggerConfig, required=True)


class ImagePreprocessConfig(PreprocessConfig):
    '''
    Config for parameters passed to ImageDataGenerator.

    See: https://thenewflesh.github.io/flatiron/core.html#module-flatiron.core.preprocess

    Attributes:
        name (str): Name of preprocess function.
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
