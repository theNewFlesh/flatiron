import schematics as scm
import schematics.types as scmt

import flatiron.core.validators as vd
# ------------------------------------------------------------------------------


class DatasetConfig(scm.Model):
    '''
    Configuration for Dataset.

    Attributes:
        source (str): Dataset directory or CSV filepath.
        limit (str or int): Limit data by number of samples or memory size.
            Default: None.
        shuffle (bool): Shuffle chunks before loading. Default: False.
        index (int): Index of axis to split on.
        axis (int): Axis to split data on. Default: -1.
        test_size (float): Test size. Default: 0.2
        train_size (float): Train size. Default: None
        random_state (int): Seed for shuffling randomness.
        shuffle (bool): Shuffle data rows. Default: True.
    '''
    source = scmt.StringType(required=True)
    load_limit = scmt.UnionType(
        (scmt.IntType, scmt.StringType), serialize_when_none=True
    )
    load_shuffle = scmt.BooleanType(required=True, default=False)
    split_index = scmt.IntType(required=True)
    split_axis = scmt.IntType(required=True, default=-1)
    split_test_size = scmt.FloatType(
        required=True, default=0.2, validators=[lambda x: vd.is_gte(x, 0)]
    )
    split_train_size = scmt.FloatType(
        serialize_when_none=True, validators=[lambda x: vd.is_gte(x, 0)]
    )
    split_random_state = scmt.IntType(required=True, default=42)
    split_shuffle = scmt.BooleanType(required=True, default=True)


class OptimizerConfig(scm.Model):
    optimizer = scmt.StringType(default='sgd')
    learning_rate = scmt.FloatType(default=0.001)
    momentum = scmt.FloatType(default=0)
    nesterov = scmt.BooleanType(default=False)
    accumulator_name = scmt.StringType()
    weight_decay = scmt.StringType(default=None)
    clipnorm = scmt.FloatType()
    clipvalue = scmt.FloatType()
    global_clipnorm = scmt.FloatType()
    use_ema = scmt.BooleanType(default=False)
    ema_momentum = scmt.FloatType(default=0.99)
    ema_overwrite_frequency = scmt.IntType(default=None)
    jit_compile = scmt.BooleanType(default=True)


class CompileConfig(scm.Model):
    loss = scmt.StringType(required=True)
    metrics = scmt.ListType(scmt.StringType, required=True)
    loss_weights = scmt.ListType(scmt.FloatType, serialize_when_none=True)
    weighted_metrics = scmt.ListType(scmt.FloatType, serialize_when_none=True)
    steps_per_execution = scmt.IntType(required=True, default=1)
    jit_compile = scmt.BooleanType(default=False)


class CallbacksConfig(scm.Model):
    project = scmt.StringType(required=True)
    root = scmt.StringType(required=True)
    timezone = scmt.StringType(required=True)
    monitor = scmt.StringType(required=True, default='val_loss')
    verbose = scmt.IntType(default=0)
    save_best_only = scmt.BooleanType(default=False)
    save_weights_only = scmt.BooleanType(default=False)
    mode = scmt.StringType(default='auto')
    save_freq = scmt.StringType(required=True, default='epoch')
    initial_value_threshold = scmt.FloatType()


class FitConfig(scm.Model):
    batch_size = scmt.IntType(required=True)
    epochs = scmt.IntType(required=True, default=30)
    verbose = scmt.StringType(default='auto')
    validation_split = scmt.FloatType(default=0.0)
    shuffle = scmt.BooleanType(default=True)
    initial_epoch = scmt.IntType(default=1)
    validation_freq = scmt.IntType(default=1)
    max_queue_size = scmt.IntType(default=10)
    workers = scmt.IntType(default=1)
    use_multiprocessing = scmt.BooleanType(default=False)


class LoggerConfig(scm.Model):
    slack_channel = scmt.StringType(serialize_when_none=True)
    slack_url = scmt.StringType(serialize_when_none=True)
    timezone = scmt.StringType(default='UTC')
    level = scmt.StringType(default='warn')


class PipelineConfig(scm.Model):
    dataset = scmt.ModelType(DatasetConfig, required=True)
    optimizer = scmt.ModelType(OptimizerConfig, required=True)
    compile = scmt.ModelType(CompileConfig, required=True)
    callbacks = scmt.ModelType(CallbacksConfig, required=True)
    fit = scmt.ModelType(FitConfig, required=True)
    logger = scmt.ModelType(LoggerConfig, required=True)
