import schematics as scm
import schematics.types as scmt
# ------------------------------------------------------------------------------


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
    loss_weights = scmt.ListType(serialize_when_none=True)
    weighted_metrics = scmt.ListType(serialize_when_none=True)
    steps_per_execution = scmt.IntType(required=True, default=1)
    jit_compile = scmt.BoolType(default=False)


class CallbacksConfig(scm.Model):
    project = scmt.StringType(required=True)
    root = scmt.StringType(required=True)
    timezone = scmt.StringType(required=True)
    monitor = scmt.StringType(required=True, default='val_loss')
    verbose = scmt.IntType(default=0)
    save_best_only = scmt.BoolType(default=False)
    save_weights_only = scmt.BoolType(default=False)
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
