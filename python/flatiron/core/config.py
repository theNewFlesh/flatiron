from typing import Optional, Union
from typing_extensions import Annotated

import pydantic as pyd

import flatiron.core.validators as vd
# ------------------------------------------------------------------------------


class DatasetConfig(pyd.BaseModel):
    '''
    Configuration for Dataset.

    See: https://thenewflesh.github.io/flatiron/core.html#module-flatiron.core.dataset

    Attributes:
        source (str): Dataset directory or CSV filepath.
        load_limit (str or int): Limit data by number of samples or memory size.
            Default: None.
        load_shuffle (bool, optional): Shuffle chunks before loading.
            Default: False.
        split_index (int, optional): Index of axis to split on.
        split_axis (int): Axis to split data on. Default: -1.
        split_test_size (float, optional): Test size. Default: 0.2
        split_train_size (float, optional): Train size. Default: None
        split_random_state (int, optional): Seed for shuffling randomness.
            Default: 42.
        split_shuffle (bool, optional): Shuffle data rows. Default: True.
    '''
    source: str
    load_limit: Optional[Union[int, str]] = None
    load_shuffle: bool = False
    split_index: int
    split_axis: int = -1
    split_test_size: Optional[Annotated[float, pyd.Field(ge=0)]] = 0.2
    split_train_size: Optional[Annotated[float, pyd.Field(ge=0)]] = None
    split_random_state: Optional[int] = 42
    split_shuffle: bool = True


class OptimizerConfig(pyd.BaseModel):
    '''
    Configuration for keras optimizer.

    See: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer

    Attributes:
        class_name: (string, optional): Name of optimizer. Default='sgd'.
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
    class_name: str = 'sgd'
    learning_rate: float = 0.001
    momentum: float = 0
    nesterov: bool = False
    weight_decay: float = 0
    clipnorm: Optional[float] = None
    clipvalue: Optional[float] = None
    global_clipnorm: Optional[float] = None
    use_ema: bool = False
    ema_momentum: float = 0.99
    ema_overwrite_frequency: Optional[int] = None
    jit_compile: bool = True


class CompileConfig(pyd.BaseModel):
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
    loss: str
    metrics: list[str] = []
    loss_weights: Optional[list[float]] = None
    weighted_metrics: Optional[list[float]] = None
    run_eagerly: bool = False
    steps_per_execution: int = 1
    jit_compile: bool = False


class CallbacksConfig(pyd.BaseModel):
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
    '''
    project: str
    root: str
    monitor: str = 'val_loss'
    verbose: int = 0
    save_best_only: bool = False
    save_weights_only: bool = False
    mode: Annotated[str, pyd.AfterValidator(vd.is_callback_mode)] = 'auto'
    save_freq: Union[str, int] = 'epoch'
    initial_value_threshold: Optional[float] = None


class FitConfig(pyd.BaseModel):
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
    '''
    batch_size: int = 32
    epochs: int = 30
    verbose: Union[str, int] = 'auto'
    validation_split: float = 0.0
    shuffle: bool = True
    initial_epoch: int = 1
    validation_freq: int = 1
    # callbacks
    # class_weight
    # initial_epoch
    # sample_weight
    # steps_per_epoch
    # validation_batch_size
    # validation_data
    # validation_steps


class LoggerConfig(pyd.BaseModel):
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
    slack_channel: Optional[str] = None
    slack_url: Optional[str] = None
    slack_methods: list[str] = pyd.Field(default=['load', 'compile', 'fit'])
    timezone: str = 'UTC'
    level: str = 'warn'

    @pyd.field_validator('slack_methods')
    def _validate_slack_methods(cls, value):
        for item in value:
            vd.is_pipeline_method(item)
        return value


class PipelineConfig(pyd.BaseModel):
    '''
    Configuration for PipelineBase classes.

    See: https://thenewflesh.github.io/flatiron/core.html#module-flatiron.core.pipeline

    Attributes:
        dataset (dict): Dataset configuration.
        optimizer (dict): Optimizer configuration.
        compile (dict): Compile configuration.
        callbacks (dict): Callbacks configuration.
        engine (str): Deep learning framework.
        fit (dict): Fit configuration.
        logger (dict): Logger configuration.
    '''
    dataset: DatasetConfig
    optimizer: OptimizerConfig
    compile: CompileConfig
    callbacks: CallbacksConfig
    engine: str
    fit: FitConfig
    logger: LoggerConfig

    @pyd.field_validator('engine')
    def _validate_engine(cls, value):
        vd.is_engine(value)
        return value
