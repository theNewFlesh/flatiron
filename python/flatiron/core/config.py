from typing import Optional, Union
from typing_extensions import Annotated
from flatiron.core.types import OptLabels, OptInt, OptFloat

import pydantic as pyd

import flatiron.core.validators as vd
# ------------------------------------------------------------------------------


class BaseConfig(pyd.BaseModel):
    model_config = pyd.ConfigDict(extra='forbid')


class DatasetConfig(BaseConfig):
    '''
    Configuration for Dataset.

    See: https://thenewflesh.github.io/flatiron/core.html#module-flatiron.core.dataset

    Attributes:
        source (str): Dataset directory or CSV filepath.
        ext_regex (str, optional): File extension pattern.
                Default: 'npy|exr|png|jpeg|jpg|tiff'.
        labels (object, optional): Label channels. Default: None.
        label_axis (int, optional): Label axis. Default: -1.
        test_size (float, optional): Test set size as a proportion.
            Default: 0.2.
        limit (str or int): Limit data by number of samples.
            Default: None.
        reshape (bool, optional): Reshape concatenated data to incorpate frames
            as the first dimension: (FRAME, ...). Analogous to the first
            dimension being batch. Default: True.
        shuffle (bool, optional): Randomize data before splitting.
            Default: True.
        seed (int, optional): Shuffle seed number. Default: None.
    '''
    source: str
    ext_regex: str = 'npy|exr|png|jpeg|jpg|tiff'
    labels: OptLabels = None
    label_axis: int = -1
    test_size: Optional[Annotated[float, pyd.Field(ge=0)]] = 0.2
    limit: Optional[Annotated[int, pyd.Field(ge=0)]] = None
    reshape: bool = True
    shuffle: bool = True
    seed: Optional[int] = None


class OptimizerConfig(BaseConfig):
    '''
    Configuration for keras optimizer.

    See: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer

    Attributes:
        name (string, optional): Name of optimizer. Default='sgd'.
        learning_rate (float, optional): Learning rate. Default=0.001.
        loss_scale_factor (OptFloat, optional): Will be multiply the loss before
            computing gradients. Default: None.
        gradient_accumulation_steps (OptInt, optional): Update model and
            optimizer at this frequency. Default: None.
        global_clipnorm (float, optional): Clip all weights so norm is not
            higher than this. Default: None.
        clipnorm (float, optional): Clip individual weights so norm is not
            higher than this. Default: None.
        clipvalue (float, optional): Clip weights at this max value.
            Default: None
        use_ema (bool, optional): Exponential moving average. Default=False.
        ema_momentum (float, optional): Exponential moving average momentum.
            Default=0.99.
        ema_overwrite_frequency (int, optional): Frequency of EMA overwrites.
            Default: None.
        sgd_momentum (float, optional): Momentum. Default=0.
        sgd_nesterov (bool, optional): User Nesterov updates. Default=False.
        adam_epsilon (float, optional): A small constant for numerical stability.
            Default: 1e-07
        adam_amsgrad (bool, optional): Whether to apply AMSGrad variant.
            Default: False.
        adam_beta_1 (float, optional): The exponential decay rate for the 1st moment
            estimates. Default: 0.9
        adam_beta_2 (float, optional): The exponential decay rate for the 2nd moment
            estimates. Default: 0.999
    '''
    name: str = 'sgd'
    learning_rate: float = 0.001
    loss_scale_factor: OptFloat = None
    gradient_accumulation_steps: OptInt = None
    global_clipnorm: OptFloat = None
    clipnorm: OptFloat = None
    clipvalue: OptFloat = None
    use_ema: bool = False
    ema_momentum: float = 0.99
    ema_overwrite_frequency: OptInt = None
    sgd_momentum: float = 0.0
    sgd_nesterov: bool = False
    adam_epsilon: float = 1e-07
    adam_amsgrad: bool = False
    adam_beta_1: float = 0.9
    adam_beta_2: float = 0.999


class CompileConfig(BaseConfig):
    '''
    Configuration for calls to model.compile.

    See: https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile

    Attributes:
        loss (string): Loss metric name.
        metrics (list[str], optional): List of metrics. Default: [].
        device (str, optional): Hardware device. Default: 'gpu'.
        tf_loss_weights (list[float], optional): List of loss weights.
            Default: None.
        tf_weighted_metrics (list[float], optional): List of metric weights.
            Default: None.
        tf_run_eagerly (bool, optional): Leave as False. Default: False.
        tf_steps_per_execution (int, optional): Number of batches per function
            call. Default: 1.
        tf_jit_compile (bool, optional): Use XLA. Default: False.
        tf_auto_scale_loss (bool, optional): Model dtype is mixed_float16 when
            True. Default: True.
    '''
    loss: str
    metrics: list[str] = []
    device: str = 'gpu'
    tf_loss_weights: Optional[list[float]] = None
    tf_weighted_metrics: Optional[list[float]] = None
    tf_run_eagerly: bool = False
    tf_steps_per_execution: int = 1
    tf_jit_compile: bool = False
    tf_auto_scale_loss: bool = True


class CallbacksConfig(BaseConfig):
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
    initial_value_threshold: OptFloat = None


class TrainConfig(BaseConfig):
    '''
    Configuration for calls to model train function.

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


class LoggerConfig(BaseConfig):
    '''
    Configuration for logger.

    See: https://thenewflesh.github.io/flatiron/core.html#module-flatiron.core.logging

    Attributes:
        slack_channel (str, optional): Slack channel name. Default: None.
        slack_url (str, optional): Slack URL name. Default: None.
        slack_methods (list[str], optional): Pipeline methods to be logged to
            Slack. Default: [load, compile, train].
        timezone (str, optional): Timezone. Default: UTC.
        level (str or int, optional): Log level. Default: warn.
    '''
    slack_channel: Optional[str] = None
    slack_url: Optional[str] = None
    slack_methods: list[str] = pyd.Field(default=['load', 'compile', 'train'])
    timezone: str = 'UTC'
    level: str = 'warn'

    @pyd.field_validator('slack_methods')
    def _validate_slack_methods(cls, value):
        for item in value:
            vd.is_pipeline_method(item)
        return value


class PipelineConfig(BaseConfig):
    '''
    Configuration for PipelineBase classes.

    See: https://thenewflesh.github.io/flatiron/core.html#module-flatiron.core.pipeline

    Attributes:
        dataset (dict): Dataset configuration.
        optimizer (dict): Optimizer configuration.
        compile (dict): Compile configuration.
        callbacks (dict): Callbacks configuration.
        engine (str): Deep learning framework.
        logger (dict): Logger configuration.
        train (dict): Train configuration.
    '''
    dataset: DatasetConfig
    optimizer: OptimizerConfig
    compile: CompileConfig
    callbacks: CallbacksConfig
    engine: Annotated[str, pyd.AfterValidator(vd.is_engine)]
    logger: LoggerConfig
    train: TrainConfig
