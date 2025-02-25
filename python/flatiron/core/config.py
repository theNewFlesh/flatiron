from typing import Optional, Union
from typing_extensions import Annotated

from flatiron.core.types import OptLabels, OptFloat, Getter

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


class FrameworkConfig(pyd.BaseModel):
    name: Annotated[str, pyd.AfterValidator(vd.is_engine)] = 'tensorflow'
    device: str = 'cpu'


class OptimizerConfig(pyd.BaseModel):
    '''
    Configuration for optimizer.

    See: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer

    Attributes:
        name (string, optional): Name of optimizer. Default='SGD'.
    '''
    name: str = 'SGD'


class LossConfig(pyd.BaseModel):
    '''
    Configuration for loss.

    Attributes:
        name (string, optional): Name of loss. Default='MeanSquaredError'.
    '''
    name: str = 'MeanSquaredError'


class CallbacksConfig(BaseConfig):
    '''
    Configuration for callbacks.

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
        seed (int, optional): Seed value. Default: 42.
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
    seed: int = 42
    shuffle: bool = True
    initial_epoch: int = 1
    validation_freq: int = 1


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
        framework (dict): Deep learning framework config.
        dataset (dict): Dataset configuration.
        optimizer (dict): Optimizer configuration.
        loss (dict): Loss configuration.
        metrics (list[dict], optional): Metric dicts. Default=[dict(name='Mean')].
        compile (dict): Compile configuration.
        callbacks (dict): Callbacks configuration.
        logger (dict): Logger configuration.
        train (dict): Train configuration.
    '''
    framework: FrameworkConfig
    dataset: DatasetConfig
    optimizer: OptimizerConfig
    loss: LossConfig
    metrics: list[Getter] = [dict(name='Mean')]
    callbacks: CallbacksConfig
    logger: LoggerConfig
    train: TrainConfig

    @pyd.field_validator('metrics')
    def _validate_metrics(cls, items):
        for item in items:
            if 'name' not in item.keys():
                msg = f'All dicts must contain name key. Given value: {item}.'
                raise ValueError(msg)
        return items
