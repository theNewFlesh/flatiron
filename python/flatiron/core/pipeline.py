from typing import Any, Optional, Union  # noqa F401
import keras.engine.functional as kef  # noqa F401
import numpy as np  # noqa F401
import schematics.models as scm  # noqa F401

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
import math

import tensorflow.data as tfd
import tensorflow.keras.losses as tfl
import tensorflow.keras.metrics as tfm
import tensorflow.keras.optimizers as tfo
import yaml

from flatiron.core.dataset import Dataset
import flatiron.core.config as ficc
import flatiron.core.logging as filog
import flatiron.core.loss as ficl
import flatiron.core.metric as ficm
import flatiron.core.preprocess as ficp
import flatiron.core.tools as fict

Filepath = Union[str, Path]
# ------------------------------------------------------------------------------


class PipelineBase(ABC):
    spec = ficc.PipelineConfig

    @classmethod
    def read_yaml(cls, filepath):
        # type: (Filepath) -> PipelineBase
        '''
        Construct PipelineBase instance from given yaml file.

        Args:
            filepath (str or Path): YAML file.

        Returns:
            PipelineBase: PipelineBase instance.
        '''
        with open(filepath) as f:
            config = yaml.safe_load(f)
        return cls(config)

    @classmethod
    def from_string(cls, text):
        # type: (str) -> PipelineBase
        '''
        Construct PipelineBase instance from given YAML text.

        Args:
            text (str): YAML text.

        Returns:
            PipelineBase: PipelineBase instance.
        '''
        config = yaml.safe_load(text)
        return cls(config)

    def __init__(self, config):
        # type: (dict) -> None
        '''
        PipelineBase is a base class for machine learning pipelines.

        Args:
            config (dict): PipelineBase config.
        '''
        config = deepcopy(config)

        # pipeline
        config = self.spec(config)
        config.validate()
        config = config.to_native()
        self.config = config

        # create Dataset instance
        src = config['dataset']['source']
        if Path(src).is_file():
            self.dataset = Dataset.read_csv(src)
        else:
            self.dataset = Dataset.read_directory(src)

        self.x_train = None  # type: Optional[np.ndarray]
        self.x_test = None  # type: Optional[np.ndarray]
        self.y_train = None  # type: Optional[np.ndarray]
        self.y_test = None  # type: Optional[np.ndarray]

    def _logger(self, method, message, config):
        # type: (str, str, dict) -> filog.SlackLogger
        '''
        Retreives a logger given a message, config and slack flag.

        Args:
            method (str): Name of method calling logger.
            message (str): Log message or Slack title.
            config (dict): Config dict.

        Returns:
            ficl.SlackLogger: Configured logger instance.
        '''
        kwargs = deepcopy(self.config['logger'])
        methods = kwargs['slack_methods']
        del kwargs['slack_methods']
        logger = filog.SlackLogger(message, config, **kwargs)
        if method not in methods:
            logger._message_func = None
            logger._callback = None
        return logger

    # DATA----------------------------------------------------------------------
    def load(self):
        # type: () -> PipelineBase
        '''
        Load dataset into memory.
        Calls `self.dataset.load` with dataset params.

        Returns:
            PipelineBase: Self.
        '''
        config = self.config['dataset']
        with self._logger('load', 'LOAD DATASET', dict(dataset=config)):
            self.dataset.load(
                limit=config['load_limit'],
                shuffle=config['load_shuffle'],
            )
        return self

    def train_test_split(self):
        # type: () -> PipelineBase
        '''
        Split dataset into train and test sets.

        Assigns the following instance members:

            * x_train
            * x_test
            * y_train
            * y_test

        Returns:
            PipelineBase: Self.
        '''
        config = self.config['dataset']

        with self._logger(
            'train_test_split', 'TRAIN TEST SPLIT', dict(dataset=config)
        ):
            x_train, x_test, y_train, y_test = self.dataset.train_test_split(
                index=config['split_index'],
                axis=config['split_axis'],
                test_size=config['split_test_size'],
                train_size=config['split_train_size'],
                random_state=config['split_random_state'],
                shuffle=config['split_shuffle'],
            )
            self.x_train = x_train
            self.x_test = x_test
            self.y_train = y_train
            self.y_test = y_test
        return self

    def xy_split(self):
        # type: () -> PipelineBase
        '''
        Split dataset into x_train and y_train sets.

        Assigns the following instance members:

            * x_train
            * y_train

        Returns:
            PipelineBase: Self.
        '''
        config = self.config['dataset']

        with self._logger(
            'xy_split', 'XY SPLIT', dict(dataset=config)
        ):
            x, y = self.dataset.xy_split(
                index=config['split_index'],
                axis=config['split_axis'],
            )
            self.x_train = x
            self.y_train = y
        return self

    def convert(self):
        # type: () -> PipelineBase
        '''
        Converts train data to tensorflow Dataset instance

        Assigns the following instance members:

            * xy_train
            * x_train
            * y_train
            * _steps_per_epoch

        Returns:
            PipelineBase: Self.
        '''
        config = self.config['fit']
        batch_size = config['batch_size']
        with self._logger('convert', 'CONVERT DATASET', dict(fit=config)):
            self.xy_train = tfd.Dataset.from_tensor_slices(dict(
                x_train=self.x_train,
                y_train=self.y_train,
            )).batch(batch_size)
        n = self.x_train.shape[0]  # type: ignore
        self._steps_per_epoch = math.ceil(n / batch_size)
        self.x_train = None
        self.y_train = None
        return self

    def preprocess(self):
        # type: () -> PipelineBase
        '''
        Applies preprocessing to training data.

        Assigns the following instance members:

            * xy_train

        Returns:
            PipelineBase: Self.
        '''
        config = deepcopy(self.config['preprocess'])
        with self._logger(
            'preprocess', 'PREPROCESS DATASET', dict(preprocess=config)
        ):
            name = config.pop('name')
            func = ficp.get(name)
            self.xy_train = self.xy_train \
                .map(lambda x: func(x, **config))
        return self

    def unload(self):
        # type: () -> PipelineBase
        '''
        Unload dataset into memory. Train and test sets will be kept.
        Calls `self.dataset.unload`.

        Returns:
            PipelineBase: Self.
        '''
        config = self.config['dataset']
        with self._logger('unload', 'UNLOAD DATASET', dict(dataset=config)):
            self.dataset.unload()
        return self

    # MODEL---------------------------------------------------------------------
    def build(self):
        # type: () -> PipelineBase
        '''
        Build machine learning model and assign it to self.model.
        Calls `self.model_func` with model params.

        Returns:
            PipelineBase: Self.
        '''
        config = self.config['model']
        with self._logger('build', 'BUILD MODEL', dict(model=config)):
            self.model = self.model_func()(**config)
        return self

    def compile(self):
        # type: () -> PipelineBase
        '''
        Call `self.model.compile` with compile params.

        Returns:
            PipelineBase: Self.
        '''
        compile_ = self.config['compile']

        cfg = dict(
            model=self.config['model'],
            optimizer=self.config['optimizer'],
            compile=compile_,
        )
        with self._logger('compile', 'COMPILE MODEL', cfg):
            # loss
            loss = compile_['loss']
            try:
                loss = ficl.get(loss)
            except NotImplementedError:
                loss = tfl.get(loss)

            # metrics
            metrics = []
            for m in compile_['metrics']:
                try:
                    metric = ficm.get(m)
                except NotImplementedError:
                    metric = tfm.get(m)
                metrics.append(metric)

            # create optimizer
            kwargs = deepcopy(self.config['optimizer'])
            del kwargs['name']
            opt = tfo.get(self.config['optimizer']['name'], **kwargs)

            # compile
            self.model.compile(
                optimizer=opt,
                loss=loss,
                metrics=metrics,
                loss_weights=compile_['loss_weights'],
                weighted_metrics=compile_['weighted_metrics'],
                run_eagerly=compile_['run_eagerly'],
                steps_per_execution=compile_['steps_per_execution'],
                jit_compile=compile_['jit_compile'],
            )
        return self

    def fit(self):
        # type: () -> PipelineBase
        '''
        Call `self.model.fit` with fit params.

        Returns:
            PipelineBase: Self.
        '''
        cb = self.config['callbacks']
        fit = self.config['fit']
        log = self.config['logger']

        # log training start
        with self._logger('fit', 'TRAINING STARTED', self.config):
            pass

        # train model
        with self._logger('fit', 'TRAINING COMPLETED', self.config):
            # create tensorboard
            tb = fict.get_tensorboard_project(
                cb['project'],
                cb['root'],
                log['timezone'],
            )

            # create checkpoint params and callbacks
            cp = deepcopy(cb)
            del cp['project']
            del cp['root']
            callbacks = fict.get_callbacks(
                tb['log_dir'], tb['checkpoint_pattern'], cp,
            )
            validation_data = None
            if self.x_test is not None and self.y_test is not None:
                validation_data = (self.x_test, self.y_test)

            # train model
            self.model.fit(
                self.xy_train,
                callbacks=callbacks,
                validation_data=validation_data,
                steps_per_epoch=self._steps_per_epoch,
                **fit,
            )
        return self

    def run(self):
        # type: () -> PipelineBase
        '''
        Run the following pipeline operations:

        * load
        * train_test_split
        * unload
        * convert
        * preprocess
        * build
        * compile
        * fit

        Returns:
            PipelineBase: Self.
        '''
        self.load() \
            .train_test_split() \
            .unload() \
            .convert() \
            .preprocess() \
            .build() \
            .compile() \
            .fit()
        return self

    @abstractmethod
    def model_func(self):
        # type: () -> kef.Functional
        '''
        Subclasses of PipelineBase need to define a function that builds and
        returns a machine learning model.

        Returns:
            kef.Functional: Machine learning model.
        '''
        pass  # pragma: no cover
