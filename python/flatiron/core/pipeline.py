from typing import Union  # noqa F401
import keras.engine.functional as kef  # noqa F401
import schematics.models as scm  # noqa F401

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
import math

import yaml
import tensorflow.keras.optimizers as tfko

from flatiron.core.dataset import Dataset
import flatiron.core.config as cfg
import flatiron.core.logging as filog
import flatiron.core.loss as ficl
import flatiron.core.metric as ficm
import flatiron.core.tools as fict

Filepath = Union[str, Path]
# ------------------------------------------------------------------------------


class PipelineBase(ABC):
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

        # model
        model = config.get('model', {})
        model = self.model_config()(model)
        model.validate()
        model = model.to_native()
        del config['model']

        # pipeline
        config = cfg.PipelineConfig(config)
        config.validate()
        config = config.to_native()
        config['model'] = model
        self.config = config

        # create Dataset instance
        src = config['dataset']['source']
        if Path(src).is_file():
            self.dataset = Dataset.read_csv(src)
        else:
            self.dataset = Dataset.read_directory(src)

    def load(self):
        # type: () -> PipelineBase
        '''
        Load dataset into memory.
        Calls `self.dataset.load` with dataset params.

        Returns:
            PipelineBase: Self.
        '''
        config = self.config['dataset']
        with filog.SlackLogger(
            'LOAD DATASET', dict(dataset=config), **self.config['logger']
        ):
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

    def unload(self):
        # type: () -> PipelineBase
        '''
        Unload dataset into memory. Train and test sets will be kept.
        Calls `self.dataset.unload`.

        Returns:
            PipelineBase: Self.
        '''
        self.dataset.unload()
        return self

    def build(self):
        # type: () -> PipelineBase
        '''
        Build machine learning model and assign it to self.model.
        Calls `self.model_func` with model params.

        Returns:
            PipelineBase: Self.
        '''
        self.model = self.model_func(**self.config['model'])
        return self

    def compile(self):
        # type: () -> PipelineBase
        '''
        Call `self.model.compile` with compile params.

        Returns:
            PipelineBase: Self.
        '''
        comp = self.config['compile']

        # get loss and metrics from flatiron modules
        loss = ficl.FUNCTIONS[comp['loss']]
        metrics = [ficm.FUNCTIONS[x] for x in comp['metrics']]

        # create optimizer
        kwargs = deepcopy(self.config['optimizer'])
        del kwargs['name']
        opt = tfko.get(self.config['optimizer']['name'], **kwargs)

        # compile
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics,
            loss_weights=comp['loss_weights'],
            weighted_metrics=comp['weighted_metrics'],
            run_eagerly=comp['run_eagerly'],
            steps_per_execution=comp['steps_per_execution'],
            jit_compile=comp['jit_compile'],
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

        # create tensorboard
        tb = fict.get_tensorboard_project(
            cb['project'],
            cb['root'],
            cb['timezone'],
        )

        # create checkpoint params and callbacks
        cp = deepcopy(cb)
        del cp['project']
        del cp['root']
        callbacks = fict.get_callbacks(
            tb['log_directory'], tb['checkpoint_pattern'], cp,
        )

        # train model
        steps = math.ceil(self.x_train.shape[0] / fit['batch_size'])
        self.model.fit(
            x=self.x_train,
            y=self.x_test,
            callbacks=callbacks,
            validation_data=(self.x_test, self.y_test),
            steps_per_epoch=steps,
            **fit,
        )
        return self

    @abstractmethod
    def model_config(self):
        # type: () -> scm.Model
        '''
        Subclasses of PipelineBase wiil need to define a config class for models
        created in the build method.

        Returns:
            scm.Model: Model config class.
        '''
        pass

    @abstractmethod
    def model_func(self):
        # type: () -> kef.Functional
        '''
        Subclasses of PipelineBase need to define a function that builds and
        returns a machine learning model.

        Returns:
            kef.Functional: Machine learning model.
        '''
        pass
