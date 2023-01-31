from typing import Union  # noqa F401
import schematics.models as scm  # noqa F401

from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
from pathlib import Path
import math

import yaml
import tensorflow.keras.optimizer as tfko

from flatiron.core.dataset import Dataset
from flatiron.core.dataset_config import DatasetConfig
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

    def _validate(self, config, spec_class):
        # type: (dict, scm.Model) -> dict
        '''
        Validate given config with given specification class.

        Args:
            config (dict): Config to be validated.
            spec_class (scm.Model): Specifiction class.

        Returns:
            dict: Completed config.
        '''
        output = spec_class(config)
        output.validate()
        return output.to_native()

    def __init__(self, config):
        # type: (dict) -> None
        '''
        PipelineBase is a base class for machine learning pipelines.

        Args:
            config (dict): PipelineBase config.
        '''
        config = deepcopy(config)
        config['dataset'] = self._validate(config['dataset'], DatasetConfig)
        config['model'] = self._validate(config['model'], self.model_config)
        # config['train'] = self._validate(config['train'], TrainConfig)
        self.config = config

        src = config['dataset']['source']
        if Path(src).is_file():
            self.dataset = Dataset.read_csv(src)
        else:
            self.dataset = Dataset.read_directory(src)

    def load(self):
        config = self.config['dataset']
        self.dataset.load(
            limit=config['load_limit'],
            shuffle=config['load_shuffle'],
        )
        return self

    def train_test_split(self):
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
        self.dataset.unload()
        return self

    @abstractproperty
    def model_config(self):
        # type: () -> scm.Model
        '''
        Returns:
            scm.Model: Model config class.
        '''
        pass

    @abstractmethod
    def build(self):
        # type: () -> PipelineBase
        '''
        Build machine learning model and assign it to self.model.

        Returns:
            PipelineBase: Self.
        '''
        pass

    def compile(self):
        config = self.config['compile']
        opt = tfko.get(config['optimizer'], **config['optimizer_params'])
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics,
            loss_weights=None,
            weighted_metrics=None,
            run_eagerly=None,
            steps_per_execution=None,
            jit_compile=None,
        )
        return self

    def fit(self):
        config = self.config['fit']
        temp = fict.get_tensorboard_project(
            config['project'],
            config['root'],
            config['timezone'],
        )
        callbacks = fict.get_callbacks(
            temp['log_directory'],
            temp['checkpoint_pattern'],
            config['checkpoint_params'],
        )
        self.model.fit(
            metric=config['metric'],
            mode=config['mode'],
            save_best_only=config['save_best_only'],
            save_freq=config['save_freq'],
            update_freq=config['update_freq'],
            steps_per_epoch=math.ceil(self.x_train.shape[0] / config['batch_size']),
            validation_data=(self.x_test, self.y_test),
            callbacks=callbacks,
        )
        return self
