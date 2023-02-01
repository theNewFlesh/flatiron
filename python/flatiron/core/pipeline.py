from typing import Union  # noqa F401
import schematics.models as scm  # noqa F401

from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
from pathlib import Path
import math

import yaml

from flatiron.core.dataset import Dataset
from flatiron.core.dataset_config import DatasetConfig
import flatiron.core.logging as ficl
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
        config['slack'] = {}
        # config['train'] = self._validate(config['train'], TrainConfig)
        self.config = config

        src = config['dataset']['source']
        if Path(src).is_file():
            self.dataset = Dataset.read_csv(src)
        else:
            self.dataset = Dataset.read_directory(src)

    def load(self):
        config = self.config['dataset']
        slack = self.config['slack']
        with ficl.SlackLogger('LOAD DATASET', config, **slack):
            self.dataset.load(
                limit=config['load_limit'],
                shuffle=config['load_shuffle'],
            )
        return self

    def train_test_split(self):
        cfg = self.config['dataset']
        x_train, x_test, y_train, y_test = self.dataset.train_test_split(
            index=cfg['split_index'],
            axis=cfg['split_axis'],
            test_size=cfg['split_test_size'],
            train_size=cfg['split_train_size'],
            random_state=cfg['split_random_state'],
            shuffle=cfg['split_shuffle'],
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
        cfg = self.config['compile']
        opt = tfko.get(cfg['optimizer'], **cfg['optimizer_params'])
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
        cfg = self.config['fit']
        temp = fict.get_tensorboard_project(
            cfg['project'],
            cfg['root'],
            cfg['timezone'],
        )
        callbacks = fict.get_callbacks(
            temp['log_directory'],
            temp['checkpoint_pattern'],
            cfg['checkpoint_params'],
        )
        steps = math.ceil(self.x_train.shape[0] / cfg['batch_size'])
        # metric=cfg['metric'],
        # mode=cfg['mode'],
        # save_best_only=cfg['save_best_only'],
        # save_freq=cfg['save_freq'],
        # update_freq=cfg['update_freq'],
        # callbacks=callbacks,
        self.model.fit(
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose='auto',
            callbacks=callbacks,
            validation_split=0.0,
            validation_data=(self.x_test, self.y_test),
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=steps,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
        )
        return self
