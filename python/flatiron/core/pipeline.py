from typing import Any, Optional, Type  # noqa F401
from flatiron.core.types import AnyModel, Filepath  # noqa F401
from pydantic import BaseModel  # noqa F401

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path

import yaml

from flatiron.core.dataset import Dataset
import flatiron.core.config as cfg
import flatiron.core.logging as filog
import flatiron.core.tools as fict
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
        model_config = config.pop('model', {})
        model = self.model_config() \
            .model_validate(model_config, strict=True) \
            .model_dump()

        # pipeline
        config = cfg.PipelineConfig \
            .model_validate(config, strict=True)\
            .model_dump()
        config['model'] = model
        self.config = config

        # create Dataset instance
        dconf = config['dataset']
        src = dconf['source']
        kwargs = dict(
            ext_regex=dconf['ext_regex'],
            labels=dconf['labels'],
            label_axis=dconf['label_axis'],
            calc_file_size=False,
        )
        if Path(src).is_file():
            self.dataset = Dataset.read_csv(src, **kwargs)
        else:
            self.dataset = Dataset.read_directory(src, **kwargs)

        self._train_data = None  # type: Optional[Dataset]
        self._test_data = None  # type: Optional[Dataset]

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

            * _train_data
            * _test_data

        Returns:
            PipelineBase: Self.
        '''
        config = self.config['dataset']
        with self._logger(
            'train_test_split', 'TRAIN TEST SPLIT', dict(dataset=config)
        ):
            self._train_data, self._test_data = self.dataset.train_test_split(
                test_size=config['test_size'],
                shuffle=config['shuffle'],
                seed=config['seed'],
            )
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

    @property
    def _engine(self):
        # type: () -> Any
        '''
        Uses config to retrieve flatiron engine subpackage.

        Returns:
            Any: flatiron.tf or flatiron.torch
        '''
        engine = self.config['engine']
        if engine == 'tensorflow':
            import flatiron.tf as engine
        # elif engine == 'torch':
        #     import flatiron.torch as engine
        return engine

    def compile(self):
        # type: () -> PipelineBase
        '''
        Call `self.model.compile` with compile params.

        Returns:
            PipelineBase: Self.
        '''
        engine = self._engine
        compile_ = self.config['compile']

        cfg = dict(
            model=self.config['model'],
            optimizer=self.config['optimizer'],
            compile=compile_,
        )
        with self._logger('compile', 'COMPILE MODEL', cfg):
            loss = engine.loss.get(compile_['loss'])
            metrics = [engine.metric.get(m) for m in compile_['metrics']]
            opt = engine.optimizer.get(self.config['optimizer'])

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

    def train(self):
        # type: () -> PipelineBase
        '''
        Call model train function with params.

        Returns:
            PipelineBase: Self.
        '''
        engine = self._engine

        cb = self.config['callbacks']
        train = self.config['train']
        log = self.config['logger']

        with self._logger('train', 'TRAIN MODEL', self.config):
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
            callbacks = engine.tools.get_callbacks(
                tb['log_dir'], tb['checkpoint_pattern'], cp,
            )

            # train model
            engine.tools.train(
                model=self.model,
                callbacks=callbacks,
                train_data=self._train_data,
                test_data=self._test_data,
                **train,
            )
        return self

    def run(self):
        # type: () -> PipelineBase
        '''
        Run the following pipeline operations:

        * load
        * train_test_split
        * unload
        * build
        * compile
        * train

        Returns:
            PipelineBase: Self.
        '''
        self.load() \
            .train_test_split() \
            .unload() \
            .build() \
            .compile() \
            .train()
        return self

    @abstractmethod
    def model_config(self):
        # type: () -> Type[BaseModel]
        '''
        Subclasses of PipelineBase will need to define a config class for models
        created in the build method.

        Returns:
            BaseModel: Pydantic BaseModel config class.
        '''
        pass  # pragma: no cover

    @abstractmethod
    def model_func(self):
        # type: () -> AnyModel
        '''
        Subclasses of PipelineBase need to define a function that builds and
        returns a machine learning model.

        Returns:
            AnyModel: Machine learning model.
        '''
        pass  # pragma: no cover
