from typing import Any, Optional, Type  # noqa F401
from flatiron.core.types import Compiled, Filepath  # noqa F401
from pydantic import BaseModel  # noqa F401

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path

import yaml

from flatiron.core.dataset import Dataset
import flatiron.core.logging as filog
import flatiron.core.tools as fict
import flatiron.core.resolve as res
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

    @classmethod
    def generate_config(
        cls,
        framework='torch',
        project='project-name',
        callback_root='/tensorboard/parent/dir',
        dataset='/mnt/data/dataset',
        optimizer='SGD',
        loss='CrossEntropyLoss',
        metrics=['MeanMetric'],
    ):
        # type: (str, str, str, str, str, str, list[str]) -> None
        '''
        Prints a generated pipeline config based on given parameters.

        Args:
            framework (str): Framework name. Default: torch.
            project (str): Project name. Default: project-name.
            callback_root (str): Callback root path. Default: /tensorboard/parent/dir.
            dataset (str): Dataset path. Default: /mnt/data/dataset.
            optimizer (str): Optimizer name. Default: SGD.
            loss (str): Loss name. Default: CrossEntropyLoss.
            metrics (list[str]): Metric names. Default: ['MeanMetric'].
        '''
        config = res._generate_config(
            framework=framework,
            project=project,
            callback_root=callback_root,
            dataset=dataset,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )
        print(yaml.safe_dump(config))
    # --------------------------------------------------------------------------

    def __init__(self, config):
        # type: (dict) -> None
        '''
        PipelineBase is a base class for machine learning pipelines.

        Args:
            config (dict): PipelineBase config.
        '''
        self.config = res.resolve_config(config, self.model_config())

        # create Dataset instance
        dconf = self.config['dataset']
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

        self._compiled = {}  # type: Compiled
        self._train_data = None  # type: Optional[Dataset]
        self._test_data = None  # type: Optional[Dataset]
        self._loaded = False
    # --------------------------------------------------------------------------

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
        Loads train and test datasets into memory.
        Calls `load` on self._train_data and self._test_data.

        Raises:
            RuntimeError: If train and test data are not datasets.

        Returns:
            PipelineBase: Self.
        '''
        if self._train_data is None or self._test_data is None:
            msg = 'Train and test data not loaded. '
            msg += 'Please call train_test_split method first.'
            raise RuntimeError(msg)

        config = self.config['dataset']
        kwargs = dict(
            reshape=config['reshape'],
            limit=config['limit'],
            shuffle=config['shuffle'],
        )
        with self._logger('load', 'LOAD DATASETS', dict(dataset=config)):
            self._train_data.load(**kwargs)
            self._test_data.load(**kwargs)

        self._loaded = True
        return self

    def unload(self):
        # type: () -> PipelineBase
        '''
        Unload train and test datasets from memory.
        Calls `unload` on self._train_data and self._test_data.

        Raises:
            RuntimeError: If train and test data are not datasets.
            RuntimeError: If train and test data are not loaded.

        Returns:
            PipelineBase: Self.
        '''
        if self._train_data is None or self._test_data is None or not self._loaded:
            msg = 'Train and test data not loaded. '
            msg += 'Please call train_test_split, then load methods first.'
            raise RuntimeError(msg)

        config = self.config['dataset']
        with self._logger('unload', 'UNLOAD DATASETS', dict(dataset=config)):
            self._train_data.unload()
            self._test_data.unload()
        self._loaded = False
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
                limit=config['limit'],
                shuffle=config['shuffle'],
                seed=config['seed'],
            )
        return self

    def build(self):
        # type: () -> PipelineBase
        '''
        Build machine learning model and assign it to self.model.
        Calls `self.model_func` with model params.

        Returns:
            PipelineBase: Self.
        '''
        self._engine.tools.pre_build(self.config['framework']['device'])
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
        if self.config['framework']['name'] == 'tensorflow':
            import flatiron.tf as __tf_engine
            return __tf_engine
        import flatiron.torch as __torch_engine
        return __torch_engine

    def compile(self):
        # type: () -> PipelineBase
        '''
        Sets self._compiled to a dictionary of compiled objects.

        Returns:
            PipelineBase: Self.
        '''
        config = deepcopy(self.config)
        msg = dict(
            framework=config['framework'],
            model=config['model'],
            optimizer=config['optimizer'],
            loss=config['loss'],
            metrics=config['metrics'],
        )
        with self._logger('compile', 'COMPILE MODEL', msg):
            self._compiled = self._engine.tools.compile(
                framework=config['framework'],
                model=self.model,
                optimizer=config['optimizer'],
                loss=config['loss'],
                metrics=config['metrics'],
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

        callbacks = self.config['callbacks']
        train = self.config['train']
        log = self.config['logger']
        ext = 'safetensors'
        if self.config['framework']['name'] == 'tensorflow':
            ext = 'keras'

        with self._logger('train', 'TRAIN MODEL', self.config):
            # create tensorboard
            tb = fict.get_tensorboard_project(
                project=callbacks['project'],
                root=callbacks['root'],
                timezone=log['timezone'],
                extension=ext,
            )

            # create checkpoint params and callbacks
            ckpt_params = deepcopy(callbacks)
            del ckpt_params['project']
            del ckpt_params['root']
            callbacks = engine.tools.get_callbacks(
                tb['log_dir'], tb['checkpoint_pattern'], ckpt_params,
            )

            # train model
            engine.tools.train(
                compiled=self._compiled,
                callbacks=callbacks,
                train_data=self._train_data,
                test_data=self._test_data,
                params=train,
            )
        return self

    def run(self):
        # type: () -> PipelineBase
        '''
        Run the following pipeline operations:

        * build
        * compile
        * train_test_split
        * load (for tensorflow only)
        * train

        Returns:
            PipelineBase: Self.
        '''
        if self.config['framework']['name'] == 'tensorflow':
            return self \
                .build() \
                .compile() \
                .train_test_split() \
                .load() \
                .train()

        return self \
            .build() \
            .compile() \
            .train_test_split() \
            .train()

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
        # type: () -> Any
        '''
        Subclasses of PipelineBase need to define a function that builds and
        returns a machine learning model.

        Returns:
            object: Machine learning model.
        '''
        pass  # pragma: no cover
