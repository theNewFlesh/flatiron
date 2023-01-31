from typing import Union  # noqa F401
import schematics.models as scm  # noqa F401

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path

import yaml

from flatiron.core.dataset import Dataset
from flatiron.core.dataset_config import DatasetConfig

Filepath = Union[str, Path]
# ------------------------------------------------------------------------------


class PipelineBase(ABC):
    _model_config_class = None

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
        config['model'] = self._validate(config['model'], self._model_config_class)
        config['dataset'] = self._validate(config['dataset'], DatasetConfig)
        self.config = config

        src = config['dataset']['source']
        if Path(src).is_file():
            self.dataset = Dataset.read_csv(src)
        else:
            self.dataset = Dataset.read_directory(src)

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
        self.model.compile()
        return self

    def load(self):
        self.dataset.load(**self.config['dataset'])
        return self
