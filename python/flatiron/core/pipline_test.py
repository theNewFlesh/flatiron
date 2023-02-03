from pathlib import Path
from tempfile import TemporaryDirectory
import logging
import os
import unittest

from schematics.exceptions import DataError
import numpy as np
import pandas as pd
import schematics.models as scm
import schematics.types as scmt
import yaml

import flatiron.core.dataset as ficd
import flatiron.core.pipeline as ficp
# ------------------------------------------------------------------------------


class TestModel:
    def __init__(self, **kwargs):
        self.state = 'init'
        pass

    def compile(self, **kwargs):
        self.state = 'compile'
        pass

    def fit(self, **kwargs):
        self.state = 'fit'
        pass


class TestConfig(scm.Model):
    foo = scmt.StringType(required=True)
    bar = scmt.StringType(default='taco')


class TestPipeline(ficp.PipelineBase):
    def model_config(self):
        return TestConfig

    def model_func(self):
        return TestModel
# ------------------------------------------------------------------------------


class PipelineTests(unittest.TestCase):
    def write_npy(self, target, shape=(10, 10, 3)):
        target = Path(target)
        os.makedirs(target.parent, exist_ok=True)
        array = np.ones(shape, dtype=np.uint8)
        np.save(target, array)

    def create_dataset_files(self, root, shape=(10, 10, 3)):
        os.makedirs(Path(root, 'data'))
        info = pd.DataFrame()
        info['filepath_relative'] = [f'data/chunk_f{i:02d}.npy' for i in range(10)]
        info['asset_path'] = root
        info.filepath_relative \
            .apply(lambda x: Path(root, x)) \
            .apply(lambda x: self.write_npy(x, shape))
        info_path = Path(root, 'info.csv').as_posix()
        info.to_csv(info_path, index=None)
        return info, info_path

    def get_config(self, root):
        proj = Path(root, 'proj').as_posix()
        os.makedirs(proj)
        dset = Path(proj, 'dset001', 'dset001_v001').as_posix()
        _, info_path = self.create_dataset_files(dset)
        return dict(
            model=dict(
                foo='bar'
            ),
            dataset=dict(
                source=info_path,
                split_index=-1,
            ),
            callbacks=dict(
                project='proj',
                root=root,
            ),
            optimizer=dict(),
            compile=dict(
                loss='dice_loss',
                metrics=['jaccard', 'dice'],
            ),
            fit=dict(),
            logger=dict(),
        )

    def test_init(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            result = TestPipeline(config).config['optimizer']['name']
            self.assertEqual(result, 'sgd')

    def test_init_model(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)

            result = TestPipeline(config).config['model']
            expected = dict(foo='bar', bar='taco')
            self.assertEqual(result, expected)

            config['model'] = {}
            with self.assertRaises(DataError):
                TestPipeline(config)

    def test_init_dataset(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            asset = Path(root, 'proj/dset001/dset001_v001').as_posix()

            # directory
            config['dataset']['source'] = asset
            result = TestPipeline(config).dataset
            self.assertIsInstance(result, ficd.Dataset)

            # file
            src = Path(asset, 'info.csv').as_posix()
            config['dataset']['source'] = src
            result = TestPipeline(config).dataset
            self.assertIsInstance(result, ficd.Dataset)

    def test_from_string(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            config = yaml.dump(config)
            TestPipeline.from_string(config)

    def test_read_yaml(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            src = Path(root, 'config.yaml')
            with open(src, 'w') as f:
                yaml.safe_dump(config, f)
            TestPipeline.read_yaml(src)

    def test_load(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            pipe = TestPipeline(config)
            self.assertIsNone(pipe.dataset.data)

            with self.assertLogs(level=logging.WARNING) as log:
                result = pipe.load().dataset.data
            self.assertRegex(log.output[0], 'LOAD DATASET')
            self.assertIsInstance(result, np.ndarray)

    def test_train_test_split(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            pipe = TestPipeline(config).load()
            self.assertIsNone(pipe.x_train)
            self.assertIsNone(pipe.x_test)
            self.assertIsNone(pipe.y_train)
            self.assertIsNone(pipe.y_test)

            with self.assertLogs(level=logging.WARNING) as log:
                result = pipe.train_test_split()
            self.assertRegex(log.output[0], 'TRAIN TEST SPLIT')
            self.assertIsInstance(result.x_train, np.ndarray)
            self.assertIsInstance(result.x_test, np.ndarray)
            self.assertIsInstance(result.y_train, np.ndarray)
            self.assertIsInstance(result.y_test, np.ndarray)

    def test_unload(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            pipe = TestPipeline(config).load()

            with self.assertLogs(level=logging.WARNING) as log:
                result = pipe.unload().dataset.data
            self.assertRegex(log.output[0], 'UNLOAD DATASET')
            self.assertIsNone(result)

    def test_build(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            pipe = TestPipeline(config)

            with self.assertLogs(level=logging.WARNING) as log:
                pipe.build()
            self.assertRegex(log.output[0], 'BUILD MODEL')

    def test_compile(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            pipe = TestPipeline(config).build()

            with self.assertLogs(level=logging.WARNING) as log:
                pipe.compile()
            self.assertRegex(log.output[0], 'COMPILE MODEL')
            self.assertEqual(pipe.model.state, 'compile')

    def test_fit(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            pipe = TestPipeline(config) \
                .load() \
                .train_test_split() \
                .unload() \
                .build() \
                .compile()

            with self.assertLogs(level=logging.WARNING) as log:
                pipe.fit()
            self.assertRegex(log.output[0], 'FIT MODEL')
            self.assertEqual(pipe.model.state, 'fit')
