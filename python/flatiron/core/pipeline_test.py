from pathlib import Path
from tempfile import TemporaryDirectory
import logging
import os
import unittest

from pydantic import BaseModel
from tensorflow import keras  # noqa F401
from keras import layers as tfl
from keras import models as tfmodels
import numpy as np
import pandas as pd
import yaml

import flatiron.core.dataset as ficd
import flatiron.core.pipeline as ficp
import flatiron.tf as fitf

# disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# ------------------------------------------------------------------------------


def get_fake_model(shape):
    input_ = tfl.Input(shape, name='input')
    output = tfl.Conv2D(1, (1, 1), activation='relu', name='output')(input_)
    model = tfmodels.Model(inputs=[input_], outputs=[output])
    return model


class FakeConfig(BaseModel):
    shape: list[int]


class FakePipeline(ficp.PipelineBase):
    def model_config(self):
        return FakeConfig

    def model_func(self):
        return get_fake_model
# ------------------------------------------------------------------------------


class PipelineTests(unittest.TestCase):
    def write_npy(self, target, shape=(10, 10, 10, 4)):
        target = Path(target)
        os.makedirs(target.parent, exist_ok=True)
        array = np.ones(shape, dtype=np.float16)
        np.save(target, array)

    def create_dataset_files(self, root, shape=(10, 10, 10, 4)):
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
            engine='tensorflow',
            model=dict(
                shape=[10, 10, 3]
            ),
            dataset=dict(
                source=info_path,
                labels=[2],
                label_axis=-1,
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
            train=dict(
                epochs=1,
            ),
            logger=dict(
                slack_url='https://hooks.slack.com/services/fake-service',
                slack_channel='test',
                slack_methods=['load'],
            ),
        )

    def test_init(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            result = FakePipeline(config).config['optimizer']['name']
            self.assertEqual(result, 'sgd')

    def test_init_model(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)

            result = FakePipeline(config).config['model']
            expected = dict(shape=[10, 10, 3])
            self.assertEqual(result, expected)

            config['model'] = {}
            with self.assertRaises(ValueError):
                FakePipeline(config)

    def test_logger(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            asset = Path(root, 'proj/dset001/dset001_v001').as_posix()
            pipe = FakePipeline(config)

            # no slack
            config['dataset']['source'] = asset
            result = pipe._logger('foobar', 'some-message', dict(foo='bar'))
            self.assertIsNone(result._message_func)
            self.assertIsNone(result._callback)

            # slack
            config['dataset']['source'] = asset
            result = pipe._logger('load', 'some-message', dict(foo='bar'))
            self.assertIsNotNone(result._message_func)
            self.assertIsNotNone(result._callback)

    def test_init_dataset(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            asset = Path(root, 'proj/dset001/dset001_v001').as_posix()

            # directory
            config['dataset']['source'] = asset
            result = FakePipeline(config).dataset
            self.assertIsInstance(result, ficd.Dataset)

            # file
            src = Path(asset, 'info.csv').as_posix()
            config['dataset']['source'] = src
            result = FakePipeline(config).dataset
            self.assertIsInstance(result, ficd.Dataset)

    def test_from_string(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            config = yaml.dump(config)
            FakePipeline.from_string(config)

    def test_read_yaml(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            src = Path(root, 'config.yaml')
            with open(src, 'w') as f:
                yaml.safe_dump(config, f)
            FakePipeline.read_yaml(src)

    def test_load(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            pipe = FakePipeline(config)
            self.assertIsNone(pipe._train_data)
            self.assertIsNone(pipe._test_data)
            pipe.train_test_split()
            self.assertIsNotNone(pipe._train_data)
            self.assertIsNotNone(pipe._test_data)
            self.assertFalse(pipe._loaded)

            with self.assertLogs(level=logging.WARNING) as log:
                result = pipe.load()
            self.assertRegex(log.output[0], 'LOAD DATASET')
            self.assertIsInstance(result._train_data.data, np.ndarray)
            self.assertIsInstance(result._test_data.data, np.ndarray)
            self.assertTrue(pipe._loaded)

    def test_load_errors(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            pipe = FakePipeline(config)
            self.assertIsNone(pipe.dataset.data)

            expected = 'Train and test data not loaded. '
            expected += 'Please call train_test_split method first'
            with self.assertRaisesRegex(RuntimeError, expected):
                pipe.load()

    def test_unload(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            pipe = FakePipeline(config).train_test_split().load()
            self.assertTrue(pipe._loaded)

            with self.assertLogs(level=logging.WARNING) as log:
                result = pipe.unload()
            self.assertRegex(log.output[0], 'UNLOAD DATASET')
            self.assertIsNone(result._train_data.data)
            self.assertIsNone(result._test_data.data)
            self.assertFalse(pipe._loaded)

    def test_unload_errors(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            pipe = FakePipeline(config)
            self.assertIsNone(pipe.dataset.data)

            expected = 'Train and test data not loaded. '
            expected += 'Please call train_test_split, then load methods first.'
            with self.assertRaisesRegex(RuntimeError, expected):
                pipe.unload()

            pipe.train_test_split()
            with self.assertRaisesRegex(RuntimeError, expected):
                pipe.unload()

    def test_train_test_split(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            pipe = FakePipeline(config)
            self.assertIsNone(pipe._train_data)
            self.assertIsNone(pipe._test_data)

            with self.assertLogs(level=logging.WARNING) as log:
                result = pipe.train_test_split()
            self.assertRegex(log.output[0], 'TRAIN TEST SPLIT')
            self.assertIsInstance(result._train_data, ficd.Dataset)
            self.assertIsInstance(result._test_data, ficd.Dataset)

    def test_build(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            pipe = FakePipeline(config)

            with self.assertLogs(level=logging.WARNING) as log:
                pipe.build()
            self.assertRegex(log.output[0], 'BUILD MODEL')

    def test_engine(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            config['engine'] = 'tensorflow'
            result = FakePipeline(config)._engine
            self.assertIs(result, fitf)

    def test_compile_tf(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            pipe = FakePipeline(config).build()

            self.assertEqual(pipe._compiled, {})

            with self.assertLogs(level=logging.WARNING) as log:
                pipe.compile()
            self.assertRegex(log.output[0], 'COMPILE MODEL')
            self.assertEqual(pipe._compiled, dict(model=pipe.model))

    def test_compile_loss(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            config['compile']['loss'] = 'mse'
            pipe = FakePipeline(config).build().compile()
            self.assertIs(pipe.model.loss.__name__, 'mean_squared_error')

    def test_train(self):
        with TemporaryDirectory(prefix='test-train-') as root:
            config = self.get_config(root)
            pipe = FakePipeline(config) \
                .train_test_split() \
                .load() \
                .build() \
                .compile()

            with self.assertLogs(level=logging.WARNING) as log:
                pipe.train()
            self.assertRegex(log.output[0], 'TRAIN MODEL')
            self.assertTrue(Path(root, 'proj/tensorboard').is_dir())

    def test_run(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            config = yaml.dump(config)
            tb = Path(root, 'proj/tensorboard')

            self.assertFalse(tb.is_dir())
            FakePipeline.from_string(config).run()
            self.assertTrue(tb.is_dir())
