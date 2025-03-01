from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import logging
import os
import unittest

import cv_depot.api as cvd
import numpy as np
import pandas as pd
import pytest
import yaml

import flatiron.core.dataset as ficd
import flatiron.tf as fitf
import flatiron.tf.models.dummy as fi_tfdummy
import flatiron.torch.models.dummy as fi_torchdummy

# disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# ------------------------------------------------------------------------------


class PipelineTestBase(unittest.TestCase):
    def write_npy(self, target, shape=(10, 10, 10, 4)):
        target = Path(target)
        os.makedirs(target.parent, exist_ok=True)
        array = np.ones(shape, dtype=np.float16)
        np.save(target, array)

    def write_png(self, target, shape=(10, 10, 4)):
        target = Path(target)
        os.makedirs(target.parent, exist_ok=True)
        array = np.floor(np.random.random(shape) * 255).astype(np.uint8)
        cvd.Image.from_array(array).write(target)

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

    def create_png_dataset_files(self, root, shape=(10, 10, 3), indicator='f'):
        os.makedirs(Path(root, 'data'))
        info = pd.DataFrame()
        info['filepath_relative'] = [f'data/foo_{indicator}{i:02d}.png' for i in range(10)]
        info['asset_path'] = root
        info.filepath_relative \
            .apply(lambda x: Path(root, x)) \
            .apply(lambda x: self.write_png(x, shape))
        info_path = Path(root, 'info.csv').as_posix()
        info.to_csv(info_path, index=None)
        return info, info_path

    def get_config(self, root, png=False):
        proj = Path(root, 'proj').as_posix()
        os.makedirs(proj)
        dset = Path(proj, 'dset001', 'dset001_v001').as_posix()
        if png:
            _, info_path = self.create_png_dataset_files(dset)
        else:
            _, info_path = self.create_dataset_files(dset)
        return dict(
            framework=dict(
                name='tensorflow',
                device='cpu',
            ),
            model=dict(
                shape=[10, 10, 3]
            ),
            optimizer=dict(name='SGD'),
            loss=dict(name='dice_loss'),
            metrics=[dict(name='jaccard'), dict(name='dice')],
            dataset=dict(
                source=info_path,
                labels=[2],
                label_axis=-1,
            ),
            callbacks=dict(
                project='proj',
                root=root,
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


class TFPipelineTests(PipelineTestBase):
    def test_init(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            result = fi_tfdummy.DummyPipeline(config).config['optimizer']['name']
            self.assertEqual(result, 'SGD')

    def test_init_model(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)

            result = fi_tfdummy.DummyPipeline(config).config['model']
            expected = dict(shape=[10, 10, 3], activation='relu')
            self.assertEqual(result, expected)

            config['model'] = {}
            with self.assertRaises(ValueError):
                fi_tfdummy.DummyPipeline(config)

    def test_resolve_model(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            expected = {'shape': [10, 10, 3]}
            self.assertEqual(config['model'], expected)

            pipe = fi_tfdummy.DummyPipeline(deepcopy(config))
            result = pipe._resolve_model(config)
            expected = {'shape': [10, 10, 3], 'activation': 'relu'}
            self.assertEqual(result['model'], expected)

    def test_resolve_pipeline(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            pipe = fi_tfdummy.DummyPipeline(deepcopy(config))
            tz = config['logger'].get('timezone', None)
            self.assertIs(tz, None)

            result = pipe._resolve_pipeline(config)

            self.assertEqual(result['logger']['timezone'], 'UTC')

            expected = {'shape': [10, 10, 3]}
            self.assertEqual(result['model'], expected)

    def test_resolve_field(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            pipe = fi_tfdummy.DummyPipeline(deepcopy(config))

            result = pipe._resolve_field(deepcopy(config), 'optimizer')
            res = result['optimizer']
            self.assertEqual(res['name'], 'SGD')
            self.assertIs(res['clipnorm'], None)

            del config['optimizer']
            del result['optimizer']
            self.assertEqual(result, config)

    def test_resolve_field_metrics(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            pipe = fi_tfdummy.DummyPipeline(deepcopy(config))

            result = pipe._resolve_field(config, 'metrics')['metrics']
            self.assertEqual(result[0]['name'], 'jaccard')
            self.assertEqual(result[1]['name'], 'dice')

    def test_resolve_subconfig(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            pipe = fi_tfdummy.DummyPipeline(deepcopy(config))

            result = pipe._resolve_subconfig(
                dict(name='Huber'),
                'TFLoss',
                True,
                'flatiron.tf.config',
                'flatiron.tf.loss',
            )
            self.assertEqual(result['name'], 'Huber')
            self.assertIs(result['dtype'], None)

    def test_resolve_subconfig_config_module(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            pipe = fi_tfdummy.DummyPipeline(deepcopy(config))

            result = pipe._resolve_subconfig(
                dict(name='jaccard_loss'),
                'TFLoss',
                False,
                'flatiron.tf.loss',
                None,
            )
            self.assertEqual(result['name'], 'jaccard_loss')

    def test_resolve_subconfig_no_prepend(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            pipe = fi_tfdummy.DummyPipeline(deepcopy(config))

            result = pipe._resolve_subconfig(
                dict(name='tensorflow'),
                'TFFramework',
                False,
                'flatiron.tf.config',
                None,
            )
            self.assertEqual(result['name'], 'tensorflow')
            self.assertIs(result['jit_compile'], False)

    def test_resolve_subconfig_custom(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            pipe = fi_tfdummy.DummyPipeline(deepcopy(config))

            expected = dict(name='jaccard_loss')
            result = pipe._resolve_subconfig(
                expected,
                'TFLoss',
                True,
                'flatiron.tf.config',
                'flatiron.tf.loss',
            )
            self.assertEqual(result, expected)

    def test_logger(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            asset = Path(root, 'proj/dset001/dset001_v001').as_posix()
            pipe = fi_tfdummy.DummyPipeline(config)

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
            result = fi_tfdummy.DummyPipeline(config).dataset
            self.assertIsInstance(result, ficd.Dataset)

            # file
            src = Path(asset, 'info.csv').as_posix()
            config['dataset']['source'] = src
            result = fi_tfdummy.DummyPipeline(config).dataset
            self.assertIsInstance(result, ficd.Dataset)

    def test_from_string(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            config = yaml.dump(config)
            fi_tfdummy.DummyPipeline.from_string(config)

    def test_read_yaml(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            src = Path(root, 'config.yaml')
            with open(src, 'w') as f:
                yaml.safe_dump(config, f)
            fi_tfdummy.DummyPipeline.read_yaml(src)

    def test_load(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            pipe = fi_tfdummy.DummyPipeline(config)
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
            pipe = fi_tfdummy.DummyPipeline(config)
            self.assertIsNone(pipe.dataset.data)

            expected = 'Train and test data not loaded. '
            expected += 'Please call train_test_split method first'
            with self.assertRaisesRegex(RuntimeError, expected):
                pipe.load()

    def test_unload(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            pipe = fi_tfdummy.DummyPipeline(config).train_test_split().load()
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
            pipe = fi_tfdummy.DummyPipeline(config)
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
            pipe = fi_tfdummy.DummyPipeline(config)
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
            pipe = fi_tfdummy.DummyPipeline(config)

            with self.assertLogs(level=logging.WARNING) as log:
                pipe.build()
            self.assertRegex(log.output[0], 'BUILD MODEL')

    def test_engine(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            config['framework']['name'] = 'tensorflow'
            result = fi_tfdummy.DummyPipeline(config)._engine
            self.assertIs(result, fitf)

    def test_compile_tf(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            pipe = fi_tfdummy.DummyPipeline(config).build()

            self.assertEqual(pipe._compiled, {})

            with self.assertLogs(level=logging.WARNING) as log:
                pipe.compile()
            self.assertRegex(log.output[0], 'COMPILE MODEL')
            self.assertEqual(pipe._compiled, dict(model=pipe.model))

    def test_compile_loss(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            config['loss']['name'] = 'Dice'
            pipe = fi_tfdummy.DummyPipeline(config).build().compile()
            self.assertIs(pipe.model.loss.name, 'dice')

    def test_train(self):
        with TemporaryDirectory(prefix='test-train-') as root:
            config = self.get_config(root)
            pipe = fi_tfdummy.DummyPipeline(config) \
                .train_test_split() \
                .load() \
                .build() \
                .compile()

            with self.assertLogs(level=logging.WARNING) as log:
                pipe.train()
            self.assertRegex(log.output[0], 'TRAIN MODEL')
            self.assertTrue(Path(root, 'proj/tensorboard').is_dir())

    def test_run_tf(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            config = yaml.dump(config)
            tb = Path(root, 'proj/tensorboard')

            self.assertFalse(tb.is_dir())
            fi_tfdummy.DummyPipeline.from_string(config).run()
            self.assertTrue(tb.is_dir())


class TorchPipelineTests(PipelineTestBase):
    def get_config(self, root):
        config = super().get_config(root, png=True)
        config.update(dict(
            framework=dict(name='torch', device='cpu'),
            model=dict(
                input_channels=3,
                output_channels=1,
            ),
            optimizer=dict(name='Adam'),
            loss=dict(name='MSELoss'),
            metrics=[dict(name='MeanMetric')],
        ))
        return config

    @pytest.mark.skipif('SKIP_SLOW_TESTS' in os.environ, reason='slow test')
    def test_run_torch(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            config = yaml.dump(config)
            tb = Path(root, 'proj/tensorboard')

            self.assertFalse(tb.is_dir())
            fi_torchdummy.DummyPipeline.from_string(config).run()
            self.assertTrue(tb.is_dir())
