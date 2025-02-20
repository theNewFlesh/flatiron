from tempfile import TemporaryDirectory
import os
from pathlib import Path

import pytest
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.utils.data as torchdata

import flatiron
import flatiron.core.tools as fict
import flatiron.torch.tools as fi_torchtools
from flatiron.core.dataset import Dataset
from flatiron.core.dataset_test import DatasetTestBase
# ------------------------------------------------------------------------------


class SimpleModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
        )

    def forward(self, x):
        return self.layer_stack(x)


class TorchToolsTests(DatasetTestBase):
    def test_modelcheckpoint_init(self):
        result = fi_torchtools.ModelCheckpoint('/foo/bar', 'batch')
        self.assertEqual(result._filepath, '/foo/bar')
        self.assertEqual(result.save_freq, 'batch')

    def test_modelcheckpoint_save(self):
        with TemporaryDirectory() as root:
            target = Path(root, 'foo_{epoch:02d}.safetensors')
            check = fi_torchtools.ModelCheckpoint(target, 'batch')
            model = SimpleModel(2, 1, 2)
            check.save(model, 1)

            expected = Path(check._filepath.format(epoch=1))
            self.assertTrue(expected.is_file())

    def test_get_callbacks(self):
        with TemporaryDirectory() as root:
            proj = fict.get_tensorboard_project('proj', root)
            result = fi_torchtools.get_callbacks(
                proj['log_dir'], proj['checkpoint_pattern']
            )
            self.assertIsInstance(result['tensorboard'], SummaryWriter)
            self.assertIsInstance(result['checkpoint'], fi_torchtools.ModelCheckpoint)

    def test_pre_build(self):
        fi_torchtools.pre_build('cpu')

    def test_torchdataset_monkey_patch(self):
        with TemporaryDirectory() as root:
            self.create_png_dataset_files(root, shape=(10, 10, 4))
            expected = Dataset.read_directory(root, labels='a')
            result = fi_torchtools.TorchDataset.monkey_patch(expected)
            self.assertIs(result._info, expected._info)
            self.assertEqual(result.data, expected.data)
            self.assertEqual(result.labels, expected.labels)
            self.assertEqual(result.label_axis, expected.label_axis)
            self.assertEqual(result._ext_regex, expected._ext_regex)
            self.assertEqual(result._calc_file_size, expected._calc_file_size)
            self.assertIs(result._sample_gb, expected._sample_gb)

    def test_torchdataset_getitem(self):
        with TemporaryDirectory() as root:
            self.create_png_dataset_files(root, shape=(10, 10, 4))
            data = Dataset.read_directory(root, labels='a')
            tdata = fi_torchtools.TorchDataset.monkey_patch(data)
            result = tdata[3]
            self.assertIsInstance(result, list)
            for item in result:
                self.assertIsInstance(item, torch.Tensor)

    def test_torchdataset_getitem_no_labels(self):
        with TemporaryDirectory() as root:
            self.create_png_dataset_files(root, shape=(10, 10, 4))
            data = Dataset.read_directory(root, labels=[])
            tdata = fi_torchtools.TorchDataset.monkey_patch(data)
            self.assertIsInstance(tdata[3], torch.Tensor)

    def test_compile(self):
        model = SimpleModel(2, 1, 2)
        result = fi_torchtools.compile(
            model=model, optimizer=dict(name='Adam'), loss='CrossEntropyLoss',
            metrics=['Accuracy'], device='gpu', kwargs=dict(mode='reduce-overhead')
        )

        self.assertEqual(result['model'].__class__.__name__, 'OptimizedModule')

        expected = flatiron.torch.optimizer.get(dict(name='Adam'), model).__class__
        self.assertIsInstance(result['optimizer'], expected)

        expected = flatiron.torch.loss.get('CrossEntropyLoss').__class__
        self.assertIsInstance(result['loss'], expected)

        expected = flatiron.torch.metric.get('Accuracy').__class__
        self.assertIsInstance(result['metrics'][0], expected)

    @pytest.mark.timeout(30)
    @pytest.mark.skipif('SKIP_SLOW_TESTS' in os.environ, reason='slow test')
    def test_execute_epoch(self):
        with TemporaryDirectory() as root:
            self.create_png_dataset_files(root, shape=(10, 10, 4))
            data = Dataset.read_directory(root, labels='a')
            data = fi_torchtools.TorchDataset.monkey_patch(data)

            model = SimpleModel(2, 1, 2)
            loader = torchdata.DataLoader(
                fi_torchtools.TorchDataset.monkey_patch(data),
                batch_size=32,
            )
            opt = flatiron.torch.optimizer.get(dict(name='Adam'), model)
            loss = flatiron.torch.loss.get('CrossEntropyLoss')
            device = torch.device('cuda')
            torch.manual_seed(42)
            model = model.to(device)

            fi_torchtools._execute_epoch(
                epoch=1,
                model=model,
                data_loader=loader,
                optimizer=opt,
                loss_func=loss,
                device=device,
                mode='train',
            )
