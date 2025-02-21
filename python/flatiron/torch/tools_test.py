import os
from tempfile import TemporaryDirectory
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.utils.data as torchdata
import torchmetrics

import flatiron
import flatiron.core.tools as fict
import flatiron.torch.tools as fi_torchtools
from flatiron.core.dataset import Dataset
from flatiron.core.dataset_test import DatasetTestBase
# ------------------------------------------------------------------------------


class DummyModel(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels, out_channels=output_channels,
                kernel_size=(3, 3), dtype=torch.float16, padding=1
            ),
            nn.ReLU(),
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
            model = DummyModel(2, 2)
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
            result = fi_torchtools.TorchDataset \
                .monkey_patch(expected, channels_first=False)

            self.assertIs(result._info, expected._info)
            self.assertEqual(result.data, expected.data)
            self.assertEqual(result.labels, expected.labels)
            self.assertEqual(result.label_axis, expected.label_axis)
            self.assertEqual(result._ext_regex, expected._ext_regex)
            self.assertEqual(result._calc_file_size, expected._calc_file_size)
            self.assertIs(result._sample_gb, expected._sample_gb)
            self.assertFalse(result._channels_first)

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

            result = tdata[3]
            self.assertIsInstance(result, list)

            result = result[0]
            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(result.shape, (4, 10, 10))

    def test_torchdataset_getitem_no_channels_first(self):
        with TemporaryDirectory() as root:
            self.create_png_dataset_files(root, shape=(10, 10, 3))
            data = Dataset.read_directory(root, labels=[])
            tdata = fi_torchtools.TorchDataset \
                .monkey_patch(data, channels_first=False)

            result = tdata[3][0]
            self.assertEqual(result.shape, (10, 10, 3))

    def test_compile(self):
        model = DummyModel(2, 2)
        result = fi_torchtools.compile(
            model=model, optimizer=dict(name='Adam'),
            loss=dict(name='CrossEntropyLoss'),
            metrics=['Accuracy'],
            device='gpu',
            kwargs=dict(mode='reduce-overhead'),
        )

        self.assertEqual(result['model'].__class__.__name__, 'OptimizedModule')

        expected = flatiron.torch.optimizer.get(dict(name='Adam'), model).__class__
        self.assertIsInstance(result['optimizer'], expected)

        expected = flatiron.torch.loss.get(dict(name='CrossEntropyLoss')).__class__
        self.assertIsInstance(result['loss'], expected)

        expected = flatiron.torch.metric.get('Accuracy').__class__
        self.assertIsInstance(result['metrics'][0], expected)

    def get_dataset(self, root, name):
        asset = Path(root, name)
        os.makedirs(asset)
        self.create_png_dataset_files(asset, shape=(10, 10, 4))
        data = Dataset.read_directory(asset, labels='a')
        return data

    def get_dataloader(self, root, name):
        data = self.get_dataset(root, name)
        data = fi_torchtools.TorchDataset.monkey_patch(data)
        loader = torchdata.DataLoader(
            fi_torchtools.TorchDataset.monkey_patch(data),
            batch_size=4,
        )
        return loader

    def get_execute_epoch_params(self, root):
        device = torch.device('cpu')
        model = DummyModel(3, 1).to(device)
        opt = flatiron.torch.optimizer.get(dict(name='Adam'), model)
        loss = flatiron.torch.loss.get(dict(name='MSELoss'))
        torch.manual_seed(42)
        metrics = dict(mean=torchmetrics.MeanMetric())

        project = fict.get_tensorboard_project(
            'project', root, extension='safetensors'
        )
        callbacks = fi_torchtools.get_callbacks(
            project['log_dir'],
            project['checkpoint_pattern'],
            dict(save_freq='batch'),
        )
        return device, model, opt, loss, metrics, project, callbacks

    # EXECUTE-EPOCH-------------------------------------------------------------
    def test_execute_epoch(self):
        with TemporaryDirectory() as root:
            loader = self.get_dataloader(root, 'train')
            device, model, opt, loss, metrics, proj, clbk = self.get_execute_epoch_params(root)

            fi_torchtools._execute_epoch(
                epoch=1,
                model=model,
                data_loader=loader,
                optimizer=opt,
                loss_func=loss,
                metrics_funcs=metrics,
                writer=clbk['tensorboard'],
                checkpoint=clbk['checkpoint'],
                device=device,
                mode='train',
            )

            # checkpoint
            models = Path(proj['log_dir'], 'models')
            expected = os.listdir(models)
            self.assertEqual(len(expected), 1)
            self.assertRegex(expected[0], r'p-project.*\.safetensors')

            # tensorboard
            expected = os.listdir(proj['log_dir'])
            expected.remove('models')
            self.assertRegex(expected[0], 'events')

    def test_execute_epoch_no_checkpoint(self):
        with TemporaryDirectory() as root:
            loader = self.get_dataloader(root, 'train')
            device, model, opt, loss, metrics, proj, _ = self.get_execute_epoch_params(root)

            fi_torchtools._execute_epoch(
                epoch=1,
                model=model,
                data_loader=loader,
                optimizer=opt,
                loss_func=loss,
                metrics_funcs=metrics,
                writer=None,
                checkpoint=None,
                device=device,
                mode='train',
            )

            # checkpoint
            models = Path(proj['log_dir'], 'models')
            expected = os.listdir(models)
            self.assertFalse(len(expected), 0)

    def test_execute_epoch_test(self):
        with TemporaryDirectory() as root:
            loader = self.get_dataloader(root, 'train')
            device, model, opt, loss, metrics, proj, _ = self.get_execute_epoch_params(root)

            fi_torchtools._execute_epoch(
                epoch=1,
                model=model,
                data_loader=loader,
                optimizer=opt,
                loss_func=loss,
                metrics_funcs=metrics,
                writer=None,
                checkpoint=None,
                device=device,
                mode='test',
            )

            # checkpoint
            models = Path(proj['log_dir'], 'models')
            expected = os.listdir(models)
            self.assertFalse(len(expected), 0)

    # TRAIN---------------------------------------------------------------------
    def test_train(self):
        with TemporaryDirectory() as root:
            train_data = self.get_dataset(root, 'train')
            test_data = self.get_dataset(root, 'test')
            device, model, opt, loss, metrics, proj, clbk = self.get_execute_epoch_params(root)
            compiled = dict(
                model=model,
                optimizer=opt,
                loss=loss,
                metrics=metrics,
                device=device,
                kwargs={},
            )

            fi_torchtools.train(
                compiled=compiled,
                callbacks=clbk,
                train_data=train_data,
                test_data=test_data,
                batch_size=4,
                epochs=1,
                seed=42,
            )

            # checkpoint
            models = Path(proj['log_dir'], 'models')
            expected = os.listdir(models)
            self.assertEqual(len(expected), 1)
            self.assertRegex(expected[0], r'p-project.*\.safetensors')

            # tensorboard
            expected = os.listdir(proj['log_dir'])
            expected.remove('models')
            self.assertRegex(expected[0], 'events')
