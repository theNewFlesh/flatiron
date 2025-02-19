from tempfile import TemporaryDirectory
from pathlib import Path
import unittest

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import flatiron
import flatiron.core.tools as fict
import flatiron.torch.tools as fi_torchtools
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


class TorchToolsTests(unittest.TestCase):
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
        fi_torchtools.pre_build()

    def test_compile(self):
        model = SimpleModel(2, 1, 2)
        result = fi_torchtools.compile(
            model=model, optimizer='Adam', loss='CrossEntropyLoss',
            metrics=['Accuracy'], device='gpu', kwargs=dict(mode='reduce-overhead')
        )

        self.assertEqual(result['model'].__class__.__name__, 'OptimizedModule')

        expected = flatiron.torch.optimizer.get('Adam').__class__
        self.assertIsInstance(result['optimizer'], expected)

        expected = flatiron.torch.loss.get('CrossEntropyLoss').__class__
        self.assertIsInstance(result['loss'], expected)

        expected = flatiron.torch.metric.get('Accuracy').__class__
        self.assertIsInstance(result['metrics'][0], expected)
