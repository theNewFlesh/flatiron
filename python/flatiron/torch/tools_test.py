import unittest

import torch.nn as nn

import flatiron
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
    def test_compile(self):
        model = SimpleModel(2, 1, 2)
        result = fi_torchtools.compile(
            model=model, optimizer='Adam', loss='CrossEntropyLoss',
            metrics=['Accuracy'], kwargs=dict(mode='reduce-overhead')
        )

        self.assertEqual(result['model'].__class__.__name__, 'OptimizedModule')

        expected = flatiron.torch.optimizer.get('Adam').__class__
        self.assertIsInstance(result['optimizer'], expected)

        expected = flatiron.torch.loss.get('CrossEntropyLoss').__class__
        self.assertIsInstance(result['loss'], expected)

        expected = flatiron.torch.metric.get('Accuracy').__class__
        self.assertIsInstance(result['metrics'][0], expected)
