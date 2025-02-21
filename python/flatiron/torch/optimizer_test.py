import unittest

import flatiron.torch.optimizer as fi_torchoptim
from flatiron.torch.tools_test import DummyModel
# ------------------------------------------------------------------------------


class TorchOptimizerTests(unittest.TestCase):
    def test_get(self):
        model = DummyModel(2, 2)
        fi_torchoptim.get(dict(name='sgd', learning_rate=0.01), model)
        fi_torchoptim.get(dict(name='adam'), model)
