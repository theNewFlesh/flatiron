import unittest

from flatiron.torch.models.dummy import DummyModel
import flatiron.torch.optimizer as fi_torchoptim
# ------------------------------------------------------------------------------


class TorchOptimizerTests(unittest.TestCase):
    def test_get(self):
        model = DummyModel(2, 2)
        fi_torchoptim.get(dict(name='SGD', learning_rate=0.01), model)
        fi_torchoptim.get(dict(name='Adam'), model)
