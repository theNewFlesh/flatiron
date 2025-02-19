import unittest

import flatiron.torch.optimizer as fi_torchoptim
# ------------------------------------------------------------------------------


class TorchOptimizerTests(unittest.TestCase):
    def test_get(self):
        fi_torchoptim.get(dict(name='sgd', learning_rate=0.01))
        fi_torchoptim.get(dict(name='adam'))
