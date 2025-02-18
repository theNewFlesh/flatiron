import unittest

import flatiron.tf.optimizer as fi_tfoptim
# ------------------------------------------------------------------------------


class TFOptimizerTests(unittest.TestCase):
    def test_get(self):
        fi_tfoptim.get(dict(name='sgd', learning_rate=0.01))
        fi_tfoptim.get(dict(name='adam'))
