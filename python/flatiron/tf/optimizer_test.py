import unittest

import flatiron.tf.optimizer as fi_tfoptim
# ------------------------------------------------------------------------------


class TFOptimizerTests(unittest.TestCase):
    def test_get(self):
        fi_tfoptim.get(dict(class_name='sgd'))
        fi_tfoptim.get(dict(class_name='adam'))
