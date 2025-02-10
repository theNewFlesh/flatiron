import unittest

import flatiron.tf.optimizer as f_tfoptim
# ------------------------------------------------------------------------------


class TFOptimizerTests(unittest.TestCase):
    def test_get(self):
        f_tfoptim.get(dict(class_name='sgd'))
        f_tfoptim.get(dict(class_name='adam'))
