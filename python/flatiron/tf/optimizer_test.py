import unittest

import flatiron.tf.optimizer as ftfo
# ------------------------------------------------------------------------------


class TFOptimizerTests(unittest.TestCase):
    def test_get(self):
        ftfo.get(dict(class_name='sgd'))
        ftfo.get(dict(class_name='adam'))
