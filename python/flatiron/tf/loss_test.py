import unittest

import flatiron.tf.loss as ftfl
# ------------------------------------------------------------------------------


class LossTests(unittest.TestCase):
    def test_get(self):
        ftfl.get('jaccard_loss')
        ftfl.get('dice_loss')
