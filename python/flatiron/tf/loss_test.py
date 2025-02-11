import unittest

import flatiron.tf.loss as fi_tfloss
# ------------------------------------------------------------------------------


class TFLossTests(unittest.TestCase):
    def test_get(self):
        fi_tfloss.get('jaccard_loss')
        fi_tfloss.get('dice_loss')
