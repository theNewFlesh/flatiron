import unittest

import flatiron.tf.loss as f_tfloss
# ------------------------------------------------------------------------------


class TFLossTests(unittest.TestCase):
    def test_get(self):
        f_tfloss.get('jaccard_loss')
        f_tfloss.get('dice_loss')
