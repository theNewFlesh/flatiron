import unittest

import flatiron.core.loss as ficl
# ------------------------------------------------------------------------------


class LossTests(unittest.TestCase):
    def test_get(self):
        ficl.get('jaccard_loss')
        ficl.get('dice_loss')
