from tempfile import TemporaryDirectory
import unittest

from keras import callbacks as tfc

import flatiron.core.tools as fict
import flatiron.tf.tools as ftft
# ------------------------------------------------------------------------------


class TFToolsTests(unittest.TestCase):
    def test_get_callbacks(self):
        with TemporaryDirectory() as root:
            proj = fict.get_tensorboard_project('proj', root)
            result = ftft.get_callbacks(
                proj['log_dir'], proj['checkpoint_pattern']
            )
            self.assertIsInstance(result[0], tfc.TensorBoard)
            self.assertIsInstance(result[1], tfc.ModelCheckpoint)
