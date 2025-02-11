from tempfile import TemporaryDirectory
import unittest

from keras import callbacks as tfcallbacks

import flatiron.core.tools as fict
import flatiron.tf.tools as f_tftools
# ------------------------------------------------------------------------------


class TFToolsTests(unittest.TestCase):
    def test_get_callbacks(self):
        with TemporaryDirectory() as root:
            proj = fict.get_tensorboard_project('proj', root)
            result = f_tftools.get_callbacks(
                proj['log_dir'], proj['checkpoint_pattern']
            )
            self.assertIsInstance(result[0], tfcallbacks.TensorBoard)
            self.assertIsInstance(result[1], tfcallbacks.ModelCheckpoint)
