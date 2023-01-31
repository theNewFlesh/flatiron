from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from schematics.exceptions import DataError

from flatiron.core.dataset_config import DatasetConfig
# ------------------------------------------------------------------------------


class DatasetConfigTests(unittest.TestCase):
    def get_config(self, root):
        return dict(
            source=Path(root).as_posix(),
            load_limit=None,
            load_shuffle=False,
            split_index=-1,
            split_axis=-1,
            split_test_size=0.2,
            split_train_size=None,
            split_random_state=42,
            split_shuffle=True,
        )

    def test_validate(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            DatasetConfig(config).validate()

    def test_to_native(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            keys = [
                'load_limit',
                'load_shuffle',
                'split_axis',
                'split_test_size',
                'split_train_size',
                'split_random_state',
                'split_shuffle',
            ]
            for key in keys:
                del config[key]

            result = DatasetConfig(config).to_native()
            self.assertEqual(result, self.get_config(root))

    def test_split_test_size(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            config['split_test_size'] = -0.2
            with self.assertRaises(DataError):
                DatasetConfig(config).validate()

    def test_split_train_size(self):
        with TemporaryDirectory() as root:
            config = self.get_config(root)
            config['split_train_size'] = -0.2
            with self.assertRaises(DataError):
                DatasetConfig(config).validate()
