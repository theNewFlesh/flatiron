from pathlib import Path
from tempfile import TemporaryDirectory
import os
import unittest

from lunchbox.enforce import EnforceError
from pandas import DataFrame

from flatiron.core.dataset import Dataset
# ------------------------------------------------------------------------------


class DatasetTests(unittest.TestCase):
    def create_dataset_files(self, root):
        os.makedirs(Path(root, 'data'))
        info = DataFrame()
        info['filepath_relative'] = [f'data/foo_f{i:02d}.npy' for i in range(10)]
        info['asset_path'] = root
        info.filepath_relative.apply(lambda x: Path(root, x).touch())
        info_path = Path(root, 'info.csv').as_posix()
        info.to_csv(info_path, index=None)
        return info, info_path

    def test_read_csv(self):
        with TemporaryDirectory() as root:
            _, csv = self.create_dataset_files(root)
            Dataset.read_csv(csv)

    def test_read_csv_errors(self):
        expected = 'Filepath does not exist: /foo/bar.csv'
        with self.assertRaisesRegex(EnforceError, expected):
            Dataset.read_csv('/foo/bar.csv')

        with TemporaryDirectory() as root:
            info = Path(root, 'foo.bar')
            info.touch()
            expected = 'Filepath extension must be csv. '
            expected += f'Given filepath: {info.as_posix()}'
            with self.assertRaisesRegex(EnforceError, expected):
                Dataset.read_csv(info)

    def test_read_directory(self):
        with TemporaryDirectory() as root:
            self.create_dataset_files(root)
            Dataset.read_directory(root)

    def test_read_directory_errors(self):
        # no directory
        expected = 'Directory does not exist: /tmp/foo'
        with self.assertRaisesRegex(EnforceError, expected):
            Dataset.read_directory('/tmp/foo')

        # no csv
        with TemporaryDirectory() as root:
            expected = r'Dataset directory must contain only 1 CSV file\. '
            expected += r'CSV files found: \[\]'
            with self.assertRaisesRegex(EnforceError, expected):
                Dataset.read_directory(root)

        # 2 csv
        with TemporaryDirectory() as root:
            self.create_dataset_files(root)
            Path(root, 'foobar.csv').touch()
            expected = r'Dataset directory must contain only 1 CSV file\. '
            expected += r'CSV files found:.*/foobar\.csv.*info\.csv'
            with self.assertRaisesRegex(EnforceError, expected):
                Dataset.read_directory(root)

    def test_init(self):
        with TemporaryDirectory() as root:
            info, _ = self.create_dataset_files(root)
            result = Dataset(info)._info
            cols = [
                'chunk', 'asset_path', 'filepath_relative', 'filepath', 'loaded'
            ]
            for col in cols:
                self.assertIn(col, result.columns)

            result = result.columns.tolist()[-5:]
            self.assertEqual(result, cols)

            # loaded column
            result = Dataset(info)._info.loaded.unique().tolist()
            self.assertEqual(result, [False])

    def test_init_columns_error(self):
        with TemporaryDirectory() as root:
            info, _ = self.create_dataset_files(root)
            cols = ['asset_path', 'filepath_relative']
            for col in cols:
                temp = info.drop(col, axis=1)
                expected = f'Required columns not found in info:.*{col}'
                with self.assertRaisesRegex(EnforceError, expected):
                    Dataset(temp)

    def test_init_root_error(self):
        with TemporaryDirectory() as root:
            info, _ = self.create_dataset_files(root)
            info.loc[3, 'asset_path'] = '/foo/bar'
            expected = 'Info must contain only 1 root path. '
            expected += f'Paths found:.*{root}.*/foo/bar'
            with self.assertRaisesRegex(EnforceError, expected):
                Dataset(info)

            info['asset_path'] = '/foo/bar'
            expected = 'Directory does not exist: /foo/bar'
            with self.assertRaisesRegex(EnforceError, expected):
                Dataset(info)

    def test_init_filepath_error(self):
        with TemporaryDirectory() as root:
            info, _ = self.create_dataset_files(root)
            info.loc[3, 'filepath_relative'] = '/foo/bar.npy'
            expected = 'Chunk files do not exist:.*/foo/bar.npy'
            with self.assertRaisesRegex(EnforceError, expected):
                Dataset(info)

    def test_init_extension_error(self):
        with TemporaryDirectory() as root:
            info, _ = self.create_dataset_files(root)
            src = info.loc[3, 'filepath_relative']
            src = Path(root, src).as_posix()
            tgt = src.replace('npy', 'txt')
            os.rename(src, tgt)
            info.loc[3, 'filepath_relative'] = tgt
            expected = 'Found chunk files missing npy extension:.*foo_f03.txt'
            with self.assertRaisesRegex(EnforceError, expected):
                Dataset(info)

    def test_asset_path(self):
        with TemporaryDirectory() as root:
            self.create_dataset_files(root)
            result = Dataset.read_directory(root).asset_path
            self.assertEqual(result, root)

    def test_asset_name(self):
        with TemporaryDirectory() as root:
            self.create_dataset_files(root)
            result = Dataset.read_directory(root).asset_name
            self.assertEqual(result, Path(root).name)

    def test_stats(self):
        pass

    def test_get_stats(self):
        pass

    def test_repr(self):
        pass

    def test_load(self):
        pass