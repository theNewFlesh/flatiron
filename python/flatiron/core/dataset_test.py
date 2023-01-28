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
        data = DataFrame()
        data['filepath_relative'] = [f'data/foo-{i:02d}.npy' for i in range(10)]
        data['root_path'] = root
        data.filepath_relative.apply(lambda x: Path(root, x).touch())
        info_path = Path(root, 'info.csv').as_posix()
        data.to_csv(info_path, index=None)
        return data, info_path

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
            data, _ = self.create_dataset_files(root)
            result = Dataset(data)._data
            cols = ['root_path', 'filepath_relative', 'filepath']
            for col in cols:
                self.assertIn(col, result.columns)

            result = result.columns.tolist()[-3:]
            self.assertEqual(result, cols)

    def test_init_columns_error(self):
        with TemporaryDirectory() as root:
            data, _ = self.create_dataset_files(root)
            cols = ['root_path', 'filepath_relative']
            for col in cols:
                temp = data.drop(col, axis=1)
                expected = f'Required columns not found in data:.*{col}'
                with self.assertRaisesRegex(EnforceError, expected):
                    Dataset(temp)

    def test_init_root_error(self):
        with TemporaryDirectory() as root:
            data, _ = self.create_dataset_files(root)
            data.loc[3, 'root_path'] = '/foo/bar'
            expected = 'Data must contain only 1 root path. '
            expected += f'Paths found:.*{root}.*/foo/bar'
            with self.assertRaisesRegex(EnforceError, expected):
                Dataset(data)

            data['root_path'] = '/foo/bar'
            expected = 'Directory does not exist: /foo/bar'
            with self.assertRaisesRegex(EnforceError, expected):
                Dataset(data)

    def test_init_filepath_error(self):
        with TemporaryDirectory() as root:
            data, _ = self.create_dataset_files(root)
            data.loc[3, 'filepath_relative'] = '/foo/bar.npy'
            expected = 'Files listed in data do not exist:.*/foo/bar.npy'
            with self.assertRaisesRegex(EnforceError, expected):
                Dataset(data)

    def test_init_extension_error(self):
        with TemporaryDirectory() as root:
            data, _ = self.create_dataset_files(root)
            src = data.loc[3, 'filepath_relative']
            src = Path(root, src).as_posix()
            tgt = src.replace('npy', 'txt')
            os.rename(src, tgt)
            data.loc[3, 'filepath_relative'] = tgt
            expected = 'Data lists files without npy extension:.*foo-03.txt'
            with self.assertRaisesRegex(EnforceError, expected):
                Dataset(data)
