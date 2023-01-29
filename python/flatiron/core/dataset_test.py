from pathlib import Path
from tempfile import TemporaryDirectory
import os
import unittest

from lunchbox.enforce import EnforceError
from pandas import DataFrame
import numpy as np

from flatiron.core.dataset import Dataset
# ------------------------------------------------------------------------------


class DatasetTests(unittest.TestCase):
    def write_npy(self, target, shape=(10, 10, 3)):
        target = Path(target)
        os.makedirs(target.parent, exist_ok=True)
        array = np.ones(shape)
        np.save(target, array)

    def create_dataset_files(self, root, shape=(10, 10, 3)):
        os.makedirs(Path(root, 'data'))
        info = DataFrame()
        info['filepath_relative'] = [f'data/foo_f{i:02d}.npy' for i in range(10)]
        info['asset_path'] = root
        info.filepath_relative \
            .apply(lambda x: Path(root, x)) \
            .apply(lambda x: self.write_npy(x, shape))
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
                'size_gib', 'chunk', 'asset_path', 'filepath_relative',
                'filepath', 'loaded'
            ]
            for col in cols:
                self.assertIn(col, result.columns)

            result = result.columns.tolist()[:6]
            self.assertEqual(result, cols)

            # size_gib column
            result = Dataset(info)._info.size_gib.sum()
            self.assertLess(result, 3.0e-05)
            self.assertGreater(result, 2.0e-05)

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

    def test_load(self):
        with TemporaryDirectory() as root:
            self.create_dataset_files(root)

    def test_get_stats(self):
        info = DataFrame()
        info['size_gib'] = [1.1, 1.0, 1.1, 0.5]
        info['chunk'] = [0, 1, 2, 3]
        stats = Dataset._get_stats(info)

        exp = info.describe().applymap(lambda x: round(x, 2))
        exp.loc['total', 'size_gib'] = info['size_gib'].sum()
        exp.loc['total', 'chunk'] = info['chunk'].count()
        exp.loc['mean', 'chunk'] = np.nan
        exp.loc['std', 'chunk'] = np.nan

        # index
        index = ['min', 'max', 'mean', 'std', 'total']
        exp = exp.loc[index]
        result = stats.index.tolist()
        self.assertEqual(result, index)

        # values
        cols = ['size_gib', 'chunk']
        for col in cols:
            for i in index:
                result = stats.loc[i, col]
                expected = exp.loc[i, col]
                if np.isnan(expected):
                    self.assertTrue(np.isnan(result))
                else:
                    self.assertEqual(result, expected)

    def test_stats_unloaded(self):
        with TemporaryDirectory() as root:
            self.create_dataset_files(root, shape=(150, 100, 100, 4))
            result = Dataset.read_directory(root).stats

            # size_gib
            self.assertEqual(result.loc['loaded_count', 'size_gib'], 0)
            self.assertEqual(result.loc['loaded_total', 'size_gib'], 0)
            self.assertEqual(result.loc['count', 'size_gib'], 10)
            self.assertEqual(result.loc['total', 'size_gib'], 0.48)

            # chunk
            self.assertEqual(result.loc['loaded_count', 'chunk'], 0)
            self.assertEqual(result.loc['loaded_total', 'chunk'], 0)
            self.assertEqual(result.loc['count', 'chunk'], 10)
            self.assertEqual(result.loc['total', 'chunk'], 45)

    def test_stats_loaded(self):
        with TemporaryDirectory() as root:
            self.create_dataset_files(root, shape=(150, 100, 100, 4))
            result = Dataset.read_directory(root).load(limit=200).stats

            # size_gib
            self.assertEqual(result.loc['loaded_total', 'size_gib'], 0.06)
            self.assertEqual(result.loc['total', 'size_gib'], 0.48)

            # chunk
            self.assertEqual(result.loc['loaded_total', 'chunk'], 0)
            self.assertEqual(result.loc['total', 'chunk'], 45)

            # sample
            self.assertEqual(result.loc['loaded_total', 'chunk'], 200)

    def test_repr(self):
        pass

    def test_load(self):
        pass
