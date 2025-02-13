from pathlib import Path
from tempfile import TemporaryDirectory
import os
import unittest

from lunchbox.enforce import EnforceError
from pandas import DataFrame
import numpy as np

from flatiron.core.dataset import Dataset
import flatiron.core.tools as fict
# ------------------------------------------------------------------------------


class DatasetTests(unittest.TestCase):
    def write_npy(self, target, shape=(10, 10, 3)):
        target = Path(target)
        os.makedirs(target.parent, exist_ok=True)
        array = np.ones(shape, dtype=np.uint8)
        np.save(target, array)

    def create_dataset_files(self, root, shape=(10, 10, 3), indicator='f'):
        os.makedirs(Path(root, 'data'))
        info = DataFrame()
        info['filepath_relative'] = [f'data/foo_{indicator}{i:02d}.npy' for i in range(10)]
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
            expected = 'Dataset directory must contain only 1 CSV file. '
            expected += 'CSV files found:.*/foobar.csv.*info.csv'
            with self.assertRaisesRegex(EnforceError, expected):
                Dataset.read_directory(root)

    def test_init(self):
        with TemporaryDirectory() as root:
            info, _ = self.create_dataset_files(root)
            result = Dataset(info)._info
            cols = [
                'gb', 'frame', 'asset_path', 'filepath_relative',
                'filepath', 'loaded'
            ]
            for col in cols:
                self.assertIn(col, result.columns)

            result = result.columns.tolist()[:6]
            self.assertEqual(result, cols)

            # gb column
            result = int(Dataset(info)._info.gb.sum() * 10**9)
            self.assertEqual(result, 4280)

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
            expected = 'Files do not exist:.*/foo/bar.npy'
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
            expected = 'Found files missing npy extension:.*foo_f03.txt'
            with self.assertRaisesRegex(EnforceError, expected):
                Dataset(info)

    def test_init_frame_indicator(self):
        with TemporaryDirectory() as root:
            info, _ = self.create_dataset_files(root, indicator='f')
            Dataset(info)

        with TemporaryDirectory() as root:
            info, _ = self.create_dataset_files(root, indicator='c')
            Dataset(info)

    def test_init_frame_indicator_error(self):
        with TemporaryDirectory() as root:
            info, _ = self.create_dataset_files(root, indicator='q')
            expected = 'Found files missing frame indicators. '
            expected += r'File names must match.*f\|c'
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

    def test_filepaths(self):
        with TemporaryDirectory() as root:
            self.create_dataset_files(root)
            base = Path(root, 'data')
            filepaths = os.listdir(base)
            expected = sorted([Path(base, x).as_posix() for x in filepaths])
            result = Dataset.read_directory(root).filepaths
            self.assertEqual(result, expected)

    def test_get_stats(self):
        info = DataFrame()
        info['gb'] = [1.1, 1.0, 1.1, 0.5]
        info['frame'] = [0, 1, 2, 3]
        stats = Dataset._get_stats(info)

        exp = info.describe().map(lambda x: round(x, 2))
        exp.loc['total', 'gb'] = info['gb'].sum()
        exp.loc['total', 'frame'] = info['frame'].count()
        exp.loc['mean', 'frame'] = np.nan
        exp.loc['std', 'frame'] = np.nan

        # index
        index = ['min', 'max', 'mean', 'std', 'total']
        exp = exp.loc[index]
        result = stats.index.tolist()
        self.assertEqual(result, index)

        # values
        cols = ['gb', 'frame']
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
            self.create_dataset_files(root, shape=(200, 100, 100, 4))
            result = Dataset.read_directory(root).stats

            # gb
            self.assertEqual(result.loc['loaded', 'gb'], 0)
            self.assertEqual(result.loc['total', 'gb'], 0.08)

            # frame
            self.assertEqual(result.loc['loaded', 'frame'], 0)
            self.assertEqual(result.loc['total', 'frame'], 10)

    def test_stats_loaded(self):
        with TemporaryDirectory() as root:
            self.create_dataset_files(root, shape=(200, 100, 100, 4))
            result = Dataset.read_directory(root).load(limit=500).stats

            # gb
            self.assertEqual(result.loc['loaded', 'gb'], 0.02)
            self.assertEqual(result.loc['total', 'gb'], 0.08)

            # frame
            self.assertEqual(result.loc['loaded', 'frame'], 3)
            self.assertEqual(result.loc['total', 'frame'], 10)

            # sample
            self.assertEqual(result.loc['loaded', 'sample'], 500)

    def test_repr(self):
        with TemporaryDirectory() as root:
            self.create_dataset_files(root)
            dset = Dataset.read_directory(root)
            result = str(dset)
            name = Path(root).name
            expected = f'''
                <Dataset>
                    ASSET_NAME: {name}
                    ASSET_PATH: {root}
                    STATS:
                                   gb  frame  sample
                          min     0.0    0.0     NaN
                          max     0.0    9.0     NaN
                          mean    0.0    NaN     NaN
                          std     0.0    NaN     NaN
                          loaded  0.0    0.0     NaN
                          total   0.0   10.0     NaN'''[1:]
            expected = fict.unindent(expected, spaces=16)
            self.assertEqual(result, expected)

    def test_resolve_limit(self):
        # sample
        result = Dataset._resolve_limit(10)
        self.assertEqual(result, (10, 'samples'))

        # memory
        result = Dataset._resolve_limit('1 gb')
        self.assertEqual(result, (10**9, 'memory'))

        # none
        result = Dataset._resolve_limit(None)
        self.assertEqual(result, (-1, 'None'))

    def test_load(self):
        with TemporaryDirectory() as root:
            shape = (99, 10, 10, 4)
            self.create_dataset_files(root, shape=shape)
            dset = Dataset.read_directory(root).load()

            # data shape
            self.assertEqual(dset.data.shape, (990, 10, 10, 4))

            # sample_gib
            expected = np.ones(shape, dtype=np.uint8)[0].nbytes / 10**9
            self.assertEqual(dset._sample_gb, expected)

            # loaded
            result = dset._info.loaded.unique().tolist()
            self.assertEqual(result, [True])

    def test_load_limit_int(self):
        with TemporaryDirectory() as root:
            shape = (99, 10, 10, 4)
            self.create_dataset_files(root, shape=shape)
            dset = Dataset.read_directory(root).load(limit=200)

            # data shape
            self.assertEqual(dset.data.shape, (200, 10, 10, 4))

            # sample_gib
            expected = np.ones(shape, dtype=np.uint8)[0].nbytes / 10**9
            self.assertEqual(dset._sample_gb, expected)

            # loaded
            result = dset._info['loaded'].tolist()
            expected = ([True] * 3) + ([False] * 7)
            self.assertEqual(result, expected)

    def test_load_limit_str(self):
        with TemporaryDirectory() as root:
            shape = (1000, 100, 100, 3)
            self.create_dataset_files(root, shape=shape)
            dset = Dataset.read_directory(root).load(limit='0.2 gb')

            # data shape
            self.assertEqual(dset.data.shape, (6667, 100, 100, 3))

            # sample_gib
            expected = np.ones(shape, dtype=np.uint8)[0].nbytes / 10**9
            self.assertEqual(dset._sample_gb, expected)

            # loaded
            result = dset._info['loaded'].tolist()
            expected = ([True] * 7) + ([False] * 3)
            self.assertEqual(result, expected)

    def test_load_reload(self):
        with TemporaryDirectory() as root:
            shape = (100, 10, 10, 1)
            self.create_dataset_files(root, shape=shape)
            dset = Dataset.read_directory(root).load()

            # data shape
            self.assertEqual(dset.data.shape, (1000, 10, 10, 1))

            # loaded
            result = dset._info.loaded.unique().tolist()
            self.assertEqual(result, [True])

            # reload
            dset.load(limit=200)

            # data shape
            self.assertEqual(dset.data.shape, (200, 10, 10, 1))

            # loaded
            result = dset._info['loaded'].tolist()
            expected = ([True] * 2) + ([False] * 8)
            self.assertEqual(result, expected)

    def test_load_shuffle(self):
        with TemporaryDirectory() as root:
            shape = (100, 10, 10, 1)
            self.create_dataset_files(root, shape=shape)
            dset = Dataset.read_directory(root)

            a = 99
            b = 99
            for i in range(10):
                a = dset.load(limit=200, shuffle=True).info.loaded.tolist()
                b = dset.load(limit=200, shuffle=True).info.loaded.tolist()
                if a != b:
                    break
            self.assertNotEqual(a, b)

    def test_unload(self):
        with TemporaryDirectory() as root:
            shape = (99, 10, 10, 4)
            self.create_dataset_files(root, shape=shape)
            dset = Dataset.read_directory(root).load().unload()

            # data
            self.assertIs(dset.data, None)

            # loaded
            result = dset._info.loaded.unique().tolist()
            self.assertEqual(result, [False])

    def test_xy_split(self):
        with TemporaryDirectory() as root:
            shape = (100, 10, 10, 4)
            self.create_dataset_files(root, shape=shape)
            dset = Dataset.read_directory(root).load(limit=200)

            # index -1
            x, y = dset.xy_split(-1)
            self.assertEqual(x.shape, (200, 10, 10, 3))
            self.assertEqual(y.shape, (200, 10, 10, 1))

            # index -2
            x, y = dset.xy_split(-2)
            self.assertEqual(x.shape, (200, 10, 10, 2))
            self.assertEqual(y.shape, (200, 10, 10, 2))

            # index -1 axis -2
            x, y = dset.xy_split(-1, axis=-2)
            self.assertEqual(x.shape, (200, 10, 9, 4))
            self.assertEqual(y.shape, (200, 10, 1, 4))

            # index 4 axis -2
            x, y = dset.xy_split(4, axis=-2)
            self.assertEqual(x.shape, (200, 10, 4, 4))
            self.assertEqual(y.shape, (200, 10, 6, 4))

    def test_xy_split_error(self):
        with TemporaryDirectory() as root:
            shape = (100, 10, 10, 4)
            self.create_dataset_files(root, shape=shape)
            dset = Dataset.read_directory(root)
            expected = 'Data not loaded. Please call load method.'
            with self.assertRaisesRegex(EnforceError, expected):
                dset.xy_split(-1)

    def test_train_test_split(self):
        with TemporaryDirectory() as root:
            shape = (50, 10, 10, 5)
            self.create_dataset_files(root, shape=shape)
            dset = Dataset.read_directory(root).load(limit=100)

            # two classes
            x_train, x_test, y_train, y_test = dset \
                .train_test_split(-2, test_size=0.4)
            self.assertEqual(x_train.shape, (60, 10, 10, 3))
            self.assertEqual(x_test.shape, (40, 10, 10, 3))
            self.assertEqual(y_train.shape, (60, 10, 10, 2))
            self.assertEqual(y_test.shape, (40, 10, 10, 2))
