from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
import os

import pandas as pd

from flatiron.core.dataset import Dataset
from flatiron.core.dataset_test import DatasetTestBase
from flatiron.core.multidataset import MultiDataset
# ------------------------------------------------------------------------------


class MultiDatasetTests(DatasetTestBase):
    def get_datasets(self, root):
        exts = ['png', 'jpeg', 'tiff']
        dirs = [Path(root, x).as_posix() for x in exts]
        data = {}
        for exts, dir_ in zip(exts, dirs):
            os.makedirs(dir_)
            self.create_image_dataset_files(dir_, extension=exts)
            data[exts] = Dataset.read_directory(
                dir_, labels=[2], label_axis=-1
            )
        return data

    def test_init(self):
        with TemporaryDirectory() as root:
            datasets = self.get_datasets(root)
            result = MultiDataset(datasets)

            self.assertIs(result.datasets, datasets)

            expected = ['frame', 'filepath_jpeg', 'filepath_png', 'filepath']
            self.assertEqual(result._info.columns.tolist(), expected)

            expected = datasets['png'].info.frame.tolist()
            self.assertEqual(result._info.frame.tolist(), expected)

    def test_init_single(self):
        with TemporaryDirectory() as root:
            datasets = self.get_datasets(root)
            datasets = dict(png=datasets['png'])
            result = MultiDataset(datasets)

            self.assertIs(result.datasets, datasets)

            expected = ['frame', 'filepath']
            self.assertEqual(result._info.columns.tolist(), expected)

            expected = datasets['png'].info.frame.tolist()
            self.assertEqual(result._info.frame.tolist(), expected)

    def test_info(self):
        with TemporaryDirectory() as root:
            datasets = self.get_datasets(root)
            result = MultiDataset(datasets)
            self.assertIsInstance(result.info, pd.DataFrame)
            self.assertIsNot(result.info, result._info)

    def test_len(self):
        with TemporaryDirectory() as root:
            datasets = self.get_datasets(root)
            result = MultiDataset(datasets)
            self.assertEqual(len(result), len(result._info))

    def test_getitem(self):
        with TemporaryDirectory() as root:
            datasets = self.get_datasets(root)
            data = MultiDataset(datasets)

            expected = dict(
                png=datasets['png'][3],
                jpeg=datasets['jpeg'][3],
                tiff=datasets['tiff'][3],
            )
            result = data[3]
            self.assertEqual(result, expected)

    def test_get_filepaths(self):
        with TemporaryDirectory() as root:
            datasets = self.get_datasets(root)
            data = MultiDataset(datasets)

            expected = dict(
                png=datasets['png'].get_filepath(3),
                jpeg=datasets['jpeg'].get_filepath(3),
                tiff=datasets['tiff'].get_filepath(3),
            )
            result = data.get_filepaths(3)
            self.assertEqual(result, expected)

    def test_get_arrays(self):
        with TemporaryDirectory() as root:
            datasets = self.get_datasets(root)
            data = MultiDataset(datasets)

            expected = dict(
                png=datasets['png'].get_arrays(3),
                jpeg=datasets['jpeg'].get_arrays(3),
                tiff=datasets['tiff'].get_arrays(3),
            )
            result = data.get_arrays(3)
            self.assertEqual(result.keys(), expected.keys())

            result = [type(x) for x in result.values()]
            expected = [type(x) for x in expected.values()]
            self.assertEqual(result, expected)

    def test_load(self):
        with TemporaryDirectory() as root:
            datasets = self.get_datasets(root)
            result = MultiDataset(datasets)
            ld = [x.info.loaded.unique().tolist() for x in result.datasets.values()]
            ld = set(chain(*ld))
            self.assertEqual(ld, {False})

            result.load()
            ld = [x.info.loaded.unique().tolist() for x in result.datasets.values()]
            ld = set(chain(*ld))
            self.assertEqual(ld, {True})

    def test_unload(self):
        with TemporaryDirectory() as root:
            datasets = self.get_datasets(root)
            result = MultiDataset(datasets)

            result.load()
            ld = [x.info.loaded.unique().tolist() for x in result.datasets.values()]
            ld = set(chain(*ld))
            self.assertEqual(ld, {True})

            result.unload()
            ld = [x.info.loaded.unique().tolist() for x in result.datasets.values()]
            ld = set(chain(*ld))
            self.assertEqual(ld, {False})

    def test_xy_split(self):
        with TemporaryDirectory() as root:
            datasets = self.get_datasets(root)
            data = MultiDataset(datasets).load()

            expected = dict(
                png=datasets['png'].xy_split(),
                jpeg=datasets['jpeg'].xy_split(),
                tiff=datasets['tiff'].xy_split(),
            )
            result = data.xy_split()
            self.assertEqual(result.keys(), expected.keys())

            result = [(x.shape, y.shape) for x, y in result.values()]
            expected = [(x.shape, y.shape) for x, y in expected.values()]
            self.assertEqual(result, expected)

    def test_train_test_split(self):
        with TemporaryDirectory() as root:
            datasets = self.get_datasets(root)
            train, test = MultiDataset(datasets).train_test_split()

            self.assertEqual(len(train), 8)
            self.assertEqual(len(test), 2)
