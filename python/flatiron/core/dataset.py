from typing import Any, Optional, Union  # noqa F401
from flatiron.core.types import Filepath, OptArray, OptInt, OptLabels  # noqa F401

from pathlib import Path
import os
import random
import re

from lunchbox.enforce import Enforce
from tqdm.notebook import tqdm
import cv_depot.api as cvd
import humanfriendly as hf
import numpy as np
import pandas as pd

import flatiron.core.tools as fict
# ------------------------------------------------------------------------------


class Dataset:
    @classmethod
    def read_csv(cls, filepath, **kwargs):
        # type: (Filepath, Any) -> Dataset
        '''
        Construct Dataset instance from given csv filepath.

        Args:
            filepath (str or Path): Info CSV filepath.

        Raises:
            EnforceError: If filepath does not exist or is not a CSV.

        Returns:
            Dataset: Dataset instance.
        '''
        fp = Path(filepath)
        msg = f'Filepath does not exist: {fp}'
        Enforce(fp.is_file(), '==', True, message=msg)

        msg = f'Filepath extension must be csv. Given filepath: {fp}'
        Enforce(fp.suffix.lower()[1:], '==', 'csv', message=msg)
        # ----------------------------------------------------------------------

        info = pd.read_csv(filepath)
        return cls(info, **kwargs)

    @classmethod
    def read_directory(cls, directory, **kwargs):
        # type: (Filepath, Any) -> Dataset
        '''
        Construct dataset from directory.

        Args:
            directory (str or Path): Dataset directory.

        Raises:
            EnforceError: If directory does not exist.
            EnforceError: If more or less than 1 CSV file found in directory.

        Returns:
            Dataset: Dataset instance.
        '''
        msg = f'Directory does not exist: {directory}'
        Enforce(Path(directory).is_dir(), '==', True, message=msg)

        files = [Path(directory, x) for x in os.listdir(directory)]  # type: Any
        files = list(filter(lambda x: x.suffix.lower()[1:] == 'csv', files))
        files = sorted([x.as_posix() for x in files])
        msg = 'Dataset directory must contain only 1 CSV file. '
        msg += f'CSV files found: {files}'
        Enforce(len(files), '==', 1, message=msg)
        # ----------------------------------------------------------------------

        return cls.read_csv(files[0], **kwargs)

    def __init__(
        self, info, ext_regex='npy|exr|png|jpeg|jpg|tiff', calc_file_size=True,
        labels=None, label_axis=-1
    ):
        # type: (pd.DataFrame, str, bool, OptLabels, int) -> None
        '''
        Construct a Dataset instance.
        If labels is an integer it will assumed to be an axis which the
        data will be split upon.

        Args:
            info (pd.DataFrame): Info DataFrame.
            ext_regex (str, optional): File extension pattern.
                Default: 'npy|exr|png|jpeg|jpg|tiff'.
            calc_file_size (bool, optional): Calculate file size in GB.
                Default: True.
            labels (object, optional): Label channels. Default: None.
            label_axis (int, optional): Label axis. Default: -1.

        Raises:
            EnforceError: If info is not an instance of DataFrame.
            EnforceError: If required columns not found in info.
        '''
        Enforce(info, 'instance of', pd.DataFrame)

        # columns
        columns = info.columns.tolist()
        required = ['asset_path', 'filepath_relative']
        diff = sorted(list(set(required).difference(columns)))
        msg = f'Required columns not found in info: {diff}'
        Enforce(len(diff), '==', 0, message=msg)

        # root path
        root = info.asset_path.unique().tolist()
        msg = f'Info must contain only 1 root path. Paths found: {root}'
        Enforce(len(root), '==', 1, message=msg)
        root = root[0]
        msg = f'Directory does not exist: {root}'
        Enforce(Path(root).is_dir(), '==', True, message=msg)

        # info files
        info['filepath'] = info.filepath_relative \
            .apply(lambda x: Path(root, x).as_posix())
        mask = info.filepath.apply(lambda x: not Path(x).is_file())
        absent = info.loc[mask, 'filepath'].tolist()
        msg = f'Files do not exist: {absent}'
        Enforce(len(absent), '==', 0, message=msg)

        # extension
        mask = info.filepath \
            .apply(lambda x: Path(x).suffix.lower()[1:]) \
            .apply(lambda x: re.search(ext_regex, x, re.I) is None)
        bad_ext = sorted(info.loc[mask, 'filepath'].tolist())
        msg = f'Found files extensions that do not match ext_regex: {bad_ext}'
        Enforce(len(bad_ext), '==', 0, message=msg)

        # frame indicators
        frame_regex = r'_(f|c)(\d+)\.' + f'({ext_regex})$'
        mask = info.filepath.apply(lambda x: re.search(frame_regex, x) is None)
        bad_frames = info.loc[mask, 'filepath'].tolist()
        msg = 'Found files missing frame indicators. '
        msg += f"File names must match '{frame_regex}'. "
        msg += f'Invalid frames: {bad_frames}'
        Enforce(len(bad_frames), '==', 0, message=msg)

        # frame column
        info['frame'] = info.filepath \
            .apply(lambda x: re.search(frame_regex, x).group(2)).astype(int)  # type: ignore

        # gb column
        info['gb'] = np.nan
        if calc_file_size:
            info['gb'] = info.filepath \
                .apply(lambda x: os.stat(x).st_size / 10**9)

        # loaded column
        info['loaded'] = False
        # ----------------------------------------------------------------------

        # reorganize columns
        cols = [
            'gb', 'frame', 'asset_path', 'filepath_relative', 'filepath',
            'loaded'
        ]
        cols = cols + info.drop(cols, axis=1).columns.tolist()
        info = info[cols]

        self._info = info  # type: pd.DataFrame
        self.data = None  # type: OptArray
        self.labels = labels
        self.label_axis = label_axis
        self._ext_regex = ext_regex
        self._calc_file_size = calc_file_size
        self._sample_gb = np.nan  # type: Union[float, np.ndarray]

    @property
    def info(self):
        # type: () -> pd.DataFrame
        '''
        Returns:
            DataFrame: Copy of info DataFrame.
        '''
        return self._info.copy()

    @property
    def filepaths(self):
        # type: () -> list[str]
        '''
        Returns:
            list[str]: Filepaths sorted by frame.
        '''
        return self._info.sort_values('frame').filepath.tolist()

    @property
    def asset_path(self):
        # type: () -> str
        '''
        Returns:
            str: Asset path of Dataset.
        '''
        return self.info.loc[0, 'asset_path']

    @property
    def asset_name(self):
        # type: () -> str
        '''
        Returns:
            str: Asset name of Dataset.
        '''
        return Path(self.asset_path).name

    @property
    def stats(self):
        # type: () -> pd.DataFrame
        '''
        Generates a table of statistics of info data.

        Metrics include:

        * min
        * max
        * mean
        * std
        * loaded
        * total

        Units include:

        * gb
        * frame
        * sample

        Returns:
            DataFrame: Table of statistics.
        '''
        info = self.info
        a = self._get_stats(info)
        b = self._get_stats(info.loc[info.loaded]) \
            .loc[['total']].rename(lambda x: 'loaded')
        stats = pd.concat([a, b])
        stats['sample'] = np.nan

        if self.data is not None:
            loaded = round(self.data.nbytes / 10**9, 2)
            stats.loc['loaded', 'gb'] = loaded

            # sample stats
            total = info['gb'].sum() / self._sample_gb
            stats.loc['loaded', 'sample'] = self.data.shape[0]
            stats.loc['total', 'sample'] = total
            stats.loc['mean', 'sample'] = total / info.shape[0]
            stats['sample'] = stats['sample'].apply(lambda x: round(x, 0))

        index = ['min', 'max', 'mean', 'std', 'loaded', 'total']
        stats = stats.loc[index]
        return stats

    @staticmethod
    def _get_stats(info):
        # type: (pd.DataFrame) -> pd.DataFrame
        '''
        Creates table of statistics from given info DataFrame.

        Args:
            info (pd.DataFrame): Info DataFrame.

        Returns:
            pd.DataFrame: Stats DataFrame.
        '''
        stats = info.describe()
        rows = ['min', 'max', 'mean', 'std', 'count']
        stats = stats.loc[rows]
        stats.loc['total'] = info[stats.columns].sum()
        stats.loc['total', 'frame'] = stats.loc['count', 'frame']
        stats.loc['mean', 'frame'] = np.nan
        stats.loc['std', 'frame'] = np.nan
        stats = stats.map(lambda x: round(x, 2))
        stats.drop('count', inplace=True)
        return stats

    def __repr__(self):
        # type: () -> str
        '''
        Returns:
            str: Info statistics.
        '''
        msg = f'''
        <Dataset>
            ASSET_NAME: {self.asset_name}
            ASSET_PATH: {self.asset_path}
            STATS:
                  '''[1:]
        msg = fict.unindent(msg, spaces=8)
        cols = ['gb', 'frame', 'sample']
        stats = str(self.stats[cols])
        stats = '\n          '.join(stats.split('\n'))
        msg = msg + stats
        return msg

    def __len__(self):
        # tyope: () -> int
        '''
        Returns:
            int: Number of frames.
        '''
        return len(self._info)

    def __getitem(self, frame):
        # type: (int) -> Any
        '''
        Get data by frame.
        Thisi is needed to avoid recursion errors when overloading __getitem__.

        Raises:
            IndexError: If frame is missing or multiple frames were found.

        Returns:
            object: Data of given frame.
        '''
        return self._read_file(self.get_filepath(frame))

    def __getitem__(self, frame):
        # type: (int) -> Any
        '''
        Get data by frame.

        Raises:
            IndexError: If frame is missing or multiple frames were found.

        Returns:
            object: Data of given frame.
        '''
        return self.__getitem(frame)

    def get_filepath(self, frame):
        # type: (int) -> Any
        '''
        Get filepath of given frame.

        Raises:
            IndexError: If frame is missing or multiple frames were found.

        Returns:
            str: Filepath of given frame.
        '''
        info = self._info
        mask = info.frame == frame
        filepaths = info.loc[mask, 'filepath'].tolist()
        if len(filepaths) == 0:
            raise IndexError(f'Missing frame {frame}.')
        elif len(filepaths) > 1:
            raise IndexError(f'Multiple frames found for {frame}.')
        return filepaths[0]

    def get_arrays(self, frame):
        # type: (int) -> list[np.ndarray]
        '''
        Get data and convert into numpy arrays according to labels.

        Args:
            frame (int): Frame.

        Raises:
            IndexError: If frame is missing or multiple frames were found.

        Returns:
            list[np.ndarray]: List of arrays from the given frame.
        '''
        labels = self.labels  # type: Any
        if labels is None or labels == []:
            return [self._read_file_as_array(self.get_filepath(frame))]

        item = self.__getitem(frame)

        # get labels
        if not isinstance(labels, list):
            labels = [labels]

        # if item is numpy array return a np.split
        if isinstance(item, np.ndarray):
            arrays = list(np.split(item, labels, axis=self.label_axis))

        # otherwise item is an Image with channels
        else:
            chan = list(filter(lambda x: x not in labels, item.channels))
            img = item.to_bit_depth(cvd.BitDepth.FLOAT16)
            arrays = [img[:, :, chan].data, img[:, :, labels].data]

        # enforce shape equivalence
        max_dim = max(*[x.ndim for x in arrays])
        output = []
        for array in arrays:
            if array.ndim != max_dim:
                ndim = list(range(max_dim - array.ndim + 1, max_dim))
                array = np.expand_dims(array, axis=ndim)
            output.append(array)
        return output

    def _read_file(self, filepath):
        # type: (str) -> Any
        '''
        Read given file.

        Args:
            filepath (str): Filepath.

        Raises:
            IOError: If extension is not supported.

        Returns:
            object: File content.
        '''
        ext = Path(filepath).suffix[1:].lower()
        if ext == 'npy':
            return np.load(filepath)

        formats = [x.lower() for x in cvd.ImageFormat.__members__.keys()]
        formats += ['jpg']
        if ext in formats:
            return cvd.Image.read(filepath)

        raise IOError(f'Unsupported extension: {ext}')

    def _read_file_as_array(self, filepath):
        # type: (str) -> np.ndarray
        '''
        Read file as numpy array.

        Args:
            filepath (str): Filepath.

        Returns:
            np.ndarray: Array.
        '''
        item = self._read_file(filepath)

        ext = Path(filepath).suffix[1:].lower()
        if ext == 'npy':
            return item
        return item.data

    @staticmethod
    def _resolve_limit(limit):
        # type: (Union[int, str, None]) -> tuple[int, str]
        '''
        Resolves a given limit into a number of samples and limit type.

        Args:
            limit (str, int, None): Limit descriptor.

        Returns:
            tuple[int, str]: Number of samples and limit type.
        '''
        if isinstance(limit, int):
            return limit, 'samples'

        elif isinstance(limit, str):
            return hf.parse_size(limit), 'memory'

        return -1, 'None'

    def load(self, limit=None, shuffle=False, reshape=True):
        # type: (Optional[Union[str, int]], bool, bool) -> Dataset
        '''
        Load data from files.

        Args:
            limit (str or int, optional): Limit data by number of samples or
                memory size. Default: None.
            shuffle (bool, optional): Shuffle frames before loading.
                Default: False.
            reshape (bool, optional): Reshape concatenated data to incorpate
                frames as the first dimension: (FRAME, ...). Analogous to the
                first dimension being batch. Default: True.

        Returns:
            Dataset: self.
        '''
        self.unload()

        # resolve limit
        limit_, limit_type = self._resolve_limit(limit)

        # shuffle rows
        rows = list(self.info.iterrows())
        if shuffle:
            random.shuffle(rows)

        # frame vars
        frames = []
        memory = 0
        samples = 0

        # tqdm message
        desc = 'Loading Dataset Files'
        if limit_type != 'None':
            desc = f'May not total to 100% - {desc}'

        # load frames
        for i, row in tqdm(rows, desc=desc):
            if limit_type == 'samples' and samples >= limit_:
                break
            elif limit_type == 'memory' and memory >= limit_:
                break

            frame = self._read_file_as_array(row.filepath)
            if reshape:
                frame = frame[np.newaxis, ...]
            frames.append(frame)

            self._info.loc[i, 'loaded'] = True
            memory += frame.nbytes
            samples += frame.shape[0]

        # concatenate data
        data = np.concatenate(frames, axis=0)

        # limit array size by samples
        if limit_type == 'samples':
            data = data[:limit_]

        # limit array size by memory
        elif limit_type == 'memory':
            sample_mem = data[0].nbytes
            delta = data.nbytes - limit_
            if delta > 0:
                k = int(delta / sample_mem)
                n = data.shape[0]
                data = data[:n - k]

        # set class members
        self.data = data
        self._sample_gb = data[0].nbytes / 10**9
        return self

    def unload(self):
        # type: () -> Dataset
        '''
        Delete self.data and reset self.info.

        Returns:
            Dataset: self.
        '''
        del self.data
        self.data = None
        self._info['loaded'] = False
        return self

    def xy_split(self):
        # type: () -> tuple[np.ndarray, np.ndarray]
        '''
        Split data into x and y arrays, according to self.labels as the split
        index and self.label_axis as the split axis.

        Raises:
            EnforceError: If data has not been loaded.
            EnforceError: If self.labels is not a list of a single integer.

        Returns:
            tuple[np.ndarray]: x and y arrays.
        '''
        msg = 'Data not loaded. Please call load method.'
        Enforce(self.data, 'instance of', np.ndarray, message=msg)

        msg = 'self.labels must be a list of a single integer. '
        msg += f'Provided labels: {self.labels}.'

        labels = self.labels  # type: Any
        Enforce(labels, 'instance of', list, message=msg)
        Enforce(len(labels), '==', 1, message=msg)
        Enforce(labels[0], 'instance of', int, message=msg)
        # ----------------------------------------------------------------------

        return np.split(self.data, self.labels, axis=self.label_axis)  # type: ignore

    def train_test_split(
        self,
        test_size=0.2,  # type: float
        limit=None,     # type: OptInt
        shuffle=True,   # type: bool
        seed=None,      # type: OptInt
    ):
        # type: (...) -> tuple[Dataset, Dataset]
        '''
        Split into train and test Datasets.

        Args:
            test_size (float, optional): Test set size as a proportion.
                Default: 0.2.
            limit (int, optional): Limit the total length of train and test.
                Default: None.
            shuffle (bool, optional): Randomize data before splitting.
                Default: True.
            seed (float, optional): Seed number between 0 and 1. Default: None.

        Returns:
            tuple[Dataset]: Train Dataset, Test Dataset.
        '''
        train, test = fict.train_test_split(
            self.info,
            test_size=test_size, limit=limit, shuffle=shuffle, seed=seed
        )
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
        kwargs = dict(
            ext_regex=self._ext_regex,
            calc_file_size=self._calc_file_size,
            labels=self.labels,
            label_axis=self.label_axis
        )
        return Dataset(train, **kwargs), Dataset(test, **kwargs)  # type: ignore
