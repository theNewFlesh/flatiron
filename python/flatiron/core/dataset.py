from typing import Any, List, Optional, Tuple, Union  # noqa F401

from pathlib import Path
import os
import random
import re

from lunchbox.enforce import Enforce
from tqdm.notebook import tqdm
import humanfriendly as hf
import numpy as np
import pandas as pd
import sklearn.model_selection as skm

import flatiron.core.tools as fict

Filepath = Union[str, Path]
# ------------------------------------------------------------------------------


class Dataset:
    @classmethod
    def read_csv(cls, filepath):
        # type: (Filepath) -> Dataset
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
        return cls(info)

    @classmethod
    def read_directory(cls, directory):
        # type: (Union[str, Path]) -> Dataset
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
        files = [x.as_posix() for x in files]
        msg = 'Dataset directory must contain only 1 CSV file. '
        msg += f'CSV files found: {files}'
        Enforce(len(files), '==', 1, message=msg)
        # ----------------------------------------------------------------------

        return cls.read_csv(files[0])

    def __init__(self, info):
        # type: (pd.DataFrame) -> None
        '''
        Construct a Dataset instance.

        Args:
            info (pd.DataFrame): Info DataFrame.

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
        msg = f'Chunk files do not exist: {absent}'
        Enforce(len(absent), '==', 0, message=msg)

        # npy extension
        mask = info.filepath \
            .apply(lambda x: Path(x).suffix.lower()[1:]) \
            .apply(lambda x: x != 'npy')
        bad_ext = sorted(info.loc[mask, 'filepath'].tolist())
        msg = f'Found chunk files missing npy extension: {bad_ext}'
        Enforce(len(bad_ext), '==', 0, message=msg)

        # chunk indicators
        chunk_regex = r'_(f|c)(\d+)\.npy$'
        mask = info.filepath.apply(lambda x: re.search(chunk_regex, x) is None)
        bad_chunk = info.loc[mask, 'filepath'].tolist()
        msg = 'Found chunk files missing chunk indicators. '
        msg += r"File names must match '_(f|c)\d+\.npy'. "
        msg += f'Invalid chunks: {bad_chunk}'
        Enforce(len(bad_chunk), '==', 0, message=msg)

        # chunk column
        info['chunk'] = info.filepath \
            .apply(lambda x: re.search(chunk_regex, x).group(2)).astype(int)  # type: ignore
        info['GB'] = info.filepath \
            .apply(lambda x: os.stat(x).st_size / 10**9)

        # loaded column
        info['loaded'] = False
        # ----------------------------------------------------------------------

        # reorganize columns
        cols = [
            'GB', 'chunk', 'asset_path', 'filepath_relative', 'filepath',
            'loaded'
        ]
        cols = cols + info.drop(cols, axis=1).columns.tolist()
        info = info[cols]

        self._info = info  # type: pd.DataFrame
        self.data = None  # type: Optional[np.ndarray]
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
    def chunks(self):
        # type: () -> List[str]
        '''
        Returns:
            list[str]: Chunk filepaths.
        '''
        return self._info.sort_values('chunk').filepath.tolist()

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

        * GB
        * chunk
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
            stats.loc['loaded', 'GB'] = loaded

            # sample stats
            total = info['GB'].sum() / self._sample_gb
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
        stats.loc['total', 'chunk'] = stats.loc['count', 'chunk']
        stats.loc['mean', 'chunk'] = np.nan
        stats.loc['std', 'chunk'] = np.nan
        stats = stats.applymap(lambda x: round(x, 2))
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
        cols = ['GB', 'chunk', 'sample']
        stats = str(self.stats[cols])
        stats = '\n          '.join(stats.split('\n'))
        msg = msg + stats
        return msg

    @staticmethod
    def _resolve_limit(limit):
        # type: (Union[int, str, None]) -> Tuple[int, str]
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

    def load(self, limit=None, shuffle=False):
        # type: (Optional[Union[str, int]], bool) -> Dataset
        '''
        Load data from chunk files.

        Args:
            limit (str or int, optional): Limit data by number of samples or
                memory size. Default: None.
            shuffle (bool, optional): Shuffle chunks before loading.
                Default: False.

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

        # chunk vars
        chunks = []
        memory = 0
        samples = 0

        # tqdm message
        desc = 'Loading Dataset Chunks'
        if limit_type != 'None':
            desc = f'May not total to 100% - {desc}'

        # load chunks
        for i, row in tqdm(rows, desc=desc):
            if limit_type == 'samples' and samples >= limit_:
                break
            elif limit_type == 'memory' and memory >= limit_:
                break

            chunk = np.load(row.filepath)
            chunks.append(chunk)

            self._info.loc[i, 'loaded'] = True
            memory += chunk.nbytes
            samples += chunk.shape[0]

        # concatenate data
        data = np.concatenate(chunks, axis=0)

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

    def xy_split(self, index, axis=-1):
        # type: (int, int) -> Tuple[np.ndarray, np.ndarray]
        '''
        Split data into x and y arrays.
        Index and axis support negative ingegers.

        Args:
            index (int): Index of axis to split on.
            axis (int, optional): Axis to split data on. Default: -1.

        Raises:
            EnforceError: If data has not been loaded.

        Returns:
            tuple[np.ndarray]: X and y arrays.
        '''
        msg = 'Data not loaded. Please call load method.'
        Enforce(self.data, 'instance of', np.ndarray, message=msg)
        # ----------------------------------------------------------------------

        return np.split(self.data, [index], axis=axis)  # type: ignore

    def train_test_split(
        self,
        index,  # type: int
        axis=-1,  # type: int
        test_size=0.2,  # type: Optional[Union[float, int]]
        train_size=None,  # type: Optional[Union[float, int]]
        random_state=42,  # type: Optional[int]
        shuffle=True,  # type: bool
        stratify=None,  # type: Optional[np.ndarray]
    ):
        # type: (...) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        '''
        Split data into x_train, x_test, y_train, y_test arrays.

        Args:
            index (int): Index of axis to split on.

            axis (int, optional): Axis to split data on. Default: -1.

            test_size (float or int, optional): If float, should be between 0.0
                and 1.0 and represent the proportion of the dataset to include
                in the test split. If int, represents the absolute number of
                test samples. If None, the value is set to the complement of the
                train size. If ``train_size`` is also None, it will be set to
                0.25. Default: 0.2

            train_size (float or int, optional): If float, should be between 0.0
                and 1.0 and represent the proportion of the dataset to include
                in the train split. If int, represents the absolute number of
                train samples. If None, the value is automatically set to the
                complement of the test size. Default: None.

            random_state (int, optional): Controls the shuffling applied to the
                data before applying the split. Pass an int for reproducible
                output across multiple function calls. Default: 42.

            shuffle (bool, optional): Whether or not to shuffle the data before
                splitting. If False then stratify must be None. Default: True.

            stratify (np.ndarr, optional): If not None, data is split in a
                stratified fashion, using this as the class labels.
                Default: None.

        Returns:
            tuple[np.ndarray]: x_train, x_test, y_train, y_test
        '''
        x, y = self.xy_split(index, axis=axis)
        x_train, x_test, y_train, y_test = skm.train_test_split(
            x, y,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )
        return x_train, x_test, y_train, y_test
