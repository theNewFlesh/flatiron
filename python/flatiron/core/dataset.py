from typing import Any, List, Optional, Union
import pandas as pd

from pathlib import Path
import os
import re

from lunchbox.enforce import Enforce
import humanfriendly as hf
import numpy as np
import pandas as pd

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
        chunk_regex = r'_f(\d+)\.npy$'
        mask = info.filepath.apply(lambda x: re.search(chunk_regex, x) is None)
        bad_chunk = info.loc[mask, 'filepath'].tolist()
        msg = 'Found chunk files missing chunk indicators. '
        msg += r"File names must match '_f\d+\.npy'. "
        msg += f'Invalid chunks: {bad_chunk}'
        Enforce(len(bad_chunk), '==', 0, message=msg)

        # chunk column
        info['chunk'] = info.filepath \
            .apply(lambda x: re.search(chunk_regex, x).group(1)).astype(int)  # type: ignore
        info['size_gib'] = info.filepath \
            .apply(lambda x: os.stat(x).st_size / 10**9)

        # loaded column
        info['loaded'] = False
        # ----------------------------------------------------------------------

        # reorganize columns
        cols = [
            'size_gib', 'chunk', 'asset_path', 'filepath_relative', 'filepath',
            'loaded'
        ]
        cols = info.drop(cols, axis=1).columns.tolist() + cols
        info = info[cols]

        self._info = info

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
        * loaded_count
        * count
        * loaded_total
        * total

        Returns:
            DataFrame: Table of statistics.
        '''
        info = self.info
        a = self._get_stats(info)
        b = self._get_stats(info.loc[info.loaded]) \
            .loc[['count', 'total']] \
            .rename(lambda x: f'loaded_{x}')
        stats = pd.concat([a, b])
        index = [
            'min', 'max', 'mean', 'std', 'loaded_count', 'count',
            'loaded_total', 'total'
        ]
        stats = stats.loc[index]
        return stats

    def _get_stats(self, info):
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
        stats = stats.applymap(lambda x: round(x, 2))
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
        cols = ['size_gib', 'chunk']
        stats = str(self.stats[cols])
        stats = '\n          '.join(stats.split('\n'))
        msg = msg + stats
        return msg

    def load(self, limit=None, shuffle=False):
        # type: (Optional[Union[str, int]], bool) -> None
        '''
        Load data from numpy files.

        Args:
            limit (str or int, optional): Limit data by row count or memory
                size. Default: None.
            shuffle (bool, optional): Shuffle chunks before loading.
                Default: False.
        '''
        if isinstance(limit, str):
            limit = hf.parse_size(limit)
        info = self._info