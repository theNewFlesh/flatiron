from typing import Any, Union

from pathlib import Path
import os

from lunchbox.enforce import Enforce
import pandas as pd

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

        data = pd.read_csv(filepath)
        return cls(data)

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

    def __init__(self, data):
        # type: (pd.DataFrame) -> None
        '''
        Construct a Dataset instance.

        Args:
            data (pd.DataFrame): Dataset data.

        Raises:
            EnforceError: If data is not an instance of DataFrame.
            EnforceError: If required columns not found in data.
        '''
        Enforce(data, 'instance of', pd.DataFrame)

        # columns
        columns = data.columns.tolist()
        required = ['root_path', 'filepath_relative']
        diff = sorted(list(set(required).difference(columns)))
        msg = f'Required columns not found in data: {diff}'
        Enforce(len(diff), '==', 0, message=msg)

        # root path
        root = data.root_path.unique().tolist()
        msg = f'Data must contain only 1 root path. Paths found: {root}'
        Enforce(len(root), '==', 1, message=msg)
        root = root[0]
        msg = f'Directory does not exist: {root}'
        Enforce(Path(root).is_dir(), '==', True, message=msg)

        # data files
        data['filepath'] = data.filepath_relative \
            .apply(lambda x: Path(root, x).as_posix())
        mask = data.filepath.apply(lambda x: not Path(x).is_file())
        absent = data.loc[mask, 'filepath'].tolist()
        msg = f'Files listed in data do not exist: {absent}'
        Enforce(len(absent), '==', 0, message=msg)

        # npy extension
        mask = data.filepath \
            .apply(lambda x: Path(x).suffix.lower()[1:]) \
            .apply(lambda x: x != 'npy')
        bad_ext = sorted(data.loc[mask, 'filepath'].tolist())
        msg = f'Data lists files without npy extension: {bad_ext}'
        Enforce(len(bad_ext), '==', 0, message=msg)
        # ----------------------------------------------------------------------

        # reorganize columns
        cols = ['root_path', 'filepath_relative', 'filepath']
        cols = data.drop(cols, axis=1).columns.tolist() + cols
        data = data[cols]

        self._data = data
