from typing import Any, Optional, Union  # noqa F401
from flatiron.core.types import OptInt  # noqa F401
import numpy as np  # noqa F401

from pathlib import Path

import pandas as pd

from flatiron.core.dataset import Dataset
import flatiron.core.tools as fict
# ------------------------------------------------------------------------------


class MultiDataset:
    '''
    This class combines a dictionary of Dataset instances into a single dataset.
    Datasets are merged by frame.
    '''
    def __init__(self, datasets):
        # type: (dict[str, Dataset]) -> None
        '''
        Constructs a MultiDataset instance.

        Args:
            datasets (dict[str, Dataset]): Dictionary of Dataset instances.
        '''
        self.datasets = datasets

        data = None  # type: Any
        for key, item in sorted(datasets.items()):
            info = item.info
            info['filepath'] = info.apply(
                lambda x: Path(x.asset_path, x.filepath_relative).as_posix(),
                axis=1
            )
            info = info[['frame', 'filepath']]
            if data is None:
                data = info
                prev = key
            else:
                suffix = [f'_{prev}', f'_{key}']
                data = pd.merge(data, info, on='frame', suffixes=suffix)

        self._info = data  # type: pd.DataFrame

    @property
    def info(self):
        # type: () -> pd.DataFrame
        '''
        Returns:
            DataFrame: Copy of info DataFrame.
        '''
        return self._info.copy()

    def __len__(self):
        # tyope: () -> int
        '''
        Returns:
            int: Number of frames.
        '''
        return len(self._info)

    def __getitem__(self, frame):
        # type: (int) -> dict[str, Any]
        '''
        For each dataset, fetch data by given frame.

        Returns:
            dict: Dict where values are data of the given frame.
        '''
        return {k: v[frame] for k, v in self.datasets.items()}

    def get_filepaths(self, frame):
        # type: (int) -> dict[str, str]
        '''
        For each dataset, get filepath of given frame.

        Returns:
            dict: Dict where values are filepaths of the given frame.
        '''
        return {k: v.get_filepath(frame) for k, v in self.datasets.items()}

    def get_arrays(self, frame):
        # type: (int) -> dict[str, list[np.ndarray]]
        '''
        For each dataset, get data and convert into numpy arrays according to
        labels.

        Args:
            frame (int): Frame.

        Raises:
            IndexError: If frame is missing or multiple frames were found.

        Returns:
            dict: Dict where values are lists of arrays from the given frame.
        '''
        return {k: v.get_arrays(frame) for k, v in self.datasets.items()}

    def load(self, limit=None, reshape=True):
        # type: (Optional[Union[str, int]], bool) -> MultiDataset
        '''
        For each dataset, load data from files.

        Args:
            limit (str or int, optional): Limit data by number of samples or
                memory size. Default: None.
            reshape (bool, optional): Reshape concatenated data to incorpate
                frames as the first dimension: (FRAME, ...). Analogous to the
                first dimension being batch. Default: True.

        Returns:
            MultiDataset: self.
        '''
        kwargs = dict(limit=limit, shuffle=False, reshape=reshape)  # type: Any
        [x.load(**kwargs) for x in self.datasets.values()]
        return self

    def unload(self):
        # type: () -> MultiDataset
        '''
        For each dataset, delete self.data and reset self.info.

        Returns:
            MultiDataset: self.
        '''
        [x.unload() for x in self.datasets.values()]
        return self

    def xy_split(self):
        # type: () -> dict[str, tuple[np.ndarray, np.ndarray]]
        '''
        For each dataset, split data into x and y arrays, according to
        self.labels as the split index and self.label_axis as the split axis.

        Raises:
            EnforceError: If data has not been loaded.
            EnforceError: If self.labels is not a list of a single integer.

        Returns:
            dict: Dict where values are x and y arrays.
        '''
        return {k: v.xy_split() for k, v in self.datasets.items()}

    def train_test_split(
        self,
        test_size=0.2,  # type: float
        limit=None,     # type: OptInt
        shuffle=True,   # type: bool
        seed=None,      # type: OptInt
    ):
        # type: (...) -> tuple[MultiDataset, MultiDataset]
        '''
        Split into train and test MultiDatasets.

        Args:
            test_size (float, optional): Test set size as a proportion.
                Default: 0.2.
            limit (int, optional): Limit the total length of train and test.
                Default: None.
            shuffle (bool, optional): Randomize data before splitting.
                Default: True.
            seed (float, optional): Seed number between 0 and 1. Default: None.

        Returns:
            tuple[MultiDataset]: Train MultiDataset, Test MultiDataset.
        '''
        items = fict.train_test_split(
            self.info,
            test_size=test_size, limit=limit, shuffle=shuffle, seed=seed
        )

        msets = dict(train={}, test={})  # type: Any
        for key, val in zip(['train', 'test'], items):
            frames = val.frame.tolist()
            for name, dset in self.datasets.items():
                info = dset._info
                mask = info.frame.apply(lambda x: x in frames)
                info = info[mask].copy()
                info.reset_index(drop=True, inplace=True)
                msets[key][name] = Dataset(
                    info=info,
                    ext_regex=dset._ext_regex,
                    calc_file_size=dset._calc_file_size,
                    labels=dset.labels,
                    label_axis=dset.label_axis,
                )

        return MultiDataset(msets['train']), MultiDataset(msets['test'])
