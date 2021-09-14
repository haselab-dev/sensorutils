"""Daily and Sports Activity Data Set

https://archive.ics.uci.edu/ml/datasets/daily+and+sports+activities
"""

import itertools
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from .base import check_path


__all__ = ['load', 'load_raw']

# Meta Info
Sampling_Rate = 25 # Hz
COLUMNS = [
    f'{pos}_{axis}{sensor}' \
        for pos in ['T', 'RA', 'LA', 'RL', 'LL'] \
        for sensor in ['acc', 'gyro', 'mag'] \
        for axis in ['x', 'y', 'z']
]# + ['activity', 'subject']    # ローダで付与される

SUBJECTS = list(range(1, 9))    # range of subject id: [1, 8]

ACTIVITIES_NAMES = (
    'sitting (A1)',
    'standing (A2)',
    'lying on back (A3)',
    'lying on right side (A4)',
    'ascending stairs (A5)',
    'descending stairs (A6)',
    'standing in an elevator still (A7)',
    'moving around in an elevator (A8)',
    'walking in a parking lot (A9)',
    'walking on a treadmill with a speed of 4 km/h in flat (A10)',
    'walking on a treadmill with a speed of 4 km/h in 15 deg inclined positions (A11)',
    'running on a treadmill with a speed of 8 km/h (A12)',
    'exercising on a stepper (A13)',
    'exercising on a cross trainer (A14)',
    'cycling on an exercise bike in horizontal (A15)',
    'cycling on an exercise bike in vertical positions (A16)',
    'rowing (A17)',
    'jumping (A18)',
    'playing basketball (A19)',
)
ACTIVITIES = dict(
    zip(range(18+1), ACTIVITIES_NAMES)
)


def load(path:Union[Path,str]) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """Function for loading DSADS dataset

    Parameters
    ----------
    path: Union[Path, str]
        Directory path of DSADS dataset.

    Returns
    -------
    data, meta: List[pd.DataFrame], List[pd.DataFrame]
        Sensor data segmented by activity and subject.

    See Alos
    --------
    The order of 'data' and 'meta' correspond.

    e.g. meta[0] is meta data of data[0].
    """
    path = check_path(path)

    raw = load_raw(path)
    data, meta = reformat(raw)
    return data, meta


def load_raw(path:Path) -> List[pd.DataFrame]:
    """Function for loading raw data of DSADS dataset

    Parameters
    ----------
    path: Path
        Directory path of DSADS dataset.

    Returns
    -------
    chunks_per_persons: List[pd.DataFrame]
        Raw data of DSADS dataset.

        Each item in 'chunks' is a part of dataset, which is splited by activity and subject.
    """

    def _load_raw_data(path):
        try:
            seg = pd.read_csv(str(path), header=None)
            seg.columns = COLUMNS
            activity_id = int(path.parent.parent.stem[1:])
            subject_id = int(path.parent.stem[1:])
            if activity_id not in list(range(1, 20)):
                raise RuntimeError('Detected unknown file (invalid activity id: {}), {}'.format(activity_id, path))
            seg['activity'] = activity_id-1   # [1, 19] -> [0, 18]
            seg['subject'] = subject_id
            dtypes = dict(zip(COLUMNS, list(np.float64 for _ in COLUMNS)))
            dtypes['activity'] = np.int8
            dtypes['subject'] = np.int8
            seg = seg.astype(dtypes)
            return seg
        except FileNotFoundError:
            print(f'[load] {path} not found')
        except pd.errors.EmptyDataError:
            print(f'[load] {path} is empty')
        return pd.DataFrame()

    pathes = [path / 'a{:02d}'.format(act+1) / f'p{sub}' for act in ACTIVITIES.keys() for sub in SUBJECTS]
    pathes = itertools.chain(*list(map(lambda dir_path: dir_path.glob('s*.txt'), pathes)))
    pathes = list(filter(lambda path: not path.is_dir(), pathes))
    # pathes = list(filter(lambda p: p.exists(), pathes))   # このケースが存在した場合エラーを吐きたい
    chunks = [_load_raw_data(path) for path in pathes]
    return chunks

   
def reformat(raw) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """Function for reformating

    Parameters
    ----------
    raw:
        data loaded by 'load_raw'
    
    Returns
    -------
    data, meta: List[pd.DataFrame], List[pd.DataFrame]
        Sensor data segmented by activity and subject

    See Alos
    --------
    The order of 'data' and 'meta' correspond.

    e.g. meta[0] is meta data of data[0].
    """

    cols_meta = ['activity', 'subject',]
    cols_sensor = [c for c in COLUMNS if c not in cols_meta]
    data = list(map(lambda seg: seg[cols_sensor], raw))
    meta = list(map(lambda seg: seg[cols_meta], raw))
    
    return data, meta
