"""Daily and Sports Activity Data Set

https://archive.ics.uci.edu/ml/datasets/daily+and+sports+activities
"""

import itertools
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base import BaseDataset, check_path


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


class DSADS(BaseDataset):
    """
    DSADSデータセットに記録されているセンサデータとメタデータを読み込む．

    Parameters
    ----------
    path: Path
        DSADSデータセットのパス(path/to/dataset/data").
    
    Attributes
    ----------
    supported_x_labels: List[str]
        ターゲット以外のすべてのラベルのリスト

    supported_y_labels: List[str]
        ターゲットラベルのリスト
    """

    supported_x_labels = tuple(set(COLUMNS) - set(['activity', 'subject']))
    supported_y_labels = ('activity', 'subject')

    def __init__(self, path:Path):
        super().__init__(path)
        self.data_cache = None
    
    def _load(self):
        if self.data_cache is None:
            segments = load_raw(self.path)
            self.data_cache = segments
        else:
            segments = self.data_cache
        return segments
    
    def load(self,
        x_labels:Optional[list]=None, y_labels:Optional[list]=None,
        subjects:Optional[list]=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        DSASSデータセットを読み込み，sliding-window処理を行ったデータを返す．

        Parameters
        ----------
        x_labels: Optional[list]
            入力(従属変数)のラベルリスト(ラベル名は元データセットに準拠)．ここで指定したラベルのデータが入力として取り出される．

        y_labels: Optional[list]
            ターゲットのラベルリスト(仕様はx_labelsと同様)

        subjects:
            ロードする被験者を指定する．指定されない場合はすべての被験者のデータを返す．
            被験者は計9名おり，それぞれにIDが割り当てられている．
            
            被験者ID: [1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        Returns
        -------
        (x_frames, y_frames): Tuple[np.ndarray, np.ndarray]
            sliding-windowで切り出した入力とターゲットのフレームリスト

            x_framesは3次元配列で構造は大まかに(Batch, Channels, Frame)のようになっている．
            Channelsはx_labelsで指定したものが格納される．

            y_framesは2次元配列で構造は大まかに(Batch, Labels)のようになっている．
            Labelsはy_labelsで指定したものが格納される．

            y_framesはデータセット内の値をそのまま返すため，分類で用いる際はラベルの再割り当てが必要となることに注意する．
        
        See Also
        --------
        一度loadメソッドを読みだすと内部にキャッシュを作成するため2回目以降のloadメソッドの読みだしは比較的速くなる．

        x_labelsやy_labelsに指定できるラベルは次の通りである．

        'T_xacc', 'T_yacc', 'T_zacc',
        'T_xgyro', 'T_ygyro', 'T_zgyro',
        'T_xmag', 'T_ymag', 'T_zmag',
        'RA_xacc', 'RA_yacc', 'RA_zacc',
        'RA_xgyro', 'RA_ygyro', 'RA_zgyro',
        'RA_xmag', 'RA_ymag', 'RA_zmag',
        'LA_xacc', 'LA_yacc', 'LA_zacc',
        'LA_xgyro', 'LA_ygyro', 'LA_zgyro',
        'LA_xmag', 'LA_ymag', 'LA_zmag',
        'RL_xacc', 'RL_yacc', 'RL_zacc',
        'RL_xgyro', 'RL_ygyro', 'RL_zgyro',
        'RL_xmag', 'RL_ymag', 'RL_zmag',
        'LL_xacc', 'LL_yacc', 'LL_zacc',
        'LL_xgyro', 'LL_ygyro', 'LL_zgyro',
        'LL_xmag', 'LL_ymag', 'LL_zmag'
        'activity', 'subject'

        Examples
        --------
        >>> dsads_path = Path('path/to/dataset/data/')
        >>> dsads = DSASS(dsads_path)
        >>>
        >>> x_labels = ['T_accx', 'T_accy', 'T_accz']
        >>> y_labels = ['activity', 'subject']
        >>> x, y = dsads.load(x_labels=xlabels, y_labels=y_labels)
        >>> print(f'x: {x.shape}, y: {y.shape}')
        >>>
        >>> # > x: (?, 3, 125), y: (?, 1)
        """

        global SUBJECTS
        if subjects is not None:
            if not (set(subjects) <= set(SUBJECTS)):
                raise ValueError('detect unknown person, {}'.format(subjects))
        
        if x_labels is None:
            x_labels = list(self.supported_x_labels)
        if y_labels is None:
            y_labels = list(self.supported_y_labels)
        if not(set(x_labels) <= set(self.supported_x_labels)):
            raise ValueError('unsupported x labels is included: {}'.format(
                tuple(set(x_labels) - set(self.supported_x_labels).intersection(set(x_labels)))
            ))
        if not(set(y_labels) <= set(self.supported_y_labels)):
            raise ValueError('unsupported y labels is included: {}'.format(
                tuple(set(y_labels) - set(self.supported_y_labels).intersection(set(y_labels)))
            ))

        segments = self._load()
        segments = [seg[x_labels+y_labels+['subject']].to_numpy() for seg in segments]
        frames = np.stack(segments)
        assert frames.shape[-1] == len(x_labels) + len(y_labels) + 1, 'Extracted data shape does not match with the number of total labels'
        # x_labelsでサポートされているラベルはすべてfloat64で対応
        # y_labelsでサポートされているラベルはすべてint8で対応可能
        x_frames = np.float64(frames[:, :, :len(x_labels)]).transpose(0, 2, 1)
        y_frames = np.int8(frames[:, 0, len(x_labels):])

        # subject filtering
        if subjects is not None:
            subject_labels = y_frames[:, -1]
            x_frames, y_frames = self._filter_by_subject(x_frames, y_frames, subject_labels, subjects)
        
        # remove tail columns(subject)
        y_frames = y_frames[:, :-1]

        return x_frames, y_frames

    def _filter_by_subject(self, x_frames, y_frames, subject_labels, subjects):
        p_flags = np.zeros(len(x_frames), dtype=bool)
        for s in subjects:
            p_flags = np.logical_or(p_flags, subject_labels == s)
        flags = p_flags

        # filter
        x_frames, y_frames = x_frames[flags], y_frames[flags]

        return x_frames, y_frames


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
