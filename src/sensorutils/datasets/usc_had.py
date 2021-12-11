"""USC-HAD

https://sipi.usc.edu/had/
"""

import itertools
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .base import BaseDataset, check_path
from ..core import split_using_sliding_window


__all__ = ['load', 'load_raw']

# Meta Info
Sampling_Rate = 100 # Hz
COLUMNS = [
    'acc_x', 'acc_y', 'acc_z',
    'gyro_x', 'gyro_y', 'gyro_z',
    'version', 'trial',
    'activity', 'subject',
    'age', 'height', 'weight',
    'sensor_location', 'sensor_orientation',
]

SUBJECTS = list(range(1, 14+1))    # range of subject id: [1, 14]

ACTIVITIES_NAMES = (
    'Walking Forward',
    'Walking Left', 
    'Walking Right',
    'Walking Upstairs',
    'Walking Downstairs',
    'Running Forward',
    'Jumping Up',
    'Sitting',
    'Standing',
    'Sleeping',
    'Elevator Up',
    'Elevator Down',
)
ACTIVITIES = dict(
    zip(range(12), ACTIVITIES_NAMES)
)


class USC_HAD(BaseDataset):
    """
    USC-HADデータセットに記録されているセンサデータとメタデータを読み込む．

    Parameters
    ----------
    path: Path
        USC-HADデータセットのパス(path/to/dataset/").
    
    Attributes
    ----------
    supported_x_labels: List[str]
        ターゲット以外のすべてのラベルのリスト

    supported_y_labels: List[str]
        ターゲットラベルのリスト
    """

    supported_x_labels = ('acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z')
    supported_y_labels = ('version', 'trial', 'activity', 'subject', 'age', 'height', 'weight', 'sensor_location', 'sensor_orientation')

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
        window_size:int, stride:int,
        x_labels:Optional[list]=None, y_labels:Optional[list]=None,
        ftrim_sec:int=5, btrim_sec:int=5, 
        subjects:Optional[list]=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        USC-HADデータセットを読み込み，sliding-window処理を行ったデータを返す．

        Parameters
        ----------
        window_size: int
            フレーム分けするサンプルサイズ

        stride: int
            ウィンドウの移動幅

        x_labels: Optional[list]
            入力(従属変数)のラベルリスト(ラベル名は元データセットに準拠)．ここで指定したラベルのデータが入力として取り出される．

        y_labels: Optional[list]
            ターゲットのラベルリスト(仕様はx_labelsと同様)

        ftrim_sec: int
            セグメント先頭のトリミングサイズ(単位は秒)

        btrim_sec: int
            セグメント末尾のトリミングサイズ(単位は秒)

        subjects:
            ロードする被験者を指定する．指定されない場合はすべての被験者のデータを返す．
            被験者は計9名おり，それぞれにIDが割り当てられている．
            
            被験者ID: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        
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

        'Walking Forward',
        'Walking Left', 
        'Walking Right',
        'Walking Upstairs',
        'Walking Downstairs',
        'Running Forward',
        'Jumping Up',
        'Sitting',
        'Standing',
        'Sleeping',
        'Elevator Up',
        'Elevator Down',

        Examples
        --------
        >>> usc_had_path = Path('path/to/dataset/')
        >>> usc_had = USC_HAD(ucs_had_path)
        >>>
        >>> x_labels = ['acc_x', 'acc_y', 'acc_z']
        >>> y_labels = ['activity', 'subject']
        >>> x, y = usc_had.load(x_labels=xlabels, y_labels=y_labels)
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

        frames = []
        for seg in segments:
            fs = split_using_sliding_window(
                np.array(seg), window_size=window_size, stride=stride,
                ftrim=Sampling_Rate*ftrim_sec, btrim=Sampling_Rate*btrim_sec,
                return_error_value=None)
            if fs is not None:
                frames += [fs]
            else:
                # print('no frame')
                pass
        frames = np.concatenate(frames)

        assert frames.shape[-1] == len(x_labels) + len(y_labels) + 1, 'Extracted data shape does not match with the number of total labels'
        # x_labelsでサポートされているラベルはすべてfloat64で対応
        # y_labelsでサポートされているラベルはすべてint8で対応可能
        x_frames = np.float64(frames[:, :, :len(x_labels)]).transpose(0, 2, 1)
        y_frames = frames[:, 0, len(x_labels):]

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
    """Function for loading USC-HAD dataset

    Parameters
    ----------
    path: Union[Path, str]
        Directory path of USC-HAD dataset.

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
    """Function for loading raw data of USC-HAD dataset

    Parameters
    ----------
    path: Path
        Directory path of USC-HAD dataset.

    Returns
    -------
    chunks_per_persons: List[pd.DataFrame]
        Raw data of USC-HAD dataset.

        Each item in 'chunks' is a part of dataset, which is splited by activity and subject.
    """

    def _load_raw_data(path):
        try:
            dat = loadmat(str(path))
            seg = pd.DataFrame(dat['sensor_readings'], columns=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])

            if 'activity_number' in dat:
                activity = int(dat['activity_number'][0])
            elif 'activity_numbr' in dat:
                activity = int(dat['activity_numbr'][0])
            else:
                raise RuntimeError('Detected invalid format (key of activty_number is missing)')
            if activity not in list(range(1, 12+1)):
                raise RuntimeError('Detected unknown file (invalid activity id: {}), {}'.format(activity, path))

            seg['version'] = int(float(dat['version'][0]))
            seg['trial'] = int(dat['trial'][0])
            seg['activity'] = activity-1   # [1, 12] -> [0, 11]
            seg['subject'] = int(dat['subject'][0])
            seg['age'] = int(dat['age'][0])
            seg['height'] = float(dat['height'][0].replace('cm', ''))
            seg['weight'] = float(dat['weight'][0].replace('kg', ''))
            seg['sensor_location'] = dat['sensor_location'][0]
            seg['sensor_orientation'] = dat['sensor_orientation'][0]
            
            dtypes = dict(zip(['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'], list(np.float64 for _ in COLUMNS)))
            dtypes['version'] = np.int8
            dtypes['trial'] = np.int8
            dtypes['activity'] = np.int8
            dtypes['subject'] = np.int8
            dtypes['age'] = np.int8
            dtypes['height'] = np.float64
            dtypes['weight'] = np.float64
            seg = seg.astype(dtypes)
            return seg

        except FileNotFoundError:
            print(f'[load] {path} not found')
        except pd.errors.EmptyDataError:
            print(f'[load] {path} is empty')
        return pd.DataFrame()

    pathes = [path / 'Subject{}'.format(sub) for sub in SUBJECTS]
    pathes = itertools.chain(*list(map(lambda dir_path: dir_path.glob('*.mat'), pathes)))
    pathes = list(filter(lambda path: not path.is_dir(), pathes))
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

    cols_meta = [
        'version', 'trial',
        'activity', 'subject',
        'age', 'height', 'weight',
        'sensor_location', 'sensor_orientation',
    ]
    cols_sensor = [c for c in COLUMNS if c not in cols_meta]
    data = list(map(lambda seg: seg[cols_sensor], raw))
    meta = list(map(lambda seg: seg[cols_meta], raw))
    
    return data, meta
