""" mHealth dataset

URL of dataset: http://archive.ics.uci.edu/ml/datasets/mhealth+dataset
"""

import numpy as np
import pandas as pd

import itertools
from pathlib import Path
from typing import List, Tuple, Optional, Union

from ..core import split_using_target, split_using_sliding_window

from .base import BaseDataset, check_path


__all__ = ['MHEALTH', 'load', 'load_raw']


# Meta Info
Sampling_Rate = 50 # Hz

SUBJECTS = list(range(1, 10+1))

ACTIVITIES = [
    'Null',     # 0 is a null class
    'Standing still', 'Sitting and relaxing', 'Lying down', 
    'Walking', 'Climbing stairs', 'Waist bends forward', 
    'Frontal elevation of arms', 'Knees bending (crouching)', 
    'Cycling', 'Jogging', 'Running', 'Jump front & back',
]

COLUMNS = [
    'acceleration_chest_x',
    'acceleration_chest_y',
    'acceleration_chest_z',
    'electrocardiogram_1',
    'electrocardiogram_2',
    'acceleration_left-ankle_x',
    'acceleration_left-ankle_y',
    'acceleration_left-ankle_z',
    'gyro_left-ankle_x',
    'gyro_left-ankle_y',
    'gyro_left-ankle_z',
    'magnetometer_left-ankle_x',
    'magnetometer_left-ankle_y',
    'magnetometer_left-ankle_z',
    'acceleration_right-lower-arm_x',
    'acceleration_right-lower-arm_y',
    'acceleration_right-lower-arm_z',
    'gyro_right-lower-arm_x',
    'gyro_right-lower-arm_y',
    'gyro_right-lower-arm_z',
    'magnetometer_right-lower-arm_x',
    'magnetometer_right-lower-arm_y',
    'magnetometer_right-lower-arm_z',
    'activity',
    # 'subject' # ローダで付与
]


class MHEALTH(BaseDataset):
    """
    MHEALTHデータセットに記録されているセンサデータとメタデータを読み込む．

    Parameters
    ----------
    path: Path
        mHealthデータセットのパス(path/to/dataset/).

    Attributes
    ----------
    supported_x_labels: List[str]
        ターゲット以外のすべてのラベルのリスト

    supported_y_labels: List[str]
        ターゲットラベルのリスト
    """

    supported_x_labels = tuple(set(COLUMNS) - set(['activity', 'subject']))
    supported_y_labels = ('activity', 'subject')

    def __init__(self, path:Path, cache_dir:Path=Path('./')):
        super().__init__(path)
        self.cache_dir = cache_dir
        self.data_cache = None

    def _load_segments(self):
        data, meta = load(self.path)
        segments = [m.join(seg) for seg, m in zip(data, meta)]
        return segments
    
    def _filter_by_subject(self, x_frames, y_frames, subject_labels, subjects):
        s_flags = np.zeros(len(x_frames), dtype=bool)
        for subject in subjects:
            s_flags = np.logical_or(s_flags, subject_labels == subject)
        flags = s_flags

        # filter
        x_frames, y_frames = x_frames[flags], y_frames[flags]

        return x_frames, y_frames
    
    def _load(self):
        if self.data_cache is None:
            segments = self._load_segments()
            self.data_cache = segments
        else:
            segments = self.data_cache
        return segments
    
    def load(self, window_size:int=100, stride:int=100, x_labels:Optional[list]=None, y_labels:Optional[list]=None, ftrim_sec:int=0, btrim_sec:int=0, subjects:Optional[list]=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        mHealthデータセットを読み込み，sliding-window処理を行ったデータを返す．

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
            被験者は計10名おり，それぞれにIDが割り当てられている．
            
            被験者ID: [1, 2, 3, 4, 5, 6 ,7, 8, 9, 10]
        
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

        Examples
        --------
        >>> mhealth_path = Path('path/to/dataset/')
        >>> mhealth = MHEALTH(mhealth_path)
        >>>
        >>> x_labels = ['acceleration_chest_x', 'acceleration_chest_y', 'acceleration_chest_z']
        >>> y_labels = ['activity']
        >>> subjects = [1, 2, 3]
        >>> x, y = mhealth.load(window_size=128, stride=128, x_labels=xlabels, y_labels=y_labels, ftrim_sec=3, btrim_sec=3, subjects=subjects)
        >>> print(f'x: {x.shape}, y: {y.shape}')
        >>>
        >>> # > x: (?, 3, 128), y: (?, 1)
        """

        if not isinstance(subjects, list) and subjects is not None:
            raise TypeError('expected type of subjects is list or None, but {}'.format(type(subjects)))
        if subjects is not None:
            if not (set(subjects) <= set(SUBJECTS)):
                raise ValueError('detect unknown subject, {}'.format(subjects))

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
        segments = [seg[x_labels+y_labels+['activity', 'subject']] for seg in segments]
        frames = []
        for seg in segments:
            fs = split_using_sliding_window(
                np.array(seg), window_size=window_size, stride=stride,
                ftrim=Sampling_Rate*ftrim_sec, btrim=Sampling_Rate*btrim_sec,
                return_error_value=None)
            if fs is not None:
                frames += [fs]
            else:
                print('no frame: {}'.format(np.array(seg).shape))
        frames = np.concatenate(frames)
        assert frames.shape[-1] == len(x_labels) + len(y_labels) + 2, 'Extracted data shape does not match with the number of total labels'
        # x_labelsでサポートされているラベルはすべてfloat64で対応
        # y_labelsでサポートされているラベルはすべてint8で対応可能
        x_frames = np.float64(frames[:, :, :len(x_labels)]).transpose(0, 2, 1)
        y_frames = np.int8(frames[:, 0, len(x_labels):])

        # remove data which activity label is 0
        flgs = y_frames[:, -2] != 0
        x_frames = x_frames[flgs]
        y_frames = y_frames[flgs]

        # subject filtering
        if subjects is not None:
            subject_labels = y_frames[:, -1]
            x_frames, y_frames = self._filter_by_subject(x_frames, y_frames, subject_labels, subjects)
        
        # remove tail columns(activity, subject)
        y_frames = y_frames[:, :-2]

        return x_frames, y_frames


def load(path:Union[Path,str]) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """Function for loading mHealth dataset

    Parameters
    ----------
    path: Union[Path, str]
        Directory path of mHealth dataset.

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
    """Function for loading raw data of mHealth dataset

    Parameters
    ----------
    path: Path
        Directory path of mHealth dataset.

    Returns
    -------
    chunks_per_subjects: List[pd.DataFrame]
        Raw data of mHealth dataset.

        Each item in 'chunks_per_subjects ' is a part of dataset, which is splited by subject.
    """

    def _load_raw_data(path, subject_id):
        try:
            seg = pd.read_csv(str(path), sep='\s+', header=None)
            seg.columns = COLUMNS
            seg['subject'] = subject_id
            # 欠損値処理は暫定的
            seg = seg.fillna(method='ffill')
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

    pathes = [path / f'mHealth_subject{subject_id}.log' for subject_id in SUBJECTS]
    chunks_per_subjects = [
        _load_raw_data(path, subject_id) for subject_id, path in zip(SUBJECTS, pathes)
    ]
    return chunks_per_subjects

   
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

    chunks_per_subjects = raw
    segs = []
    for chunk in chunks_per_subjects:
        sub_segs = split_using_target(np.array(chunk), np.array(chunk['activity']))
        sub_segs = list(itertools.chain(*[sub_segs[k] for k in sub_segs.keys()]))  # 連結
        sub_segs = list(map(lambda x: pd.DataFrame(x, columns=chunk.columns).astype(chunk.dtypes.to_dict()), sub_segs))

        # For debug
        for seg in sub_segs:
            label = seg['activity'].iloc[0]
            if not np.array(seg['activity'] == label).all():
                raise RuntimeError('This is bug. Failed segmentation')
        segs += sub_segs

    cols_meta = ['activity', 'subject',]
    cols_sensor = [c for c in COLUMNS if c not in cols_meta]
    data = list(map(lambda seg: seg[cols_sensor], segs))
    meta = list(map(lambda seg: seg[cols_meta], segs))
    
    return data, meta

