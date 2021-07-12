"""PAMAP2 Dataset

URL of dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip
"""

import numpy as np
import pandas as pd

import itertools
from pathlib import Path
from typing import List, Tuple, Optional, Union
from ..core import split_using_target, split_using_sliding_window

from .base import BaseDataset, check_path


__all__ = ['PAMAP2', 'load', 'load_raw']


# Meta Info
Sampling_Rate = 100 # Hz
ATTRIBUTES = ['temperature', 'acc1', 'acc2', 'gyro', 'mag']
POSITIONS = ['hand', 'chest', 'ankle']
AXES = ['x', 'y', 'z']

PERSONS = [
    'subject101', 'subject102', 'subject103',
    'subject104', 'subject105', 'subject106',
    'subject107', 'subject108', 'subject109',
]

ACTIVITIES = {
    1: 'lying', 2: 'sitting', 3: 'standing', 4: 'walking', 5: 'running',
    6: 'cycling', 7: 'nordic_walking', 9: 'watching_TV', 10: 'computer_work',
    11: 'car_driving', 12: 'ascending_stairs', 13: 'descending_stairs',
    16: 'vacuum_cleaning', 17: 'ironing', 18: 'folding_laundry',
    19: 'house_cleaning', 20: 'playing_soccer',
    24: 'rope_jumping',
    0: 'other',
}

# Columns = ['timestamp(s)', 'activity_id', 'heart_rate(bpm)']
# for pos in POSITIONS:
#     Columns += ['IMU_{}_{}'.format(pos, ATTRIBUTES[0])]
#     for attr in ATTRIBUTES[1:]:
#         for axis in AXES:
#             col = 'IMU_{}_{}_{}'.format(pos, attr, axis)
#             Columns += [col]
#     Columns += ['IMU_{}_orientation{}'.format(pos, i) for i in range(4)]
Columns = [
    'timestamp(s)',
    'activity_id',
    # 'person_id',  # ローダ側で付与
    'heart_rate(bpm)',
    'IMU_hand_temperature',
    'IMU_hand_acc1_x',
    'IMU_hand_acc1_y',
    'IMU_hand_acc1_z',
    'IMU_hand_acc2_x',
    'IMU_hand_acc2_y',
    'IMU_hand_acc2_z',
    'IMU_hand_gyro_x',
    'IMU_hand_gyro_y',
    'IMU_hand_gyro_z',
    'IMU_hand_mag_x',
    'IMU_hand_mag_y',
    'IMU_hand_mag_z',
    'IMU_hand_orientation0',
    'IMU_hand_orientation1',
    'IMU_hand_orientation2',
    'IMU_hand_orientation3',
    'IMU_chest_temperature',
    'IMU_chest_acc1_x',
    'IMU_chest_acc1_y',
    'IMU_chest_acc1_z',
    'IMU_chest_acc2_x',
    'IMU_chest_acc2_y',
    'IMU_chest_acc2_z',
    'IMU_chest_gyro_x',
    'IMU_chest_gyro_y',
    'IMU_chest_gyro_z',
    'IMU_chest_mag_x',
    'IMU_chest_mag_y',
    'IMU_chest_mag_z',
    'IMU_chest_orientation0',
    'IMU_chest_orientation1',
    'IMU_chest_orientation2',
    'IMU_chest_orientation3',
    'IMU_ankle_temperature',
    'IMU_ankle_acc1_x',
    'IMU_ankle_acc1_y',
    'IMU_ankle_acc1_z',
    'IMU_ankle_acc2_x',
    'IMU_ankle_acc2_y',
    'IMU_ankle_acc2_z',
    'IMU_ankle_gyro_x',
    'IMU_ankle_gyro_y',
    'IMU_ankle_gyro_z',
    'IMU_ankle_mag_x',
    'IMU_ankle_mag_y',
    'IMU_ankle_mag_z',
    'IMU_ankle_orientation0',
    'IMU_ankle_orientation1',
    'IMU_ankle_orientation2',
    'IMU_ankle_orientation3',
]


class PAMAP2(BaseDataset):
    """
    PAMAP2データセットに記録されているセンサデータとメタデータを読み込む．

    Parameters
    ----------
    path: Path
        PAMAP2データセットのパス(path/to/dataset/PAMAP2_Dataset").

    Attributes
    ----------
    supported_x_labels: List[str]
        ターゲット以外のすべてのラベルのリスト

    supported_y_labels: List[str]
        ターゲットラベルのリスト
    """

    supported_x_labels = tuple(set(Columns) - set(['activity_id', 'person_id']))
    supported_y_labels = ('activity_id', 'person_id')

    def __init__(self, path:Path, cache_dir:Path=Path('./')):
        super().__init__(path)
        self.cache_dir = cache_dir
        self.data_cache = None

    def _load_segments(self):
        data, meta = load(self.path)
        segments = [m.join(seg) for seg, m in zip(data, meta)]
        self.min_max_vals = pd.concat(segments, axis=0).agg(['min', 'max'])
        return segments
    
    def _normalize_segment(self, segment):
        for col in segment.columns:
            if 'IMU' in col and 'temperature' not in col and 'orientation' not in col:
                min_val, max_val = self.min_max_vals[col].loc['min'], self.min_max_vals[col].loc['max']
                segment[col] = (segment[col] - min_val) / (max_val - min_val)
        return segment
    
    def _filter_by_person(self, x_frames, y_frames, person_labels, persons):
        p2id = dict(zip(PERSONS, list(range(len(PERSONS)))))
        p_flags = np.zeros(len(x_frames), dtype=bool)
        for person in persons:
            p_flags = np.logical_or(p_flags, person_labels == p2id[person])
        flags = p_flags

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
    
    def load(self, window_size:int, stride:int, x_labels:Optional[list]=None, y_labels:Optional[list]=None, ftrim_sec:int=10, btrim_sec:int=10, persons:Optional[list]=None, norm:bool=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        PAMAP2データセットを読み込み，sliding-window処理を行ったデータを返す．

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

        persons:
            ロードする被験者を指定する．指定されない場合はすべての被験者のデータを返す．
            被験者は計9名おり，それぞれに文字列のIDが割り当てられている．
            
            被験者ID: ['subject101', 'subject102', 'subject103', 'subject104', 'subject105', 'subject106', 'subject107', 'subject108', 'subject109']
        
        norm: bool
            (beta) センサデータの標準化を行うかどうかのフラグ．

            この機能はあまり確認ができていないので，使用する際は注意を払うこと．

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
        >>> pamap2_path = Path('path/to/dataset/PAMAP2_Dataset/')
        >>> pamap2 = PAMAP2(pamap2_path)
        >>>
        >>> x_labels = ['IMU_chest_acc1_x', 'IMU_chest_acc1_y', 'IMU_chest_acc1_z']
        >>> y_labels = ['activity_id']
        >>> subjects = ['subject101', 'subject102', 'subject103']
        >>> x, y = pamap2.load(window_size=256, stride=128, x_labels=xlabels, y_labels=y_labels, ftrim_sec=2, btrim_sec=2, persons=subjects)
        >>> print(f'x: {x.shape}, y: {y.shape}')
        >>>
        >>> # > x: (?, 3, 256), y: (?, 1)
        """

        if not isinstance(persons, list) and persons is not None:
            raise TypeError('expected type of persons is list or None, but {}'.format(type(persons)))
        if persons is not None:
            if not (set(persons) <= set(PERSONS)):
                raise ValueError('detect unknown person, {}'.format(persons))

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

        # if not set(self.not_supported_labels).isdisjoint(set(x_labels+y_labels)):
        #     raise ValueError('x_labels or y_labels include non supported labels')

        # segments = self._load_segments()
        segments = self._load()
        segments = [seg[x_labels+y_labels+['activity_id', 'person_id']] for seg in segments]
        if norm:
            segments = [self._normalize_segment(seg) for seg in segments]
        frames = []
        for seg in segments:
            fs = split_using_sliding_window(
                np.array(seg), window_size=window_size, stride=stride,
                ftrim=Sampling_Rate*ftrim_sec, btrim=Sampling_Rate*btrim_sec,
                return_error_value=None)
            if fs is not None:
                frames += [fs]
            else:
                print('no frame')
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
        if persons is not None:
            person_labels = y_frames[:, -1]
            x_frames, y_frames = self._filter_by_person(x_frames, y_frames, person_labels, persons)
        
        # remove tail columns(activity_id, person_id)
        y_frames = y_frames[:, :-2]

        return x_frames, y_frames


def load(path:Union[Path,str]) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """Function for loading PAMAP2 dataset

    Parameters
    ----------
    path: Union[Path, str]
        Directory path of PAMAP2 dataset.

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
    """Function for loading raw data of PAMAP2 dataset

    Parameters
    ----------
    path: Path
        Directory path of PAMAP2 dataset("PAMAP2_Dataset").

    Returns
    -------
    chunks_per_persons: List[pd.DataFrame]
        Raw data of PAMAP2 dataset.

        Each item in 'chunks_per_persons' is a part of dataset, which is splited by subject.
    """

    def _load_raw_data(path, person_id):
        try:
            seg = pd.read_csv(str(path), sep='\s+', header=None)
            seg.columns = Columns
            seg['person_id'] = person_id
            # 欠損値処理は暫定的
            seg = seg.fillna(method='ffill')
            dtypes = dict(zip(Columns, list(np.float64 for _ in Columns)))
            dtypes['activity_id'] = np.int8
            dtypes['person_id'] = np.int8
            seg = seg.astype(dtypes)
            return seg
        except FileNotFoundError:
            print(f'[load] {path} not found')
        except pd.errors.EmptyDataError:
            print(f'[load] {path} is empty')
        return pd.DataFrame()

    pathes = [path / 'Protocol' / (person + '.dat') for person in PERSONS]
    # pathes = list(filter(lambda p: p.exists(), pathes))   # このケースが存在した場合エラーを吐きたい
    chunks_per_persons = [_load_raw_data(path, p_id) for p_id, path in enumerate(pathes)]
    return chunks_per_persons

   
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

    chunks_per_persons = raw
    segs = []
    for p_id, chunk in enumerate(chunks_per_persons):
        # chunk['person_id'] = p_id
        sub_segs = split_using_target(np.array(chunk), np.array(chunk['activity_id']))
        sub_segs = list(itertools.chain(*[sub_segs[k] for k in sub_segs.keys()]))  # 連結
        sub_segs = list(map(lambda x: pd.DataFrame(x, columns=chunk.columns).astype(chunk.dtypes.to_dict()), sub_segs))

        # For debug
        for seg in sub_segs:
            label = seg['activity_id'].iloc[0]
            if not np.array(seg['activity_id'] == label).all():
                raise RuntimeError('This is bug. Failed segmentation')
        segs += sub_segs

    cols_meta = ['activity_id', 'person_id',]
    cols_sensor = list(set(Columns) - set(cols_meta))
    data = list(map(lambda seg: seg[cols_sensor], segs))
    meta = list(map(lambda seg: seg[cols_meta], segs))
    
    return data, meta

