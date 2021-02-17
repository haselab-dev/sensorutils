import numpy as np
import pandas as pd

import pickle
from pathlib import Path
from typing import Union, Optional
from ..core import split_using_target, split_using_sliding_window


class PAMAP2:
    def __init__(self, path:Path, cache_dir:Path=Path('./')):
        self.path = path
        self.cache_dir = cache_dir
        self.data_cache = None
    
    def _load_segments(self):
        segments = load(self.path)
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
        p_flags = np.zeros(len(x_frames), dtype=np.bool)
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
    
    def load(self, window_size:int, stride:int, x_labels:list, y_labels:list, ftrim_sec:int, btrim_sec:int, persons:Union[list, None]=None, norm:bool=False):
        """PAMAP2の読み込みとsliding-window

        Parameters
        ----------
        window_size: int
            フレーム分けするサンプルサイズ

        stride: int
            ウィンドウの移動幅

        x_labels: list
            入力(従属変数)のラベルリスト(ラベル名は元データセットに準拠)．ここで指定したラベルのデータが入力として取り出される．
            一部サポートしていないラベルがあるの注意

        y_labels: list
            ターゲットのラベルリスト(仕様はx_labelsと同様)

        ftrim_sec: int
            セグメント先頭のトリミングサイズ(単位は秒)

        btrim_sec: int
            セグメント末尾のトリミングサイズ(単位は秒)

        Returns
        -------
        (x_frames, y_frames): tuple
            sliding-windowで切り出した入力とターゲットのフレームリスト
            y_framesはデータセット内の値をそのまま返すため，分類で用いる際はラベルの再割り当てが必要となることに注意
        """

        if not isinstance(persons, list) and persons is not None:
            raise TypeError('expected type of persons is list or None, but {}'.format(type(persons)))
        if persons is not None:
            if not (set(persons) <= set(PERSONS)):
                raise ValueError('detect unknown person, {}'.format(persons))
        # if not set(self.not_supported_labels).isdisjoint(set(x_labels+y_labels)):
        #     raise ValueError('x_labels or y_labels include non supported labels')
        # segments = self._load_segments()
        segments = self._load()
        segments = [seg[x_labels+y_labels+['person_id']] for seg in segments]
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
        assert frames.shape[-1] == len(x_labels) + len(y_labels) + 1, 'Extracted data shape does not match with the number of total labels'
        x_frames = frames[..., :len(x_labels)]
        y_frames = frames[..., 0, len(x_labels):-1]

        if persons is not None:
            person_labels = frames[..., 0, -1]
            x_frames, y_frames = self._filter_by_person(x_frames, y_frames, person_labels, persons)

        return x_frames, y_frames
    
def load(path:Path) -> dict:
    """PAMAP2の読み込み

    Parameters
    ----------
    path: Path
        PAMAP2データセットのディレクトリ(PAMAP2_Datasetディレクトリ)

    Returns
    -------
    segments: list
        行動ラベルをもとにセグメンテーションされたデータ
    """

    import itertools

    def _load_raw_data(path):
        try:
            seg = pd.read_csv(str(path), sep='\s+', header=None)
            seg.columns = Columns
            # 欠損値処理は暫定的
            seg = seg.fillna(method='ffill')
            return seg
        except FileNotFoundError:
            print(f'[load] {path} not found')
        except pd.errors.EmptyDataError:
            print(f'[load] {path} is empty')
        return pd.DataFrame()

    pathes = [path / 'Protocol' / (person + '.dat') for person in PERSONS]
    # pathes = list(filter(lambda p: p.exists(), pathes))   # このケースが存在した場合エラーを吐きたい
    chunks_per_persons = [_load_raw_data(path) for path in pathes]
    segs = []
    for p_id, chunk in enumerate(chunks_per_persons):
        chunk['person_id'] = p_id
        sub_segs = split_using_target(np.array(chunk), np.array(chunk['activity_id']))
        sub_segs = list(itertools.chain(*[sub_segs[k] for k in sub_segs.keys()]))  # 連結
        sub_segs = list(map(lambda x: pd.DataFrame(x, columns=chunk.columns), sub_segs))
        # For debug
        for seg in sub_segs:
            label = seg['activity_id'].iloc[0]
            if not np.array(seg['activity_id'] == label).all():
                raise RuntimeError('This is bug. Failed segmentation')
        segs += sub_segs

    return segs

   
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
