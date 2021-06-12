"""WISDM dataset

URL of dataset: https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Tuple
from ..core import split_using_target, split_using_sliding_window

from .base import BaseDataset


__all__ = ['WISDM', 'load', 'load_raw']


# Meta Info
SUBJECTS = tuple(range(1, 36+1))
ACTIVITIES = tuple(['Walking', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Downstairs'])
Sampling_Rate = 20 # Hz


class WISDM(BaseDataset):
    def __init__(self, path:Path):
        super().__init__(path)
    
    def load(self, window_size:int=None, stride:int=None, ftrim_sec:int=3, btrim_sec:int=3, subjects:Union[list, None]=None):
        """WISDMの読み込みとsliding-window

        Parameters
        ----------
        window_size: int
            フレーム分けするサンプルサイズ

        stride: int
            ウィンドウの移動幅

        ftrim_sec: int
            セグメント先頭のトリミングサイズ(単位は秒)

        btrim_sec: int
            セグメント末尾のトリミングサイズ(単位は秒)
        
        subjects: list
            ロードする被験者を指定

        Returns
        -------
        (x_frames, y_frames): tuple
            sliding-windowで切り出した入力とターゲットのフレームリスト
        """

        segments, meta = load(path=self.path)
        segments = [m.join(seg) for seg, m in zip(segments, meta)]

        x_frames, y_frames = [], []
        for seg in segments:
            fs = split_using_sliding_window(
                np.array(seg), window_size=window_size, stride=stride,
                ftrim=Sampling_Rate*ftrim_sec, btrim=Sampling_Rate*btrim_sec,
                return_error_value=None)
            if fs is not None:
                x_frames += [fs[:, :, 3:]]
                y_frames += [np.uint8(fs[:, 0, 0:2][..., ::-1])] # 多分これでact, subjectの順に変わる
            else:
                # print('no frame')
                pass
        x_frames = np.concatenate(x_frames).transpose([0, 2, 1])
        y_frames = np.concatenate(y_frames)

        # subject filtering
        if subjects is not None:
            flags = np.zeros(len(x_frames), dtype=bool)
            for sub in subjects:
                flags = np.logical_or(flags, y_frames[:, 1] == sub)
                # flags = np.logical_or(flags, y_frames[:, 0] == sub)
            x_frames = x_frames[flags]
            y_frames = y_frames[flags]

        return x_frames, y_frames


def load(path:Path) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """Function for loading WISDM dataset

    Parameters
    ----------
    path: Path
        Directory path of WISDM dataset('data' directory)

    Returns
    -------
    data, meta: List[pd.DataFrame], List[pd.DataFrame]
        Sensor data segmented by activity and subject

    See Alos
    --------
    The order of 'data' and 'meta' correspond.

    e.g. meta[0] is meta data of data[0].
    """
    raw = load_raw(path)
    data, meta = reformat(raw)
    return data, meta


def load_raw(dataset_path:Path) -> pd.DataFrame:
    """Function for loading raw data of WISDM dataset

    Parameters
    ----------
    path: Path
        Directory path of WISDM dataset('data' directory)

    Returns
    -------
    raw_data : pd.DataFrame
        raw data of WISDM dataset

    See Also
    --------
    Structure of one segment:
        np.ndarray([
            [user id, activity id, timestamp, x-acceleration, y-acceleration, z-acceleration],
            [user id, activity id, timestamp, x-acceleration, y-acceleration, z-acceleration],
            ...,
            [user id, activity id, timestamp, x-acceleration, y-acceleration, z-acceleration],
        ], dtype=float64))
    
    Range of activity label: [0, 5]
    Range of subject label : [1, 36]
    """

    dataset_path = dataset_path / 'WISDM_ar_v1.1_raw.txt'
    with dataset_path.open('r') as fp:
        whole_str = fp.read()
    
    # データセットのmiss formatを考慮しつつ簡易パースを行う
    # [基本構造]
    # [user],[activity],[timestamp],[x-acceleration],[y-accel],[z-accel];
    # [miss format]
    # - ";"の前にコロンが入ってしまっている
    # - ";"が抜けている
    # - z-accelerationが抜けている(おそらく一か所だけ)
    whole_str = whole_str.replace(',;', ';')
    semi_separated = re.split('[;\n]', whole_str)
    semi_separated = list(filter(lambda x: x != '', semi_separated))
    comma_separated = [r.strip().split(',') for r in semi_separated]

    # debug
    for s in comma_separated:
        if len(s) != 6:
            print('[miss format?]: {}'.format(s))

    raw_data = pd.DataFrame(comma_separated)
    raw_data.columns = ['user', 'activity', 'timestamp', 'x-acceleration', 'y-acceleration', 'z-acceleration']
    # z-accelerationには値が''となっている行が一か所だけ存在する
    # このままだと型キャストする際にエラーが発生するためnanに置き換えておく
    raw_data['z-acceleration'] = raw_data['z-acceleration'].replace('', np.nan)

    # convert activity name to activity id
    raw_data = raw_data.replace(list(ACTIVITIES), list(range(len(ACTIVITIES))))

    raw_data = raw_data.astype({'user': 'uint8', 'activity': 'uint8', 'timestamp': 'uint64', 'x-acceleration': 'float64', 'y-acceleration': 'float64', 'z-acceleration': 'float64'})
    raw_data[['x-acceleration', 'y-acceleration', 'z-acceleration']] = raw_data[['x-acceleration', 'y-acceleration', 'z-acceleration']].fillna(method='ffill')

    return raw_data


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

    raw_array = raw.to_numpy()
    
    # segmentへの分割(by user and activity)
    sdata_splited_by_subjects = split_using_target(src=raw_array, target=raw_array[:, 0])
    segments = []
    for sub_id in sdata_splited_by_subjects.keys():
        for src in sdata_splited_by_subjects[sub_id]:
            splited = split_using_target(src=src, target=src[:, 1])
            for act_id in splited.keys():
                segments += splited[act_id]

    segments = list(map(lambda seg: pd.DataFrame(seg, columns=raw.columns).astype(raw.dtypes.to_dict()), segments))
    data = list(map(lambda seg: pd.DataFrame(seg.iloc[:, 3:], columns=raw.columns[3:]), segments))
    meta = list(map(lambda seg: pd.DataFrame(seg.iloc[:, :3], columns=raw.columns[:3]), segments))

    return data, meta 

