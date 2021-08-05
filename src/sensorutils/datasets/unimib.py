"""UniMib SHAR Dataset

URL of dataset: https://www.dropbox.com/s/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip?dl=0

+ UniMibの基本データ構造

x_data.mat (xはacc or adl or fall):

    shape: (Number of frames, 1d-vector contained x, y and z axes)
    すでにsliding-windowによってフレームに分けられたデータが格納されており，
    axis=1ではx, y, z軸のデータが連結されている．
    window_size = 151

x_labels.mat:

    shape: (Number of frames, 3)
    行動ラベル，被験者ラベル，試行ラベル(何回目の試行か？)が順に格納されている．

x_names.mat:

    shape: (2, 17)
    In axis 0, first item is class description, second item is class name.
    The number of classes is 17.

+ acc or others

UniMibのdataディレクトリには大きく分けてacc, adl, fallの3種類のデータがある．
fullというのもあるが構造は不明．

簡潔に説明すると，UniMibではacc = adl + fallである．
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat

from pathlib import Path
from typing import List, Tuple, Union, Optional
from ..core import split_using_sliding_window

from .base import BaseDataset, check_path


__all__ = ['UniMib', 'load', 'load_raw']


# Meta Info
SUBJECTS = tuple(range(1, 30+1))
ACTIVITIES = tuple(['StandingUpFS', 'StandingUpFL', 'Walking', 'Running', 'GoingUpS', 'Jumping', 'GoingDownS', 'LyingDownFS', 'SittingDown', 'FallingForw', 'FallingRight', 'FallingBack', 'HittingObstacle', 'FallingWithPS', 'FallingBackSC', 'Syncope', 'FallingLeft'])
GENDER = {'M': 0, 'F': 1}
Sampling_Rate = 50 # Hz




class UniMib(BaseDataset):
    """
    UniMib SHARデータセットに記録されているセンサデータとメタデータを読み込む．

    Parameters
    ----------
    path: Path
        UniMib SHARデータセットのパス(path/to/dataset/data)．
    """

    def __init__(self, path:Path):
        super().__init__(path)
    
    def load(self, data_type:str, window_size:Optional[int]=None, stride:Optional[int]=None, ftrim_sec:int=3, btrim_sec:int=3, subjects:Optional[list]=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        UniMib SHARデータセットを読み込み，sliding-window処理を行ったデータを返す．

        Parameters
        ----------
        data_type: str
            ロードするデータの種類(adl, fall, full, raw)を選択する(full = adl + fall)．
            rawは前処理済みデータではない生のデータを扱う．

        window_size: int
            フレーム分けするサンプルサイズ
            data_type != 'raw'の場合は強制的に151となるが，
            data_type == 'raw'の場合は必ず指定する必要がある．

        stride: int
            ウィンドウの移動幅

            data_type != 'raw'の場合は指定する必要はないが，
            data_type == 'raw'の場合は必ず指定する必要がある．

        ftrim_sec: int
            セグメント先頭のトリミングサイズ(単位は秒)

        btrim_sec: int
            セグメント末尾のトリミングサイズ(単位は秒)
        
        subjects: Optional[list]
            ロードする被験者を指定する．指定されない場合はすべての被験者のデータを返す．
            被験者は計9名おり，それぞれに整数のIDが割り当てられている．
            
            被験者ID: [1, 2, ..., 30]

        Returns
        -------
        (x_frames, y_frames): Tuple[np.ndarray, np.ndarray]
            sliding-windowで切り出した入力とターゲットのフレームリスト

            x_framesは3次元配列で構造は大まかに(Batch, Channels, Frame)のようになっている．
            Channelsは加速度センサの軸を表しており，先頭からx, y, zである．
            また，このローダはdata_typeによってwindow_sizeの挙動が変わり，
            data_type != 'raw'の場合はwindow_sizeは強制的に151となる．

            y_framesは2次元配列で構造は大まかに(Batch, Labels)のようになっている．
            Labelsは先頭からactivity，subjectを表している．

            y_framesはデータセット内の値をそのまま返すため，分類で用いる際はラベルの再割り当てが必要となることに注意する．

        Examples
        --------
        >>> unimib_path = Path('path/to/dataset')
        >>> unimib = UniMib(unimib_path)
        >>>
        >>> subjects = [1, 2, 3]
        >>>
        >>> x, y = unimib.load(data_type='full', subjects=subjects)
        >>> print('full - x: {}, y: {}'.format(x.shape, y.shape))
        >>> # > full - x: (?, 3, 151), y: (?, 2)
        >>>
        >>> x, y = unimib.load(data_type='raw', window_size=64, stride=64, ftrim_sec=0, btrim_sec=0, subjects=subjects)
        >>> print('raw - x: {}, y: {}'.format(x.shape, y.shape))
        >>> # > raw - x: (?, 3, 64), y: (?, 2)
        """

        if data_type != 'raw':
            data, meta = load(path=self.path, data_type=data_type)
            segments = {'acceleration': data, 'activity': meta['activity'], 'subject': meta['subject']}
            x = np.stack(segments['acceleration']).transpose(0, 2, 1)
            y = np.stack([segments['activity'], segments['subject']]).T
            x_frames = x
            y_frames = y

        else:
            if window_size is None or stride is None:
                raise ValueError('if data_type is "raw", window_size and stride must be specified.')
            data, meta = load(path=self.path, data_type='raw')
            segments = {'acceleration': data, 'activity': meta['activity'], 'subject': meta['subject']}
            x = segments['acceleration']
            y = np.stack([segments['activity'], segments['subject']]).T

            x_frames, y_frames = [], []
            for i in range(len(x)):
                fs = split_using_sliding_window(
                    np.array(x[i]), window_size=window_size, stride=stride,
                    ftrim=Sampling_Rate*ftrim_sec, btrim=Sampling_Rate*btrim_sec,
                    return_error_value=None)
                if fs is not None:
                    x_frames += [fs]
                    y_frames += [np.expand_dims(y[i], 0)]*len(fs)
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
            x_frames = x_frames[flags]
            y_frames = y_frames[flags]

        return x_frames, y_frames


def load(path:Union[Path,str], data_type:str='full') -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """Function for loading UniMib SHAR dataset

    Parameters
    ----------
    path: Union[Path, str]
        Directory path of UniMib SHAR dataset('data' directory).

    data_type: str
        Data type

        - 'full': segmented sensor data which contain all activities
        - 'adl' : segmented sensor data which contain ADL activities
        - 'fall': segmented sensor data which contain fall activities
        - 'raw' : raw sensor data (not segmented, all activities)

    Returns
    -------
    data, meta: List[pd.DataFrame], pd.DataFrame
        Sensor data segmented by activity and subject.

    See Alos
    --------
    The order of 'data' and 'meta' correspond.

    e.g. meta.iloc[0] is meta data of data[0].
    """
    path = check_path(path)

    raw = load_raw(path, data_type)
    data, meta = reformat(raw)
    return data, meta


def load_raw(path:Path, data_type:str='full') -> Union[Tuple[np.ndarray, pd.DataFrame], Tuple[List[pd.DataFrame], pd.DataFrame]]:
    """Function for loading raw data of UniMib SHAR dataset

    Parameters
    ----------
    path: Path
        Directory path of UniMib SHAR dataset('data' directory).

    data_type: str
        Data type

        - 'full': segmented sensor data which contain all activities
        - 'adl' : segmented sensor data which contain ADL activities
        - 'fall': segmented sensor data which contain fall activities
        - 'raw' : raw sensor data (not segmented, all activities)

    Returns
    -------
    data, meta: Tuple[np.ndarray, pd.DataFrame] or Tuple[List[np.ndarray], pd.DataFrame]

        * If data_type = 'full', 'adl' or 'fall': Tuple[np.ndarray, pd.DataFrame]

            Sensor data and meta data.
            Data shape is (?, 151, 3), and the second axis shows frames.
            Third axis is channel axis, which indicates x, y and z acceleration.
        
        * If data_type = 'raw': Tuple[List[np.ndarray], pd.DataFrame]

            Sensor data and meta data.
            Data shape is (?, ?, 3), and the second axis shows segments which is variable length.
            Third axis is channel, which indicates x, y and z acceleration.

    See Also
    --------
    [data_type is "full"]
    Range of activity label: [1, 17]
    Range of subject label : [1, 30]
    Range of trial label   : [1, 2] or [1, 6]

    [data_type is "adl"]
    Range of activity label: [1, 9]
    Range of subject label : [1, 30]

    [data_type is "fall"]
    Range of activity label: [1, 8]
    Range of subject label : [1, 30]

    [data_type is "raw"]
    Range of activity label: [1, 17]
    Range of subject label : [1, 30]
    Range of trial label   : [1, 2] or [1, 6]

    If data_type is not "raw", then segment size is 151, otherwise window size is not fixed.
    """

    if not isinstance(data_type, str):
        raise TypeError('expected type of "type" argument is str, but {}'.format(type(data_type)))
    if data_type not in ['full', 'adl', 'fall', 'raw']:
        raise ValueError('unknown data type, {}'.format(data_type))

    if data_type == 'full':
        prefix = 'acc'
    elif data_type == 'adl':
        prefix = 'adl'
    elif data_type == 'fall':
        prefix = 'fall'
    elif data_type == 'raw':
        prefix = 'full'
    # else:
    #     # not reach

    if data_type != 'raw':
        data = loadmat(str(path / f'{prefix}_data.mat'))[f'{prefix}_data'].reshape([-1, 3, 151])    # (?, 3, 151)
        labels = loadmat(str(path / f'{prefix}_labels.mat'))[f'{prefix}_labels']    # (?, 3)
        # activity_labels, subject_labels, trial_labels = labels[:, 0], labels[:, 1], labels[:, 2]
        # descriptions, class_names = loadmat(str(path / f'{prefix}_names.mat'))[f'{prefix}_names']

        meta = labels
        meta = pd.DataFrame(meta, columns=['activity', 'subject', 'trial_id'])
        meta = meta.astype({'activity': np.int8, 'subject': np.int8, 'trial_id': np.int8})
    else:
        full_data = loadmat(str(path / f'{prefix}_data.mat'))[f'{prefix}_data']
        sensor_data, activity_labels, subject_labels, trial_labels = [], [], [], []
        gender_labels, age_labels, height_labels, weight_labels = [], [], [], []
        for subject_id, d0 in enumerate(full_data):
            accs, gender, age, height, weight = d0
            expand_size = 0
            assert len(accs) == 1, '[Debug] detect not expected pattern'
            assert len(accs[0]) == 1, '[Debug] detect not expected pattern'
            for activity_id, acc in enumerate(accs[0][0]):  # accsの構造については要確認
                activity_labels += [activity_id+1]*len(acc)
                expand_size += len(acc)
                for trial_id, acc_trial in enumerate(acc):
                    assert len(acc_trial) == 1, '[Debug] detect not expected pattern'
                    trial_labels += [trial_id+1]
                    sd = acc_trial[0][:3]   # remove time instants and magnitude
                    sensor_data += [sd]
            subject_labels += [subject_id+1]*expand_size
            gender_labels += [gender[0]]*expand_size
            age_labels += [age[0][0]]*expand_size
            height_labels += [height[0][0]]*expand_size
            weight_labels += [weight[0][0]]*expand_size
        
        activity_labels = np.array(activity_labels, dtype=np.int8)
        subject_labels = np.array(subject_labels, dtype=np.int8)
        trial_labels = np.array(trial_labels, dtype=np.int8)
        gender_str_labels = np.array(gender_labels)
        gender_labels = np.zeros_like(gender_str_labels, dtype=np.int8)
        gender_labels[np.logical_or(gender_str_labels == 'M ', gender_str_labels == 'M ')] = GENDER['M']
        gender_labels[np.logical_or(gender_str_labels == 'F ', gender_str_labels == 'F ')] = GENDER['F']
        age_labels = np.array(age_labels, dtype=np.int8)
        height_labels = np.array(height_labels, dtype=np.int8)
        weight_labels = np.array(weight_labels, dtype=np.int8)

        assert len(sensor_data) == len(activity_labels)
        assert len(sensor_data) == len(subject_labels)
        assert len(sensor_data) == len(trial_labels)
        assert len(sensor_data) == len(gender_labels)
        assert len(sensor_data) == len(age_labels)
        assert len(sensor_data) == len(height_labels)
        assert len(sensor_data) == len(weight_labels)

        meta = np.stack([activity_labels, subject_labels, trial_labels, gender_labels, age_labels, height_labels, weight_labels]).T
        meta = pd.DataFrame(meta, columns=['activity', 'subject', 'trial_id', 'gender', 'age', 'height', 'weight'])
        # data = np.zeros(len(meta), dtype=np.object)
        # data[:] = sensor_data
        data = sensor_data

    return data, meta


def reformat(raw) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """Function for reformating

    Parameters
    ----------
    raw:
        data loaded by 'load_raw'
    
    Returns
    -------
    data, meta: List[pd.DataFrame], pd.DataFrame
        Sensor data segmented by activity and subject

    See Alos
    --------
    The order of 'data' and 'meta' correspond.

    e.g. meta.iloc[0] is meta data of data[0].
    """

    data, meta = raw
    data = list(map(lambda x: pd.DataFrame(x.T, columns=['x', 'y', 'z']), data))
    return data, meta


