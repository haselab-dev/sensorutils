import numpy as np
from scipy.io import loadmat

from pathlib import Path
from typing import Union
from ..core import split_using_sliding_window


"""
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

class UniMib:
    def __init__(self, path:Path):
        self.path = path
    
    def load(self, data_type:str, window_size:Union[int, None]=None, stride:Union[int, None]=None, ftrim_sec:int=3, btrim_sec:int=3, subjects:Union[list, None]=None):
        """UniMibの読み込みとsliding-window

        Parameters
        ----------
        data_type: str
            ロードするデータの種類(adl, fall, full, raw)を選択．
            full = adl + fall．
            rawは公式が提供している前処理済みデータではない真のrawデータを扱う．

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
            y_framesはデータセット内の値をそのまま返すため，分類で用いる際はラベルの再割り当てが必要となることに注意
        """

        if data_type != 'raw':
            segments = load(dataset_path=self.path, data_type=data_type)
            x = segments['acceleration']
            y = np.stack([segments['activity'], segments['subject']]).T
            x_frames = x
            y_frames = y

        else:
            if window_size is None or stride is None:
                raise ValueError('if data_type is "raw", window_size and stride must be specified.')
            segments = load(dataset_path=self.path, data_type='raw')
            x = segments['acceleration']
            y = np.stack([segments['activity'], segments['subject']]).T

            x_frames, y_frames = [], []
            for i in range(len(x)):
                fs = split_using_sliding_window(
                    x[i].T, window_size=window_size, stride=stride,
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
            flags = np.zeros(len(x_frames), dtype=np.bool)
            for sub in subjects:
                flags = np.logical_or(flags, y_frames[:, 1] == sub)
            x_frames = x_frames[flags]
            y_frames = y_frames[flags]

        return x_frames, y_frames


def load(dataset_path:Path, data_type:str='full'):
    """UniMib SHARの読み込み

    Parameters
    ----------
    dataset_path: Path
        UniMib SHARデータセットのディレクトリ(dataディレクトリ)
    
    data_type: str
        ロードするデータの種類を指定
        'full': segmented sensor data which contain all activities
        'adl' : segmented sensor data which contain ADL activities
        'fall': segmented sensor data which contain fall activities
        'raw' : raw sensor data (not segmented, all activities)

    Returns
    -------
    segments: dict
        sensor data and labels(not relabeld)
        format: {'acceleration': <acceleration>, 'activity': <activity_id>, 'subject': <subject_id>, 'trial': <trial_id>}
    
    See Also
    --------
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
        data = loadmat(str(dataset_path / f'{prefix}_data.mat'))[f'{prefix}_data'].reshape([-1, 3, 151])
        labels = loadmat(str(dataset_path / f'{prefix}_labels.mat'))[f'{prefix}_labels']
        activity_labels, subject_labels, trial_labels = labels[:, 0], labels[:, 1], labels[:, 2]
        # descriptions, class_names = loadmat(str(dataset_path / f'{prefix}_names.mat'))[f'{prefix}_names']

        assert len(data) == len(activity_labels)
        assert len(data) == len(subject_labels)
        assert len(data) == len(trial_labels)

        segments = {'acceleration': data, 'activity': activity_labels, 'subject': subject_labels, 'trial': trial_labels}
    else:
        full_data = loadmat(str(dataset_path / f'{prefix}_data.mat'))[f'{prefix}_data']
        sensor_data, activity_labels, subject_labels, trial_labels = [], [], [], []
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
        
        assert len(subject_labels) == len(activity_labels)
        assert len(subject_labels) == len(trial_labels)

        activity_labels = np.array(activity_labels, dtype=np.uint8)
        subject_labels = np.array(subject_labels, dtype=np.uint8)
        trial_labels = np.array(trial_labels, dtype=np.uint8)

        assert len(sensor_data) == len(activity_labels)
        assert len(sensor_data) == len(subject_labels)
        assert len(sensor_data) == len(trial_labels)

        segments = {'acceleration': sensor_data, 'activity': activity_labels, 'subject': subject_labels, 'trial': trial_labels}

    return segments

SUBJECTS = tuple(range(1, 30+1))
ACTIVITIES = tuple(['StandingUpFS', 'StandingUpFL', 'Walking', 'Running', 'GoingUpS', 'Jumping', 'GoingDownS', 'LyingDownFS', 'SittingDown', 'FallingForw', 'FallingRight', 'FallingBack', 'HittingObstacle', 'FallingWithPS', 'FallingBackSC', 'Syncope', 'FallingLeft'])
Sampling_Rate = 50 # Hz

if __name__ == '__main__':
    import pprint
    base_dir = Path('path/to/dataset')
    segments = load(base_dir, data_type='full')
    pprint.pprint(segments, depth=1)
    print(segments['acceleration'].shape)

    unimib = UniMib(base_dir)
    subjects = [2, 4, 6, 8, 10, 12]
    # subjects = None
    x, y = unimib.load(data_type='full', window_size=151, stride=20, ftrim_sec=3, btrim_sec=3, subjects=subjects)
    print('x: {}'.format(x.shape))
    print('y: {}'.format(y.shape))
    print(np.unique(y[:, 1]))
