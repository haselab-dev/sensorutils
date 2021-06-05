import numpy as np
import pandas as pd
from scipy.fftpack import fft, ifft, fftfreq
import itertools
import copy

from pathlib import Path
from ..core import split_using_sliding_window, split_using_target
from typing import Union, List, Dict, Tuple

from .base import BaseDataset


__all__ = ['HHAR', 'load']


# Meta Info
# ATTRIBUTES = ['acc','gyro']

DEVICE_TYPES = ['Phone', 'Watch']
SENSOR_TYPES = ['accelerometer', 'gyroscope']

SUBJECTS = dict(zip(['a','b','c','d','e','f','g','h','i'], list(range(9))))

ACTIVITIES = {
    'bike': 1, 
    'sit': 2, 
    'stand': 3, 
    'walk': 4, 
    'stairsup': 5, 
    'stairsdown': 6,
    'null': 0,
}

PHONE_DEVICES = {
    'nexus4_1': 0, 'nexus4_2': 1,
    's3_1': 2, 's3_2': 3,
    's3mini_1': 4,'s3mini_2': 5,
    'samsungold_1': 6, 'samsungold_2': 7,
}
WATCH_DEVICES = {
    'gear_1': 8, 'gear_2': 9,
    'lgwatch_1': 10, 'lgwatch_2': 11,
}

Column = [
    'Index', 
    'Arrival_Time', 
    'Creation_Time', 
    'x', 
    'y', 
    'z', 
    'User',
    'Model', 'Device', 'gt',
]

def __id2act(act_id:int) -> str:
    return ACTIVITIES[act_id]

def __act2id(act:str) -> int:
    if act in ACTIVITIES:
        return ACTIVITIES[act]
    raise ValueError(f'Unknown activity ({act})')

def __name2id(name:str, name_list:Dict[str, int]) -> int:
    if name in name_list:
        return name_list[name]
    raise ValueError(f'Unknown name ({name})')


class HHAR(BaseDataset):
    """HHAR 
    
    HHAR <https://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition> データセットの行動分類を行うためのローダクラス
    """

    def __init__(self, path:Union[str,Path]):
        """
        Parameters
        ----------
        path : Union[str,Path]
            Directory Path that include csv file: Phones_[accelerometer,gyroscope].csv, Watch_[accelerometer,gyroscope].csv
        """

        if type(path) is str:
            path = Path(path)
        super().__init__(path)
    
    def load(self, sensor_types:Union[List[str], str], device_types:Union[List[str], str], window_size:int, stride:int, subjects:Union[list, None]=None):
        """HHARの読み込みとsliding-window

        Parameters
        ----------
        sensor_type:
            センサタイプ．"acceleromter" or "gyroscope" or ["acceleromter", "gyroscope"]

            [Caution!!!]
            "accelerometer"と"gyroscope"の同時指定は非推奨．

        device_type:
            デバイスタイプ．"Phone" or "Watch"

        window_size: int
            フレーム分けするサンプルサイズ

        stride: int
            ウィンドウの移動幅

        subjects:
            ロードする被験者を指定

        Returns
        -------
        (x_frames, y_frames): tuple
            sliding-windowで切り出した入力とターゲットのフレームリスト
            y_framesはデータセット内の値をそのまま返すため，分類で用いる際はラベルの再割り当てが必要となることに注意

            y_framesのshapeは(*, 4)であり，axis=1ではUser, Model, Device, Activityの順でデータが格納されている．
        
        """

        if isinstance(sensor_types, str):
            sensor_types = [sensor_types]
        if not isinstance(sensor_types, list):
            raise TypeError('expected type of "sensor_types" is str or list, but got {}'.format(type(sensor_types)))
        if isinstance(device_types, str):
            device_types = [device_types]
        if not isinstance(device_types, list):
            raise TypeError('expected type of "device_types" is str or list, but got {}'.format(type(device_types)))

        segments = []
        for dev_type in device_types:
            segments += load(self.path, sensor_type=sensor_types, device_type=dev_type)
        n_ch = len(sensor_types)
        segments = [seg.reshape(-1, n_ch*10) for seg in segments]

        frames = []
        for seg in segments:
            fs = split_using_sliding_window(
                np.array(seg), window_size=window_size, stride=stride,
                ftrim=0, btrim=0,
                return_error_value=None)
            if fs is not None:
                frames += [fs]
            else:
                # print('no frame')
                pass
        frames = np.concatenate(frames)
        N, ws, _ = frames.shape
        frames = frames.reshape([N, ws, n_ch, 10])
        x_frames = frames[:, :, :, 3:6].reshape([N, ws, -1])
        y_frames = frames[:, 0, 0, 6:]

        # subject filtering
        if subjects is not None:
            flags = np.zeros(len(x_frames), dtype=np.bool)
            for sub in subjects:
                flags = np.logical_or(flags, y_frames[:, 0] == SUBJECTS[sub])
            x_frames = x_frames[flags]
            y_frames = y_frames[flags]

        return x_frames, y_frames


def _load_as_dataframe(path:Path, device_type:str):
    df = pd.read_csv(path)
    df['gt'] = df['gt'].fillna('null')
    df['gt'] = df['gt'].map(__act2id)
    df['User'] = df['User'].map(lambda x: __name2id(x, SUBJECTS))
    dev_list = copy.deepcopy(PHONE_DEVICES) if device_type == DEVICE_TYPES[0] else copy.deepcopy(WATCH_DEVICES)
    df['Device'] = df['Device'].map(lambda x: __name2id(x, dev_list))
    return df

def _load_segments(path:Path, sensor_type:str, device_type:str):
    """
    HHARデータセットでは被験者は決められたスクリプトに従って行動しそれに伴うセンサデータを収集している．
    被験者はHHARで採用されたすべてのデバイスでデータの収集を行っているため，
    下記のコードのように被験者とデバイスのすべての組み合わせでセグメントに分割することで，
    加速度センサとジャイロセンサを連結しやすくしている．
    """
    df = _load_as_dataframe(path, device_type)

    if device_type == DEVICE_TYPES[0]:
        n_dev, base = 8, 0
    elif device_type == DEVICE_TYPES[1]:
        n_dev, base = 4, 8

    segments = {}
    # split by subjects
    splited_sub = split_using_target(df.to_numpy(), df['User'].to_numpy())

    for sub in range(9):
        segments[sub] = {}
        if sub in splited_sub:
            assert len(splited_sub[sub]) == 1, 'detect not expected pattern'
            seg = splited_sub[sub][0]
            # split by device
            splited_sub_dev = split_using_target(seg, seg[:, -2])
        else:
            assert True, 'detect not expected pattern'
            splited_sub_dev = {}

        # PhoneとWatchで最小のIDが異なる
        for dev in range(base, base+n_dev):
            segments[sub][dev] = {}
            if dev in splited_sub_dev:
                assert len(splited_sub_dev[dev]) == 1, 'detect not expected pattern, ({})'.format(len(splited_sub_dev[dev]))
                seg = splited_sub_dev[dev][0]
                splited_sub_dev_act = split_using_target(seg, seg[:, -1])
            else:
                assert True, 'detect not expected pattern'
                splited_sub_dev_act = {}
            
            for act in range(0, 7):
                if act in splited_sub_dev_act:
                    segments[sub][dev][act] = splited_sub_dev_act[act]
                else:
                    segments[sub][dev][act] = None

    return segments

def _lpf(y:np.ndarray, fpass:int, fs:int) -> np.ndarray:
    """low pass filter
    Parameters
    ----------
    y: np.ndarray
        source data
    fpass: float
        catoff frequency
    fs: int
        sampling frequency
    Returns
    -------
    np.ndarray:
        filtered data
    """
    yf = fft(y.copy())
    freq = fftfreq(len(y), 1./fs)
    idx = np.logical_or(freq > fpass, freq < -fpass)
    yf[idx] = 0.

    yd = ifft(yf)
    yd = np.real(yd)

    return yd

def _align_creation_time(seg_acc, seg_gyro):
    if seg_acc[0, 2] == seg_gyro[0, 2]:
        return seg_acc, seg_gyro
    elif seg_acc[0, 2] < seg_gyro[0, 2]:
        fst, snd = 0, 1
    elif seg_acc[0, 2] > seg_gyro[0, 2]:
        fst, snd = 1, 0
    segs = [seg_acc, seg_gyro]

    if segs[snd][0, 2] >= segs[fst][-1, 2]:
        raise RuntimeError('This is bug. Probably, Correspondence between acc and gyro is invalid.')

    for i in range(1, len(segs[fst])):
        if segs[fst][i, 2] == segs[snd][0, 2]:
            segs[fst] = segs[fst][i:]
            return segs
        elif segs[fst][i, 2] > segs[snd][0, 2]:
            if segs[fst][i-1, 2] - segs[snd][0, 2] < segs[fst][i, 2] - segs[snd][0, 2]:
                segs[fst] = segs[fst][i-1:]
            else:
                segs[fst] = segs[fst][i:]
            return segs

def load(path:Path, sensor_type:Union[str, List[str]], device_type:str='Watch') -> List[np.ndarray]:
    """HHAR の加速度センサデータの読み込み関数

    Parameters
    ----------
    path: Path
        HHARデータセットのディレクトリ
    
    sensor_type:
        "accelerometer" or "gyroscope" or ["accelerometer", "gyroscope"]
        
        [Caution!!!]
        "accelerometer"と"gyroscope"の同時指定は非推奨．
    
    device_type:
        "Watch" or "Phone"

    Returns
    -------
    segments: list
        被験者・デバイス・行動ラベルをもとにセグメンテーションされたデータ

    See Also
    --------
    HHARデータセットは加速度センサデータとジャイロセンサデータを同時に入力として用いることをあまり想定していない(っぽい)．
    そのため，厳密に加速度とジャイロセンサデータの連結しよとすると非常にコストが高い．
    そこで今回は精度を犠牲にして計算コストを下げている．
    """
    if (device_type[0] == 'w') or (device_type[0] == 'W'):
        device_type = DEVICE_TYPES[1]
    else:
        device_type = DEVICE_TYPES[0] # default
    print('Device: {}'.format(device_type))

    if isinstance(sensor_type, (list, tuple, np.ndarray)):
        if len(sensor_type) == 0:
            raise ValueError('specified at least one type')
        if not (set(sensor_type) <= set(SENSOR_TYPES)):
            raise ValueError('include unknown sensor type, {}'.format(sensor_type))
    elif isinstance(sensor_type, str):
        if sensor_type not in SENSOR_TYPES:
            raise ValueError('unknown sensor type, {}'.format(sensor_type))
    else:
        raise TypeError('expected type of "sensor_type" is list, tuple, numpy.ndarray or str, but got {}'.format(type(sensor_type)))
    print('Sensor type: {}'.format(sensor_type))

    # prepare csv path
    sensor_type_list = [sensor_type] if isinstance(sensor_type, str) else sensor_type
    sensor_type_list = sorted(sensor_type_list) # accelerometer, gyroの順になることを保証する
    if device_type == DEVICE_TYPES[0]:
        csv_files = list(path / (f'Phones_{sensor_type}.csv') for sensor_type in sensor_type_list)
    elif device_type == DEVICE_TYPES[1]:
        csv_files = list(path / (f'Watch_{sensor_type}.csv') for sensor_type in sensor_type_list)

    if len(sensor_type_list) == 1:
        df = _load_as_dataframe(csv_files[0], device_type)
        domains = (df['gt'] + df['User']*10 + df['Device']*100).to_numpy()
        segments = split_using_target(df.to_numpy(), domains)
        segments = list(itertools.chain(*list(segments.values())))
    elif set(sensor_type_list) == set(SENSOR_TYPES):
        segs = [_load_segments(csv_path, sensor_type, device_type) for sensor_type, csv_path in zip(sensor_type_list, csv_files)]
        segs_acc_sub_dev_act, segs_gyro_sub_dev_act = segs

        if device_type == DEVICE_TYPES[0]:
            n_dev, base = 8, 0
        elif device_type == DEVICE_TYPES[1]:
            n_dev, base = 4, 8

        # concat acc and gyro
        segments = []
        patterns = list(itertools.product(range(9), range(base, base+n_dev), range(7))) # subject, device, activity
        for sub, dev, act in patterns:
            s_accs, s_gyros = segs_acc_sub_dev_act[sub][dev][act], segs_gyro_sub_dev_act[sub][dev][act]

            # このパターンは主に欠損値でヒット
            if s_accs is None or s_gyros is None:
                # print(' > [skip] ({})-({})-({}), seg_acc = {}, seg_gyro = {}'.format(sub, dev, act, type(s_accs), type(s_gyros)))
                continue
    
            # このパターンでは加速度とジャイロでセグメント数がずれているときにヒット
            # ただし，セグメント数のずれはわずかなラベリングのずれによる者であるので大部分の対応関係は保たれているはず
            if len(s_accs) != len(s_gyros):
                # print(' > [Warning] length of s_accs and s_gyros are different, {}, {}'.format(len(s_accs), len(s_gyros)))
                continue
            
            for s_acc, s_gyro in zip(s_accs, s_gyros):

                # s3_1とs3_2は加速度センサとジャイロセンサのサンプリング周波数が異なるためダウンサンプリングで対応
                # このパターンはcreation_timeのずれが大きいためこれを許容するかは検討の余地がある
                if device_type == DEVICE_TYPES[0] and s_acc[0, -2] in [2, 3]:
                    assert s_gyro[0, -2] in [2, 3], 'this is bug'
                    # s_gyro[:, 3] = _lpf(s_gyro[:, 3], fpass=150, fs=200)
                    # s_gyro[:, 4] = _lpf(s_gyro[:, 4], fpass=150, fs=200)
                    # s_gyro[:, 5] = _lpf(s_gyro[:, 5], fpass=150, fs=200)
                    # s_gyro = s_gyro[::2]
                    continue
                
                try:
                    s_acc, s_gyro = _align_creation_time(s_acc, s_gyro)
                except RuntimeError as e:
                    # Watchではなぜかこれに引っかかるsegmentが多数ある
                    print(f'>>> {e}')
                    continue

                segs = [s_acc, s_gyro]
                min_seg_idx = 0 if len(s_acc) - len(s_gyro) <= 0 else 1
                other_idx = (min_seg_idx + 1) % 2
                min_len_seg = len(segs[min_seg_idx])

                # segmentの長さを比較
                # print('diff of length of segments: {}, {} ns'.format(len(segs[0]) - len(segs[1]), (segs[0][0, 2]-segs[1][0, 2])*1e-9))

                # 各セグメントの先頭のcreation timeにほとんど差がないため，
                # 先頭をそろえて長さを短いほうに合わせることで対応
                segs[other_idx] = segs[other_idx][:min_len_seg]

                # 先頭のcreation timeを比較
                # d = (segs[min_seg_idx][0, 2] - segs[other_idx][0, 2]) * 1e-9
                # if abs(d) > 1e-3:
                #     print('diff of creation time in front: {} ns'.format(d))

                segs = np.concatenate([np.expand_dims(segs[0], 1), np.expand_dims(segs[1], 1)], axis=1)
                segments += [segs]
    
    return segments
