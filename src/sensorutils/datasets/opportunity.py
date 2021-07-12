"""Opportunity(UCI) Dataset

URL of dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip
"""

import numpy as np
import pandas as pd
import itertools
from pathlib import Path
from typing import List, Tuple, Optional, Union
from ..core import split_using_target, split_using_sliding_window

from .base import BaseDataset, check_path


__all__ = ['Opportunity', 'load', 'load_raw']


# Meta Info
Sampling_Rate = 30  # Hz

# NOTE: Locomotion と辞書があってない可能性がある
Locomotion = {
    1: 'Stand',
    2: 'Walk',
    4: 'Sit',
    5: 'Lie',
}
HL_Activity = {
    101: 'Relaxing',
    102: 'Coffee time',
    103: 'Early morning',
    104: 'Cleanup',
    105: 'Sandwich time',
}
LL_Left_Arm = {
    201: 'unlock',
    202: 'stir',
    203: 'lock',
    204: 'close',
    205: 'reach',
    206: 'open',
    207: 'sip',
    208: 'clean',
    209: 'bite',
    210: 'cut',
    211: 'spread',
    212: 'release',
    213: 'move',
}
LL_Left_Arm_Object = {
    301: 'Bottle',
    302: 'Salami',
    303: 'Bread',
    304: 'Sugar',
    305: 'Dishwasher',
    306: 'Switch',
    307: 'Milk',
    308: 'Drawer3 (lower)',
    309: 'Spoon',
    310: 'Knife cheese',
    311: 'Drawer2 (middle)',
    312: 'Table',
    313: 'Glass',
    314: 'Cheese',
    315: 'Chair',
    316: 'Door1',
    317: 'Door2',
    318: 'Plate',
    319: 'Drawer1 (top)',
    320: 'Fridge',
    321: 'Cup',
    322: 'Knife salami',
    323: 'Lazychair',
}
LL_Right_Arm = {
    401: 'unlock',
    402: 'stir',
    403: 'lock',
    404: 'close',
    405: 'reach',
    406: 'open',
    407: 'sip',
    408: 'clean',
    409: 'bite',
    410: 'cut',
    411: 'spread',
    412: 'release',
    413: 'move',
}
LL_Right_Arm_Object = {
    501: 'Bottle',
    502: 'Salami',
    503: 'Bread',
    504: 'Sugar',
    505: 'Dishwasher',
    506: 'Switch',
    507: 'Milk',
    508: 'Drawer3 (lower)',
    509: 'Spoon',
    510: 'Knife cheese',
    511: 'Drawer2 (middle)',
    512: 'Table',
    513: 'Glass',
    514: 'Cheese',
    515: 'Chair',
    516: 'Door1',
    517: 'Door2',
    518: 'Plate',
    519: 'Drawer1 (top)',
    520: 'Fridge',
    521: 'Cup',
    522: 'Knife salami',
    523: 'Lazychair',
}
ML_Both_Arms = {
    406516: 'Open Door 1',
    406517: 'Open Door 2',
    404516: 'Close Door 1',
    404517: 'Close Door 2',
    406520: 'Open Fridge',
    404520: 'Close Fridge',
    406505: 'Open Dishwasher',
    404505: 'Close Dishwasher',
    406519: 'Open Drawer 1',
    404519: 'Close Drawer 1',
    406511: 'Open Drawer 2',
    404511: 'Close Drawer 2',
    406508: 'Open Drawer 3',
    404508: 'Close Drawer 3',
    408512: 'Clean Table',
    407521: 'Drink from Cup',
    405506: 'Toggle Switch',
}
PERSONS = [f'S{i}' for i in range(1, 4+1)]
Sensors = {'Accelerometer', 'InertialMeasurementUnit', 'REED_SWITCH', 'LOCATION'}
Attributes = {
    'acc',
    'gyro',
    'magnetic',
    'Quaternion',
    'S',
    'Eu',
    'Nav_A',
    'Body_A',
    'AngVelBodyFrame',
    'AngVelNavFrame',
    'Compass',
}
Positions = {
    'RKN^',
    'HIP',
    'LUA^',
    'RUA',
    'LH',
    'BACK',
    'RKN_',
    'RWR',
    'RUA^',
    'LUA_',
    'LWR',
    'RH',
    'RUA',
    'RLA',
    'LUA',
    'LLA',
    'L-SHOE',
    'R-SHOE',
    'CUP',
    'SALAMI',
    'WATER',
    'CHEESE',
    'BREAD',
    'KNIFE1',
    'MILK',
    'SPOON',
    'SUGAR',
    'KNIFE2',
    'PLATE',
    'GLASS',
    'FRIDGE',
    'MIDDLEDRAWER',
    'LOWERDRAWER',
    'UPPERDRAWER',
    'DISHWASHER',
    'DOOR1',
    'LAZYCHAIR',
    'DOOR2',
    'TAG1',
    'TAG2',
    'TAG3',
    'TAG4',
}
Axes = {'X', 'Y', 'Z', '1', '2', '3', '4'}
Column = [
    'MILLISEC',
    'Accelerometer_RKN^_accX',
    'Accelerometer_RKN^_accY',
    'Accelerometer_RKN^_accZ',
    'Accelerometer_HIP_accX',
    'Accelerometer_HIP_accY',
    'Accelerometer_HIP_accZ',
    'Accelerometer_LUA^_accX',
    'Accelerometer_LUA^_accY',
    'Accelerometer_LUA^_accZ',
    'Accelerometer_RUA_accX',
    'Accelerometer_RUA_accY',
    'Accelerometer_RUA_accZ',
    'Accelerometer_LH_accX',
    'Accelerometer_LH_accY',
    'Accelerometer_LH_accZ',
    'Accelerometer_BACK_accX',
    'Accelerometer_BACK_accY',
    'Accelerometer_BACK_accZ',
    'Accelerometer_RKN__accX',
    'Accelerometer_RKN__accY',
    'Accelerometer_RKN__accZ',
    'Accelerometer_RWR_accX',
    'Accelerometer_RWR_accY',
    'Accelerometer_RWR_accZ',
    'Accelerometer_RUA^_accX',
    'Accelerometer_RUA^_accY',
    'Accelerometer_RUA^_accZ',
    'Accelerometer_LUA__accX',
    'Accelerometer_LUA__accY',
    'Accelerometer_LUA__accZ',
    'Accelerometer_LWR_accX',
    'Accelerometer_LWR_accY',
    'Accelerometer_LWR_accZ',
    'Accelerometer_RH_accX',
    'Accelerometer_RH_accY',
    'Accelerometer_RH_accZ',
    'InertialMeasurementUnit_BACK_accX',
    'InertialMeasurementUnit_BACK_accY',
    'InertialMeasurementUnit_BACK_accZ',
    'InertialMeasurementUnit_BACK_gyroX',
    'InertialMeasurementUnit_BACK_gyroY',
    'InertialMeasurementUnit_BACK_gyroZ',
    'InertialMeasurementUnit_BACK_magneticX',
    'InertialMeasurementUnit_BACK_magneticY',
    'InertialMeasurementUnit_BACK_magneticZ',
    'InertialMeasurementUnit_BACK_Quaternion1',
    'InertialMeasurementUnit_BACK_Quaternion2',
    'InertialMeasurementUnit_BACK_Quaternion3',
    'InertialMeasurementUnit_BACK_Quaternion4',
    'InertialMeasurementUnit_RUA_accX',
    'InertialMeasurementUnit_RUA_accY',
    'InertialMeasurementUnit_RUA_accZ',
    'InertialMeasurementUnit_RUA_gyroX',
    'InertialMeasurementUnit_RUA_gyroY',
    'InertialMeasurementUnit_RUA_gyroZ',
    'InertialMeasurementUnit_RUA_magneticX',
    'InertialMeasurementUnit_RUA_magneticY',
    'InertialMeasurementUnit_RUA_magneticZ',
    'InertialMeasurementUnit_RUA_Quaternion1',
    'InertialMeasurementUnit_RUA_Quaternion2',
    'InertialMeasurementUnit_RUA_Quaternion3',
    'InertialMeasurementUnit_RUA_Quaternion4',
    'InertialMeasurementUnit_RLA_accX',
    'InertialMeasurementUnit_RLA_accY',
    'InertialMeasurementUnit_RLA_accZ',
    'InertialMeasurementUnit_RLA_gyroX',
    'InertialMeasurementUnit_RLA_gyroY',
    'InertialMeasurementUnit_RLA_gyroZ',
    'InertialMeasurementUnit_RLA_magneticX',
    'InertialMeasurementUnit_RLA_magneticY',
    'InertialMeasurementUnit_RLA_magneticZ',
    'InertialMeasurementUnit_RLA_Quaternion1',
    'InertialMeasurementUnit_RLA_Quaternion2',
    'InertialMeasurementUnit_RLA_Quaternion3',
    'InertialMeasurementUnit_RLA_Quaternion4',
    'InertialMeasurementUnit_LUA_accX',
    'InertialMeasurementUnit_LUA_accY',
    'InertialMeasurementUnit_LUA_accZ',
    'InertialMeasurementUnit_LUA_gyroX',
    'InertialMeasurementUnit_LUA_gyroY',
    'InertialMeasurementUnit_LUA_gyroZ',
    'InertialMeasurementUnit_LUA_magneticX',
    'InertialMeasurementUnit_LUA_magneticY',
    'InertialMeasurementUnit_LUA_magneticZ',
    'InertialMeasurementUnit_LUA_Quaternion1',
    'InertialMeasurementUnit_LUA_Quaternion2',
    'InertialMeasurementUnit_LUA_Quaternion3',
    'InertialMeasurementUnit_LUA_Quaternion4',
    'InertialMeasurementUnit_LLA_accX',
    'InertialMeasurementUnit_LLA_accY',
    'InertialMeasurementUnit_LLA_accZ',
    'InertialMeasurementUnit_LLA_gyroX',
    'InertialMeasurementUnit_LLA_gyroY',
    'InertialMeasurementUnit_LLA_gyroZ',
    'InertialMeasurementUnit_LLA_magneticX',
    'InertialMeasurementUnit_LLA_magneticY',
    'InertialMeasurementUnit_LLA_magneticZ',
    'InertialMeasurementUnit_LLA_Quaternion1',
    'InertialMeasurementUnit_LLA_Quaternion2',
    'InertialMeasurementUnit_LLA_Quaternion3',
    'InertialMeasurementUnit_LLA_Quaternion4',
    'InertialMeasurementUnit_L-SHOE_EuX',
    'InertialMeasurementUnit_L-SHOE_EuY',
    'InertialMeasurementUnit_L-SHOE_EuZ',
    'InertialMeasurementUnit_L-SHOE_Nav_Ax',
    'InertialMeasurementUnit_L-SHOE_Nav_Ay',
    'InertialMeasurementUnit_L-SHOE_Nav_Az',
    'InertialMeasurementUnit_L-SHOE_Body_Ax',
    'InertialMeasurementUnit_L-SHOE_Body_Ay',
    'InertialMeasurementUnit_L-SHOE_Body_Az',
    'InertialMeasurementUnit_L-SHOE_AngVelBodyFrameX',
    'InertialMeasurementUnit_L-SHOE_AngVelBodyFrameY',
    'InertialMeasurementUnit_L-SHOE_AngVelBodyFrameZ',
    'InertialMeasurementUnit_L-SHOE_AngVelNavFrameX',
    'InertialMeasurementUnit_L-SHOE_AngVelNavFrameY',
    'InertialMeasurementUnit_L-SHOE_AngVelNavFrameZ',
    'InertialMeasurementUnit_L-SHOE_Compass',
    'InertialMeasurementUnit_R-SHOE_EuX',
    'InertialMeasurementUnit_R-SHOE_EuY',
    'InertialMeasurementUnit_R-SHOE_EuZ',
    'InertialMeasurementUnit_R-SHOE_Nav_Ax',
    'InertialMeasurementUnit_R-SHOE_Nav_Ay',
    'InertialMeasurementUnit_R-SHOE_Nav_Az',
    'InertialMeasurementUnit_R-SHOE_Body_Ax',
    'InertialMeasurementUnit_R-SHOE_Body_Ay',
    'InertialMeasurementUnit_R-SHOE_Body_Az',
    'InertialMeasurementUnit_R-SHOE_AngVelBodyFrameX',
    'InertialMeasurementUnit_R-SHOE_AngVelBodyFrameY',
    'InertialMeasurementUnit_R-SHOE_AngVelBodyFrameZ',
    'InertialMeasurementUnit_R-SHOE_AngVelNavFrameX',
    'InertialMeasurementUnit_R-SHOE_AngVelNavFrameY',
    'InertialMeasurementUnit_R-SHOE_AngVelNavFrameZ',
    'InertialMeasurementUnit_R-SHOE_Compass',
    'Accelerometer_CUP_accX',
    'Accelerometer_CUP_accX',
    'Accelerometer_CUP_accX',
    'Accelerometer_CUP_gyroX',
    'Accelerometer_CUP_gyroY',
    'Accelerometer_SALAMI_accX',
    'Accelerometer_SALAMI_accX',
    'Accelerometer_SALAMI_accX',
    'Accelerometer_SALAMI_gyroX',
    'Accelerometer_SALAMI_gyroY',
    'Accelerometer_WATER_accX',
    'Accelerometer_WATER_accX',
    'Accelerometer_WATER_accX',
    'Accelerometer_WATER_gyroX',
    'Accelerometer_WATER_gyroY',
    'Accelerometer_CHEESE_accX',
    'Accelerometer_CHEESE_accX',
    'Accelerometer_CHEESE_accX',
    'Accelerometer_CHEESE_gyroX',
    'Accelerometer_CHEESE_gyroY',
    'Accelerometer_BREAD_accX',
    'Accelerometer_BREAD_accX',
    'Accelerometer_BREAD_accX',
    'Accelerometer_BREAD_gyroX',
    'Accelerometer_BREAD_gyroY',
    'Accelerometer_KNIFE1_accX',
    'Accelerometer_KNIFE1_accX',
    'Accelerometer_KNIFE1_accX',
    'Accelerometer_KNIFE1_gyroX',
    'Accelerometer_KNIFE1_gyroY',
    'Accelerometer_MILK_accX',
    'Accelerometer_MILK_accX',
    'Accelerometer_MILK_accX',
    'Accelerometer_MILK_gyroX',
    'Accelerometer_MILK_gyroY',
    'Accelerometer_SPOON_accX',
    'Accelerometer_SPOON_accX',
    'Accelerometer_SPOON_accX',
    'Accelerometer_SPOON_gyroX',
    'Accelerometer_SPOON_gyroY',
    'Accelerometer_SUGAR_accX',
    'Accelerometer_SUGAR_accX',
    'Accelerometer_SUGAR_accX',
    'Accelerometer_SUGAR_gyroX',
    'Accelerometer_SUGAR_gyroY',
    'Accelerometer_KNIFE2_accX',
    'Accelerometer_KNIFE2_accX',
    'Accelerometer_KNIFE2_accX',
    'Accelerometer_KNIFE2_gyroX',
    'Accelerometer_KNIFE2_gyroY',
    'Accelerometer_PLATE_accX',
    'Accelerometer_PLATE_accX',
    'Accelerometer_PLATE_accX',
    'Accelerometer_PLATE_gyroX',
    'Accelerometer_PLATE_gyroY',
    'Accelerometer_GLASS_accX',
    'Accelerometer_GLASS_accX',
    'Accelerometer_GLASS_accX',
    'Accelerometer_GLASS_gyroX',
    'Accelerometer_GLASS_gyroY',
    'REED_SWITCH_DISHWASHER_S1',
    'REED_SWITCH_FRIDGE_S3',
    'REED_SWITCH_FRIDGE_S2',
    'REED_SWITCH_FRIDGE_S1',
    'REED_SWITCH_MIDDLEDRAWER_S1',
    'REED_SWITCH_MIDDLEDRAWER_S2',
    'REED_SWITCH_MIDDLEDRAWER_S3',
    'REED_SWITCH_LOWERDRAWER_S3',
    'REED_SWITCH_LOWERDRAWER_S2',
    'REED_SWITCH_UPPERDRAWER',
    'REED_SWITCH_DISHWASHER_S3',
    'REED_SWITCH_LOWERDRAWER_S1',
    'REED_SWITCH_DISHWASHER_S2',
    'Accelerometer_DOOR1_accX',
    'Accelerometer_DOOR1_accY',
    'Accelerometer_DOOR1_accZ',
    'Accelerometer_LAZYCHAIR_accX',
    'Accelerometer_LAZYCHAIR_accY',
    'Accelerometer_LAZYCHAIR_accZ',
    'Accelerometer_DOOR2_accX',
    'Accelerometer_DOOR2_accY',
    'Accelerometer_DOOR2_accZ',
    'Accelerometer_DISHWASHER_accX',
    'Accelerometer_DISHWASHER_accY',
    'Accelerometer_DISHWASHER_accZ',
    'Accelerometer_UPPERDRAWER_accX',
    'Accelerometer_UPPERDRAWER_accY',
    'Accelerometer_UPPERDRAWER_accZ',
    'Accelerometer_LOWERDRAWER_accX',
    'Accelerometer_LOWERDRAWER_accY',
    'Accelerometer_LOWERDRAWER_accZ',
    'Accelerometer_MIDDLEDRAWER_accX',
    'Accelerometer_MIDDLEDRAWER_accY',
    'Accelerometer_MIDDLEDRAWER_accZ',
    'Accelerometer_FRIDGE_accX',
    'Accelerometer_FRIDGE_accY',
    'Accelerometer_FRIDGE_accZ',
    'LOCATION_TAG1_X',
    'LOCATION_TAG1_Y',
    'LOCATION_TAG1_Z',
    'LOCATION_TAG2_X',
    'LOCATION_TAG2_Y',
    'LOCATION_TAG2_Z',
    'LOCATION_TAG3_X',
    'LOCATION_TAG3_Y',
    'LOCATION_TAG3_Z',
    'LOCATION_TAG4_X',
    'LOCATION_TAG4_Y',
    'LOCATION_TAG4_Z',
    'Locomotion',
    'HL_Activity',
    'LL_Left_Arm',
    'LL_Left_Arm_Object',
    'LL_Right_Arm',
    'LL_Right_Arm_Object',
    'ML_Both_Arms',
    #'subject',     # ユーザID(ローダ側で付与)
]


"""
下記のcolumnは重複している。
ドキュメントのミスなのか仕様なのか。。。？
Accelerometer_CHEESE_accX 3
Accelerometer_SPOON_accX 3
Accelerometer_KNIFE1_accX 3
Accelerometer_KNIFE2_accX 3
Accelerometer_PLATE_accX 3
Accelerometer_GLASS_accX 3
Accelerometer_SALAMI_accX 3
Accelerometer_WATER_accX 3
Accelerometer_SUGAR_accX 3
Accelerometer_BREAD_accX 3
Accelerometer_CUP_accX 3
Accelerometer_MILK_accX 3
"""

class Opportunity(BaseDataset):
    """
    Opportunityデータセットに記録されているセンサデータとメタデータを読み込む．

    Parameters
    ----------
    path: Path
        Opportunity(UCI)データセットのパス．
        "dataset"ディレクトリの親ディレクトリを指定する．

    Attributes
    ----------
    NOT_SUPPORTED_LABELS: List[str]
        サポートしていないラベルのリスト

    X_LABELS: List[str]
        ターゲット以外のすべてのラベルのリスト

    SUPPORTED_Y_LABELS: List[str]
        ターゲットラベルのリスト
    """

    NOT_SUPPORTED_LABELS = [
        'Accelerometer_CHEESE_accX',
        'Accelerometer_SPOON_accX',
        'Accelerometer_KNIFE1_accX',
        'Accelerometer_KNIFE2_accX',
        'Accelerometer_PLATE_accX',
        'Accelerometer_GLASS_accX',
        'Accelerometer_SALAMI_accX',
        'Accelerometer_WATER_accX',
        'Accelerometer_SUGAR_accX',
        'Accelerometer_BREAD_accX',
        'Accelerometer_CUP_accX',
        'Accelerometer_MILK_accX',
    ]

    X_LABELS = tuple(set(Column) - set(['Locomotion', 'subject', 'HL_Activity', 'LL_Left_Arm', 'LL_Left_Arm_Object', 'LL_Right_Arm', 'LL_Right_Arm_Object', 'ML_Both_Arms']))
    SUPPORTED_Y_LABELS = ('Locomotion', 'subject', 'HL_Activity', 'LL_Left_Arm', 'LL_Left_Arm_Object', 'LL_Right_Arm', 'LL_Right_Arm_Object', 'ML_Both_Arms')

    def __init__(self, path:Path):
        super().__init__(path)
        self.data_cache = None
    
    def _load(self):
        if self.data_cache is None:
            data, meta = load(self.path)
            segments = [seg.join(m) for seg, m in zip(data, meta)]
            self.data_cache = segments
        else:
            segments = self.data_cache
        return segments
    
    def load(self, window_size:int, stride:int, x_labels:Optional[list]=None, y_labels:Optional[list]=None, ftrim_sec:int=2, btrim_sec:int=2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Opportunity(UCI)データセットを読み込み，sliding-window処理を行ったデータを返す．
        ここではADLのみをサポートしている．

        Parameters
        ----------
        window_size: int
            フレーム分けするサンプルサイズ

        stride: int
            ウィンドウの移動幅

        x_labels: Optional[list]
            入力(従属変数)のラベルリスト(ラベル名は元データセットに準拠)
            ここで指定したラベルのデータが入力として取り出される．

            一部サポートしていないラベルがあることに注意．

        y_labels: Optional[list]
            ターゲットのラベルリスト(使用方法はx_labelsと同様)．

        ftrim_sec: int
            セグメント先頭のトリミングサイズ(単位は秒)

        btrim_sec: int
            セグメント末尾のトリミングサイズ(単位は秒)

        Returns
        -------
        (x_frames, y_frames): Tuple[np.ndarray, np.ndarray]
            sliding-windowで切り出した入力とターゲットのフレームリスト

            x_framesは3次元配列で構造は大まかに(Batch, Channels, Frame)のようになっている．
            Channelsはx_labelsで指定したものが格納される．

            y_framesは2次元配列で構造は大まかに(Batch, Labels)のようになっている．
            Labelsはy_labelsで指定したものが格納される．

            y_framesはデータセット内の値をそのまま返すため，分類で用いる際はラベルの再割り当てが必要となることに注意する．
        
        Examples
        --------
        >>> opportunity_path = Path('path/to/dataset')
        >>> opportunity = Opportunity(opportunity_path)
        >>>
        >>> x_labels = [
        >>>     'Accelerometer_RKN^_accX',
        >>>     'Accelerometer_RKN^_accY',
        >>>     'Accelerometer_RKN^_accZ',
        >>>     'Accelerometer_HIP_accX',
        >>>     'Accelerometer_HIP_accY',
        >>>     'Accelerometer_HIP_accZ',
        >>> ]
        >>> y_labels = ['Locomotion', 'subject']    # 基本行動と被験者をターゲットラベルとして取り出す
        >>>
        >>> x, y = opportunity.load(window_size=256, stride=256, x_labels=xlabels, y_labels=ylabels, ftrim_sec=2, btrim_sec=2)
        >>> print(f'x: {x.shape}, y: {y.shape}')
        >>>
        >>> # > x: (?, 6, 256), y: (?, 2)
        """

        if x_labels is None:
            x_labels = list(set(Opportunity.X_LABELS) - set(Opportunity.NOT_SUPPORTED_LABELS))
        if y_labels is None:
            y_labels = list(Opportunity.SUPPORTED_Y_LABELS)

        if not set(Opportunity.NOT_SUPPORTED_LABELS).isdisjoint(set(x_labels+y_labels)):
            raise ValueError('x_labels or y_labels include non supported labels')

        if not(set(x_labels) <= set(Opportunity.X_LABELS)):
            raise ValueError('unsupported x labels is included: {}'.format(
                tuple(set(x_labels) - set(Opportunity.X_LABELS).intersection(set(x_labels)))
            ))
        if not(set(y_labels) <= set(Opportunity.SUPPORTED_Y_LABELS)):
            raise ValueError('unsupported y labels is included: {}'.format(
                tuple(set(y_labels) - set(Opportunity.SUPPORTED_Y_LABELS).intersection(set(y_labels)))
            ))

        segments = self._load()
        segments = [seg[x_labels+y_labels+['Locomotion']] for seg in segments]
        frames = []
        for seg in segments:
            fs = split_using_sliding_window(
                np.array(seg), window_size=window_size, stride=stride,
                ftrim=Sampling_Rate*ftrim_sec, btrim=Sampling_Rate*btrim_sec,
                return_error_value=None)
            if fs is not None:
                frames += [fs]
        frames = np.concatenate(frames)
        assert frames.shape[-1] == len(x_labels) + len(y_labels) + 1, 'Extracted data shape does not match with the number of total labels'
        x_frames = np.float64(frames[..., :len(x_labels)]).transpose(0, 2, 1)
        y_frames = np.int32(frames[..., 0, len(x_labels):])

        # remove data which activity label is 0
        flgs = y_frames[:, -1] != 0
        x_frames = x_frames[flgs]
        y_frames = y_frames[flgs][:, :-1]

        return x_frames, y_frames


def load(path:Union[Path,str]) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """Function for loading Opportunity dataset

    Parameters
    ----------
    path: Union[Path, str]
        Directory path of Opportunity(UCI) dataset, which is parent directory of "dataset" directory.

    Returns
    -------
    data, meta: List[pd.DataFrame], List[pd.DataFrame]
        Sensor data segmented by activity(Locomotion) and subject.

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
    """Function for loading raw data of Opportunity dataset

    Parameters
    ----------
    path: Path
        Directory path of Opportunity(UCI) dataset, which is parent directory of "dataset" directory.

    Returns
    -------
    chunks: List[pd.DataFrame]
        Raw data of Opportunity dataset.

        Each item in 'chunks' is a part of dataset, which is splited by subject.
    """

    path = path / 'dataset'
    #segs = defaultdict(list)
    chunks = []
    for p_id, person in enumerate(PERSONS):
        datfiles = path.glob('{}-ADL*.dat'.format(person))
        for datfile in datfiles:
            print(datfile)
            df = pd.read_csv(datfile, sep=r'\s+', header=None)
            df.columns = Column
            df['subject'] = p_id + 1
            # 将来的には欠損値処理はもう少しきちんと行う必要がある
            df = df.fillna(method='ffill')  # NANは周辺の平均値で埋める
            dtypes = dict(zip(Column, list(np.float64 for _ in Column)))
            dtypes['Locomotion'] = np.int32
            dtypes['subject'] = np.int32
            dtypes['HL_Activity'] = np.int32
            dtypes['LL_Left_Arm'] = np.int32
            dtypes['LL_Left_Arm_Object'] = np.int32
            dtypes['LL_Right_Arm'] = np.int32
            dtypes['LL_Right_Arm_Object'] = np.int32
            dtypes['ML_Both_Arms'] = np.int32
            df = df.astype(dtypes)

            chunks.append(df)

    return chunks


def reformat(raw) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """Function for reformating

    Parameters
    ----------
    raw:
        data loaded by 'load_raw'.
    
    Returns
    -------
    data, meta: List[pd.DataFrame], List[pd.DataFrame]
        Sensor data segmented by activity(Locomotion) and subject.

    See Alos
    --------
    The order of 'data' and 'meta' correspond.

    e.g. meta[0] is meta data of data[0].
    """

    chunks = raw
    segs = []
    for chunk in chunks:
        sub_segs = split_using_target(np.array(chunk), np.array(chunk['Locomotion']))
        sub_segs = list(itertools.chain(*[sub_segs[k] for k in sub_segs.keys()]))  # 連結
        sub_segs = list(map(lambda x: pd.DataFrame(x, columns=chunk.columns), sub_segs))
        sub_segs = list(map(lambda x: pd.DataFrame(x, columns=chunk.columns).astype(chunk.dtypes.to_dict()), sub_segs))
        # For debug
        for seg in sub_segs:
            label = seg['Locomotion'].iloc[0]
            if not np.array(seg['Locomotion'] == label).all():
                raise RuntimeError('This is bug. Failed segmentation')
        segs += sub_segs

    cols_meta = ['Locomotion', 'HL_Activity', 'LL_Left_Arm', 'LL_Left_Arm_Object', 'LL_Right_Arm', 'LL_Right_Arm_Object', 'ML_Both_Arms', 'subject']
    cols_sensor = list(set(Column) - set(cols_meta))
    data = list(map(lambda seg: seg[cols_sensor], segs))
    meta = list(map(lambda seg: seg[cols_meta], segs))
    
    return data, meta

