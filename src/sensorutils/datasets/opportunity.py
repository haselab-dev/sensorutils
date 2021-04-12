"""
Opportunity データの読み込みなど
"""

import numpy as np
import pandas as pd
from pathlib import Path
from ..core import split_using_target, split_using_sliding_window
#from collections import defaultdict

from .base import BaseDataset

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
    """Opportunity

    Opportunity データセットの行動分類を行うためのローダクラス。

    Attributes
    ----------
    not_supported_labels: サポートしていないラベルの一覧。
    """

    not_supported_labels = [
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

    def __init__(self, path:Path):
        super().__init__(path)
    
    def act2id(self):
        global Locomotion
        return dict([(Locomotion[k], k) for k in Locomotion.keys()])
    
    def subject2id(self):
        global PERSONS
        subs = list(map(lambda x: str(x), PERSONS))
        return dict(zip(subs, list(range(len(subs)))))

    def load(self, window_size:int, stride:int, x_labels:list, y_labels:list, ftrim_sec:int, btrim_sec:int):
        """Opportunityの読み込み(ADL)とsliding-window

        Parameters
        ----------
        window_size: int
            フレーム分けするサンプルサイズ

        stride: int
            ウィンドウの移動幅

        x_labels: list
            入力(従属変数)のラベルリスト(ラベル名は元データセットに準拠。)。ここで指定したラベルのデータが入力として取り出される。
            一部サポートしていないラベルがあるの注意。

        y_labels: list
            ターゲットのラベルリスト。使用はx_labelsと同様。

        ftrim_sec: int
            セグメント先頭のトリミングサイズ。単位は秒。

        btrim_sec: int
            セグメント末尾のトリミングサイズ。単位は秒。

        Returns
        -------
        (x_frames, y_frames): tuple
            sliding-windowで切り出した入力とターゲットのフレームリスト
        """
        if not set(self.not_supported_labels).isdisjoint(set(x_labels+y_labels)):
            raise ValueError('x_labels or y_labels include non supported labels')
        segments = load(self.path)
        segments = [seg[x_labels+y_labels] for seg in segments]
        frames = []
        for seg in segments:
            fs = split_using_sliding_window(
                np.array(seg), window_size=window_size, stride=stride,
                ftrim=Sampling_Rate*ftrim_sec, btrim=Sampling_Rate*btrim_sec,
                return_error_value=None)
            if fs is not None:
                frames += [fs]
        frames = np.concatenate(frames)
        assert frames.shape[-1] == len(x_labels) + len(y_labels), 'Extracted data shape does not match with the number of total labels'
        x_frames = frames[..., :len(x_labels)]
        y_frames = frames[..., len(x_labels):]
        return x_frames, y_frames

def load(path:Path) -> dict:
    """Opportunity の読み込み

    Parameters
    ----------
    path: Path
        Opportunity(UCI)のdatasetディレクトリがあるディレクトリ

    Returns
    -------
    segments:
        Locomotionをもとにセグメンテーションされたデータ
    """
    import itertools
    path = path / 'dataset'
    #segs = defaultdict(list)
    chunks = []
    for p_id, person in enumerate(PERSONS):
        datfiles = path.glob('{}-ADL*.dat'.format(person))
        for datfile in datfiles:
            print(datfile)
            df = pd.read_csv(datfile, sep='\s+', header=None)
            df.columns = Column
            df['User'] = p_id
            # 将来的には欠損値処理はもう少しきちんと行う必要がある
            df = df.fillna(method='ffill')  # NANは周辺の平均値で埋める
            chunks.append(df)

    segs = []
    for chunk in chunks:
        sub_segs = split_using_target(np.array(chunk), np.array(chunk['Locomotion']))
        sub_segs = list(itertools.chain(*[sub_segs[k] for k in sub_segs.keys()]))  # 連結
        sub_segs = list(map(lambda x: pd.DataFrame(x, columns=chunk.columns), sub_segs))
        # For debug
        for seg in sub_segs:
            label = seg['Locomotion'].iloc[0]
            if not np.array(seg['Locomotion'] == label).all():
                raise RuntimeError('This is bug. Failed segmentation')
        segs += sub_segs

    return segs

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
PERSONS = [f'S{i+1}' for i in range(4)]
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
    #'User',     # ユーザID(ローダ側で付与)
]