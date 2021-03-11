import numpy as np
from pathlib import Path
from sensorutils.datasets.pamap2 import PAMAP2

# About PAMAP2
## https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring


window_size = 256   # sliding-windowでセグメンテーションするときのwindow幅
stride = 256        # stride幅

pamap2_path = Path('E:/<any where>/Optional')  # PAMAP2データセットパス
pamap2 = PAMAP2(pamap2_path)    # pamap2ローダインスタンスの作成



# Case 1
## chest(腰)の3軸加速度データと対応する行動ラベルを取り出す
## 前方トリミングサイズ: 2s, 後方トリミングサイズ: 2s
## x_labelsとy_labelsで指定するラベルは下記のリストから選ぶ
## 返ってくる値はウィンドウで切り分けれらたセンサデータと対応するラベル
print('[Case 1]')
x_labels = ['IMU_chest_acc1_x', 'IMU_chest_acc1_y', 'IMU_chest_acc1_z']
y_labels = ['activity_id']
x, y = pamap2.load(window_size, stride, x_labels, y_labels, ftrim_sec=2, btrim_sec=2)
print(f'x: {x.shape}')
print(f'y: {y.shape}')


# Case 2
## chest(腰)とankel(くるぶし)の3軸加速度データと3軸ジャイロデータと対応する行動ラベルと被験者(id)を取り出す
## 前方トリミングサイズ: 2s, 後方トリミングサイズ: 2s
print('[Case 2]')
x_labels = [
    'IMU_chest_acc1_x', 'IMU_chest_acc1_y', 'IMU_chest_acc1_z',
    'IMU_chest_gyro_x', 'IMU_chest_gyro_y', 'IMU_chest_gyro_z',
    'IMU_ankle_acc1_x', 'IMU_ankle_acc1_y', 'IMU_ankle_acc1_z',
    'IMU_ankle_gyro_x', 'IMU_ankle_gyro_y', 'IMU_ankle_gyro_z',
]
y_labels = ['activity_id', 'person_id']
x, y = pamap2.load(window_size, stride, x_labels, y_labels, ftrim_sec=2, btrim_sec=2)
print(f'x: {x.shape}')
print(f'y: {y.shape}')


"""
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
"""
