import numpy as np
from pathlib import Path
from sensorutils.datasets.opportunity import Opportunity

# About Opportunity
## https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition

window_size = 256   # sliding-windowでセグメンテーションするときのwindow幅
stride = 256        # stride幅

opportunity_path = Path('E:/<any where>/OpportunityUCIDataset') # Opportunity Datasetのパス
opportunity = Opportunity(opportunity_path)    # opportunityローダインスタンスの作成


# 使い方はPAMAP2と全く同じなのでそちらを参照されたし。

# x_labelsで指定したデータと対応する行動ラベルと被験者(id)を取り出す。
# 前方トリミングサイズ: 2s, 後方トリミングサイズ: 2s
# x_labelsとy_labelsで指定するラベルは下記のリストから選ぶ。
# 返ってくるのはx_labelsとy_labelsでしたデータをsliding-windowで切り分けたデータ
x_labels = [
    'Accelerometer_RKN^_accX',
    'Accelerometer_RKN^_accY',
    'Accelerometer_RKN^_accZ',
    'Accelerometer_HIP_accX',
    'Accelerometer_HIP_accY',
    'Accelerometer_HIP_accZ',
]
y_labels = ['Locomotion', 'User']
x, y = opportunity.load(window_size, stride, x_labels, y_labels, ftrim_sec=2, btrim_sec=2)
print(f'x: {x.shape}')
print(f'y: {y.shape}')


"""
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
    'Accelerometer_CUP_gyroX',
    'Accelerometer_CUP_gyroY',
    'Accelerometer_SALAMI_gyroX',
    'Accelerometer_SALAMI_gyroY',
    'Accelerometer_WATER_gyroX',
    'Accelerometer_WATER_gyroY',
    'Accelerometer_CHEESE_gyroX',
    'Accelerometer_CHEESE_gyroY',
    'Accelerometer_BREAD_gyroX',
    'Accelerometer_BREAD_gyroY',
    'Accelerometer_KNIFE1_gyroX',
    'Accelerometer_KNIFE1_gyroY',
    'Accelerometer_MILK_gyroX',
    'Accelerometer_MILK_gyroY',
    'Accelerometer_SPOON_gyroX',
    'Accelerometer_SPOON_gyroY',
    'Accelerometer_SUGAR_gyroX',
    'Accelerometer_SUGAR_gyroY',
    'Accelerometer_KNIFE2_gyroX',
    'Accelerometer_KNIFE2_gyroY',
    'Accelerometer_PLATE_gyroX',
    'Accelerometer_PLATE_gyroY',
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
"""
