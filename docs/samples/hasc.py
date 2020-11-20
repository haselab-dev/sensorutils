import numpy as np
from pathlib import Path
from sensorutils.datasets.hasc import HASC

# About HASC
## http://hasc.jp/

window_size = 256   # sliding-windowでセグメンテーションするときのwindow幅
stride = 256        # stride幅

hasc_path = Path('E:/<any where>/HASC-PAC2016/')  # HASCデータセットパス
# hascローダインスタンスの作成
# 第2引数はメタデータのキャッシュ先パス
hasc = HASC(hasc_path, Path('hasc.csv'))


# [Case 1]
# サンプリングレートが100Hzのデータを取り出す
## 前方トリミングサイズ: 2s, 後方トリミングサイズ: 2s
## act2idは行動の種類と数字ラベルの対応表
print('[Case 1]')
queries = {'Frequency': 'Frequency == 100'}
print(f'Queries:\n{queries}')
x, y, act2id = hasc.load(window_size, stride, queries=queries, ftrim=2*100, btrim=2*100)
print(f'x: {x.shape}')
print(f'y: {y.shape}')


# [Case 2]
# サンプリングレートが100Hzかつ被験者の体重が80kg以下かつ被験者の身長が140cmより大きいデータを取り出す
## 前方トリミングサイズ: 5s, 後方トリミングサイズ: 5s
print('[Case 2]')
queries = {
    'Frequency': 'Frequency == 100', # サンプリングレートが100Hzのデータのみを取得
    'Weight': 'Weight <= 80',   # 体重が80g以下の人
    'Height': 'Height > 140',   # 身長が140cmより大きい人
}
print(f'Queries:\n{queries}')
x, y, act2id = hasc.load(window_size, stride, queries=queries, ftrim=5*100, btrim=5*100)
print(f'x: {x.shape}')
print(f'y: {y.shape}')


# [Case 3]
# 指定した被験者(persons)のデータを取り出す
print('[Case 3]')
persons = ['person01068', 'person03053', 'person02033', 'person01106', 'person03079', 'person02007']
queries = {
    'Person': f'Person in {persons}',
}
print(f'Queries:\n{queries}')
x, y, act2id = hasc.load(window_size, stride, queries=queries)
print(f'x: {x.shape}')
print(f'y: {y.shape}')

