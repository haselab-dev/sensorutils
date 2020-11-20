import numpy as np
from pathlib import Path
from sensorutils.datasets.ucihar import UCIHAR

# About UCIHAR Dataset
## UCIHARはあらかじめ訓練データと検証データが分けれらており、
## かつsliding-windowによるセグメンテーション処理も行われている。
## sliding-windowのwindow幅は128，stride幅は64である．
## ちなみにサンプリングレートは50Hz
## 詳細な情報は下記リンクを参照されたし。
## https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones


window_size = 256   # sliding-windowでセグメンテーションするときのwindow幅
stride = 256        # stride幅

ucihar_path = Path('E:/uci/har/path')  # uciharデータセットパス
ucihar_path = Path('E:/datasets/UCI_HAR_Dataset/UCI HAR Dataset')
ucihar = UCIHAR(ucihar_path)    # uciharローダインスタンスの作成


# 加速度センサデータと対応する行動ラベルと被験者ラベルを取り出す。
# 返ってくる値はウィンドウで切り分けれらたセンサデータと対応する行動ラベルと被験者(id)
# 被験者idを指定
# 被験者は全部で30人。idの範囲は1 - 30
person_list = [1, 2, 5, 7, 9]
# include_gravityで重力成分を含むかどうかを指定(今回は含める)
# trainで訓練データと検証データのどちらを取り出すか指定する(今回は訓練データ)
x, y, person_id = ucihar.load(train=True, person_list=person_list, include_gravity=True)
print(f'x: {x.shape}')
print(f'y: {y.shape}')
print(f'person_id: {person_id.shape}')

