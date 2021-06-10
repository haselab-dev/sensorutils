# Quickstart

## 1. とにかくHASCデータセットをロードしてみる

HASCデータセットを読み込んで、
ウィンドウ幅256、ストライド幅256でsliding-window処理したデータを取得する。

**手順**

1. HASCデータセットを取得する
2. 下記コードの`hasc_path`の箇所を取得したデータセットのパスに書き換える
3. (Optional) 16行目のキャッシュを作成するパスを変更

    デフォルトではスクリプトの実行ディレクトリ下に`hasc.csv`という名前でキャッシュが作成される。
4. 実行!


!!! Warning
    初回の実行ではメタデータの収集と統合の処理を行うため時間がかかる。

```python
import numpy as np
from pathlib import Path
from sensorutils.datasets.hasc import HASC

# About HASC
## http://hasc.jp/

window_size = 256   # sliding-windowでセグメンテーションするときのwindow幅
stride = 256        # stride幅

# HASCデータセットパス
hasc_path = Path('E:/<any where>/HASC-PAC2016/')

# hascローダインスタンスの作成
# 第2引数はメタデータのキャッシュ先パス
hasc = HASC(hasc_path, Path('hasc.csv'))


x, y, act2id = hasc.load(window_size, stride, ftrim=2*100, btrim=2*100)
print(f'x: {x.shape}')
print(f'y: {y.shape}')
```

## 2. HASCデータセットを被験者でフィルタリングして読み込む

手順は1.と同じ

```python
import numpy as np
from pathlib import Path
from sensorutils.datasets.hasc import HASC

# About HASC
## http://hasc.jp/

window_size = 256   # sliding-windowでセグメンテーションするときのwindow幅
stride = 256        # stride幅

# HASCデータセットパス
hasc_path = Path('E:/<any where>/HASC-PAC2016/')

# hascローダインスタンスの作成
# 第2引数はメタデータのキャッシュ先パス
hasc = HASC(hasc_path, Path('hasc.csv'))

# クエリの作成
# この5人の被験者のセンサデータのみを取得する
persons = ['person01068', 'person03053', 'person02033', 'person01106', 'person03079', 'person02007']
queries = {'Person': f'Person in {persons}'}

# 読み込み
x, y, act2id = hasc.load(window_size, stride, queries=queries, ftrim=2*100, btrim=2*100)
print(f'x: {x.shape}')
print(f'y: {y.shape}')
```
