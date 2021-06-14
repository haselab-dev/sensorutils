# User Guide

このページではsensorutilsの代表的なツールの利用方法を紹介しています．

## sensorutils.coreの利用

### ターゲットラベルをもとにデータを分割する

行動認識データセットでは，すべてのデータが一つのCSVファイルにまとめられていることがあり，このデータから行動ごとのセンサデータを取り出すには行動ラベルをもとに配列を分割する必要があります．

sensorutilsでは`split_using_target`関数を用いることでラベルをもとにした配列の分割を高速に実行することが可能です．次のコードは`split_using_target`関数の簡単な利用例です．

```python
import numpy as np
from sensorutils.core import split_using_target

target = np.array([0, 0, 1, 1, 2, 2, 1])    # ターゲットラベル
source = np.array([1, 2, 3, 4, 5, 6, 7])    # 分割する配列
splited = split_from_target(src, tgt)
print(splited)

# splited == {
#    0: [np.array([1, 2])],
#    1: [np.array([3, 4]), np.array([7])],
#    2: [np.array([5, 6])]
# }
```

### Numpy配列に対してsliding-window処理を行う

機械学習モデルや深層学習モデルへの入力は多くの場合一定のサイズである必要がありますが，センサデータは可変長の時系列データであるため，そのままではモデルへ入力することができません．そこでセンサを用いた行動認識では，可変長の時系列データからスライディングウィンドウを用いて一定サイズの部分系列を取り出すことでモデルへの入力を作成します．

sensorutilsでは`to_frames`関数を用いることで高速なsliding-window処理を簡単に実行することが可能です．次のコードは`to_frames`関数の使用例です．

```python
import numpy as np
from sensorutils.core import to_frames

source = np.array([1, 2, 3, 4, 5, 6, 7])    # 分割する配列
window_size = 3
stride = 2
frames = to_frames(source, window_size, stride)

# frames == np.array([
#     [1, 2, 3],
#     [3, 4, 5],
#     [5, 6, 7],
# ])
```

---

## データセットローダの利用

センサに関するオープンなベンチマークデータセットを利用するには，提供されているデータセットを読み込むコードを実装する必要があります．しかし，行動認識データセットなどのデータセットは画像系のデータセットと異なり，データセットによって提供される形が大きく異なるため，読み込みコードの実装は煩雑な作業となります．

sensorutilsでは，読み込みコードの実装の手間を削減するために代表的なセンサ系のベンチマークデータセットのローダを提供しています．これを活用することで，読み込みコードの実装の手間を減らし，より本質的な作業に注力することが可能です．

以下では実装が非常に厄介なHASCデータセットの読み込みをsensorutilsを利用して行います．

### HASCデータセットをロードしてみる

HASCデータセットを読み込んで、ウィンドウ幅256、ストライド幅256でsliding-window処理したデータを取得します．

**手順**

1. HASCデータセットを取得する．
2. 下記コードの`hasc_path`の箇所を取得したデータセットのパスに書き換える．
3. (Optional) 16行目のキャッシュを作成するパスを変更

    デフォルトではスクリプトの実行ディレクトリ下に`hasc.csv`という名前でキャッシュが作成される．

4. 実行!


!!! Warning
    初回の実行ではメタデータの収集と統合の処理を行うため時間がかかります．

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

### HASCデータセットを被験者でフィルタリングして読み込む

手順は"HASCデータセットをロードしてみる"と同様です．

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
# この6人の被験者のセンサデータのみを取得する
persons = ['person01068', 'person03053', 'person02033', 'person01106', 'person03079', 'person02007']
queries = {'Person': f'Person in {persons}'}

# 読み込み
x, y, act2id = hasc.load(window_size, stride, queries=queries, ftrim=2*100, btrim=2*100)
print(f'x: {x.shape}')
print(f'y: {y.shape}')
```
