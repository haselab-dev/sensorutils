# sensorutils

センサデータのロード関数やその他便利な関数群の共有リポジトリ。

## ファイル構成について

特に指定はないけどジャンル別にすると使用するときに使いやすいかも。
各ファイルに関して長くなったら分割、ファイル分けなどを行うこと。

* datasets  : データセットのロード関数
* stats     : 統計的処理
* core      : センサデータ処理関数（適宜分離）
* metrics   : センサデータの評価関数

## usage

```python
import sensorutils
import sensorutils.datasets
```

## requirement

**共通**
* python 3
    * 3.8 以外でも動くように設計しているが、エラーが出たら教えてください
* numpy
* pandas
* scipy