# sensorutils

センサデータのロード関数やその他便利な関数群の共有リポジトリ。

こうしたほうが使いやすくない等の相談は issues や slack、直接言うなど気軽に行いたい。

## CAUTION

`core.py` を破壊的変更したため `dataset` 等のアップデートが必要！

## ファイル構成について

特に指定はないけどジャンル別にすると使用するときに使いやすいかも。
各ファイルに関して長くなったら分割、ファイル分けなどを行うこと。

* datasets  : データセットのロード関数
* core      : センサデータ処理関数（主に `preprocessing` を担当。肥大化したら適宜分離）
* stats     : 統計的処理
    * 可能なら `pandas` の `goupby` や `rolling` 関数を用いたほうが良い
* metrics   : センサデータの評価関数
    * 可能なら `sklearn.metrics` を使用したほうが良い
* doc       : ドキュメントファイルを格納
* tests     : テストコードを格納
    * `python -m unittest discover tests` で実行できるようにしておく？

詳細は `docs` ディレクトリへ

## usage

`sensorutils` をインポートすると `core.py` と `dataset` が自動的に読み込まれる。

インポートしたときに `doc` と `test` が補完に表示されるがインポートする意味はない。

```python
import sensorutils
import sensorutils.datasets
```

* [sensorutils.datasetの使い方](doc/samples)

詳細は `docs` ディレクトリへ

## requirement

**共通**
* python 3
    * 3.8 以外でも動くように設計しているが、エラーが出たら教えてください
* numpy
* pandas
* scipy

## install

### pip

```bash
pip install git+https://github.com/haselab-dev/sensorutils
```

ブランチやタグを指定することもできます。
```bash
pip install git+https://github.com/haselab-dev/sensorutils[@{ブランチ名 | タグ名}]
```
