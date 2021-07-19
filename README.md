# sensorutils

センサデータのロード関数やその他便利な関数群の共有リポジトリ。

こうしたほうが使いやすくない等の相談は issues や slack、直接言うなど気軽に行いたい。

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
* python >= 3.7
* numpy >= 1.20
* pandas >= 1.2
* scipy >= 1.6

## install

### pip

```bash
pip install git+https://github.com/haselab-dev/sensorutils
# pip install -e[--editable] ... # DEVELOPER MODE
```

ブランチやタグを指定することもできます。
```bash
pip install git+https://github.com/haselab-dev/sensorutils[@{ブランチ名 | タグ名}]
```

## test

```bash
# cd sensorutils
python tests/test_core.py
```

## build package

### pip

```bash
# pip install --upgrade pip setuptools
python setup.py sdist

pip install dist/***.tar.gz
```

```bash
# pip install --upgrade pip setuptools wheel
python setup.py bdist_wheel

pip install dist/***.whl
```

掃除

```bash
python setup.py clean --all
```

### conda

<details>
<summary>仮想環境が壊れる可能性あり，注意を守って使うこと</summary>
<span style="color: red;">注意：`conda-build`を`base`環境以外にインストールすると`conda-build`がインストールされた環境のパスがおかしくなる可能性があり</span>

`conda-build`を`base`環境にインストールすればその他の環境でも使えるうえ，おそらくパスのバグは起きない．

#### with recipe

conda packageはビルド用のrecipeを用意して作成する．

```bash
git clone <sensorutils url>
cd sensorutils

conda activate base 
# conda install conda-build # or conda install -n base conda-build
conda build recipe

conda install --use-local sensorutils
# conda install -c file:///C:/Users/{name}/Miniconda3/conda-bld sensorutils
```
Windowsだと `%USERPROFILE%\Miniconda3\conda-bld` 配下にファイルができる.
以下のコマンドで確認することも可能．
```
conda build recipe --output
```

#### without recipe (未検証)

```bash
python setup.py bdist_conda
```

掃除

```bash
conda build purge # not delete conda package
conda build purge-all # delete conda package
```

</details>

## Uninstall package

### pip

```bash
pip uninstall sensorutils
```

### conda

```bash
conda uninstall sensorutils
```

## loadmap

* 1.0
   * [x] create abstruct class for dataset loader
   * [x] data argumentation
   * [ ] pytest
* 2.0（signal utils に向ける？）
   * [ ] pipeline
   * [ ] 前処理の拡充（適当）
