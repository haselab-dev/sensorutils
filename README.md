# sensorutils

sensorutilsは行動認識データセットのロードや前処理の実装を提供するライブラリです．

リファレンスサイトは[こちら](https://haselab-dev.github.io/sensorutils/mkdocs/site/)です．

## Project Structure

* datasets  : データセットのロード関数
* core      : センサデータ処理関数（主に `preprocessing`関連の処理）
* stats     : 統計的処理
    <!-- * 可能なら `pandas` の `goupby` や `rolling` 関数を用いたほうが良い -->
* metrics   : センサデータの評価関数
    <!-- * 可能なら `sklearn.metrics` を使用したほうが良い -->
* docs       : ドキュメントファイルを格納
* tests     : テストコードを格納
    <!-- * `python -m unittest discover tests` で実行できるようにしておく？ -->

## Usage

`sensorutils` をインポートすると `core.py` と `dataset` が自動的に読み込まれます．
(インポートしたときに `doc` と `test` が補完に表示されるがインポートする意味はありません．)

```python
import sensorutils
import sensorutils.datasets
```

sensorutilsの代表的な使用方法は[こちら](https://haselab-dev.github.io/sensorutils/mkdocs/site/guide.html)が参考になります．
具体的なコード例は[サンプル](docs/samples)を参照してください．


## Requirement

**共通**
* python >= 3.7
* numpy >= 1.19
* pandas >= 1.2
* scipy >= 1.6

## Installation

sensorutilsはpipを利用してインストールすることができます．
condaを用いてインストールすることもできますが，
pipを用いる方法が最も簡単です．
condaを用いたインストール方法は
[リファレンスサイト](https://haselab-dev.github.io/sensorutils/mkdocs/site/install.html)か
下記の`Build package`を参照してください．


### pip

```bash
pip install git+https://github.com/haselab-dev/sensorutils
# pip install -e[--editable] ... # DEVELOPER MODE
```

ブランチやタグを指定することもできます。
```bash
pip install git+https://github.com/haselab-dev/sensorutils[@{ブランチ名 | タグ名}]
```

## Build package

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

## Test

テストコードは実装されていますが，現状は完全なものではありません．

## Loadmap

* 1.0
   * [x] create abstruct class for dataset loader
   * [x] data argumentation
   * [ ] pytest
* 2.0（signal utils に向ける？）
   * [ ] pipeline
   * [ ] 前処理の拡充（適当）
