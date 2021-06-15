# Installation

sensorutilsはpipからインストールすることで利用できるようになります。

## Requirements

- python 3.x
- numpy
- pandas
- scipy

---

## Pip

最新版は下記のコマンドでインストールすることが可能です．

以前のバージョンは[こちら](install_prev_ver.md)を参照してください．

```sh
pip install git+https://github.com/haselab-dev/sensorutils
```

**玄人向け**

特定のブランチやタグを指定してインストールする場合

```sh
pip install git+https://github.com/haselab-dev/sensorutils@{ブランチ名 | タグ名}
```

---

## Conda

!!! Note
    特にこだわりがなければpipでインストールすることをおすすめします．

!!! Warning
    以下の手順を守らない場合，環境が壊れる可能性があるため，十分に注意を払って作業を行ってください！

    環境が壊れても責任はとれません．

### conda-buildのインストール

conda用にパッケージをビルドするためのツールをインストールします．すでにインストールされている場合はこのステップは飛ばしてください．

`conda-build`のインストールは次のコマンドで実行可能ですが，**実行時には必ず`base`環境にインストールされることを確認してください**．`conda-build`が`base`以外の環境にインストールされるとconda環境が壊れる可能性があります．

```sh
conda install -n base conda-build
```

### パッケージのビルドとインストール

次にsensorutilsパッケージのビルドを行います．次のコマンドでは最新版のパッケージがビルド及びインストールされます．

他のバージョンをインストールしたい場合は2行目のリポジトリリンクでバージョンを指定して下さい([参考](install_prev_ver.html))．

```sh
cd <適当な作業ディレクトリ>
git clone https://github.com/haselab-dev/sensorutils.git
cd sensorutils

conda activate <インストールするconda環境>
conda build recipe

conda install --use-local sensorutils
# conda install -c file:///C:/Users/{name}/Miniconda3/conda-bld sensorutils
```

### 後始末

```sh
conda build purge # not delete conda package
conda build purge-all # delete conda package
```

---

## Uninstall

パッケージを削除する場合

### Pip

```sh
pip uninstall sensorutils
```

### Conda

```sh
conda uninstall sensorutils
```
