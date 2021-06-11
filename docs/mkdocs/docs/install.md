# Installation

sensorutilsはpipからインストールすることで利用できるようになります。

## Requirements

- python 3.x
- numpy
- pandas
- scipy


## Pip

最新版は下記のコマンドでインストールすることが可能です。

以前のバージョンは[こちら](install_prev_ver.md)を参照

```sh
pip install git+https://github.com/haselab-dev/sensorutils
```

**玄人向け**

特定のブランチやタグを指定してインストールする場合

```sh
pip install git+https://github.com/haselab-dev/sensorutils@{ブランチ名 | タグ名}
```

## Conda

Feature working...

!!! Warning
    現在のバージョン(v0.12.0)では一応condaを通じたインストールが可能だが、
    場合によっては環境が壊れることがあるため非推奨

## Uninstall

パッケージを削除する場合



### Pip

```sh
pip uninstall sensorutils
```
