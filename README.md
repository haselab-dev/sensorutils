# sensorutils

センサデータのロード関数やその他便利な関数群の共有リポジトリ。

こうしたほうが使いやすくない等の相談は issues や slack、直接言うなど気軽に行いたい。

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

## Git 関連

サブモジュールを扱うときのコマンド（適宜引数は変えること）

サブモジュールとして clone する方法。
```bash
# すでにリポジトリが存在し、サブモジュールとして追加する場合
git submodule add https://github.com/haselab-dev/sensorutils.git
# サブモジュールの clone をし忘れた場合
git submodule update --init --recursive
# サブモジュールと同時にリポジトリを clone する場合
git clone --recursive [リポジトリの url など]
```

サブモジュールの更新を受け取るとき
```bash
git submodule foreach git fetch
git submodule foreach git merge origin/master
# 上二つの代わりに下でも可能
git submodule update --remote --merge
```

サブモジュールの更新をした時
```bash
git submodule foreach git add .
git submodule foreach git commit -m "2 on parent"
git submodule foreach git push
# このようにしてサブモジュールのリモートを更新したのち
git add .
git commit -m "update child 2"
git push --recurse-submodules=check
# 元のリポジトリのリモートの更新もする（確認）
```

サブモジュールが `Detached HEAD` になったときの対処
```bash
git submodule foreach git status #HEADのリビジョン名(SHA-1)を調べる
git submodule foreach git checkout master
git submodule foreach git merge <HEADのSHA-1> #rebse
git submodule foreach git push
```