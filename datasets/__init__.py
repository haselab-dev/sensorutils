"""データセットのロード関数。

ロード関数は確実に存在する。
読み込むだけの関数なのでどこかにキャッシュし、読み込み処理を省くと高速化が図れる。

クラスに関しては実装者依存。

対応データセット
* HASC
* Opportunity
* PAMAP2
* UCIHAR
"""

from . import (
    hasc,
    opportunity,
    pamap2,
    ucihar,
)
