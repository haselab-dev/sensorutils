"""
データセットのロード関数。

読み込むだけの関数なのでどこかにキャッシュし、読み込み処理を省くと高速化が図れる。

対応データセット
* [ ] HASC
* [ ] Opportunity
* [ ] PAMAP2
"""

from . import (
    hasc,
    opportunity,
)