"""
sensor utils。
とりあえずインポートしたら使える。
便利な関数や分類しにくい関数をとりあえず入れておく。

分離できそうなら分離すること。
"""

import typing

import numpy as np

def framing(segment:np.ndarray, **options) -> np.ndarray:
    """segment をフレーム分けする。

    各シェープは以下のようになる。

    segment: (segment_size, ch)

    frames: (num_frames, window_size, ch)

    ch はいくらでも対応しているため、ラベルを ch に追加しておけばラベルもフレーム分けできる。

    Parameters
    ----------
    segments: np.ndarray
        分割対象のデータ。
        len(segment.shape) == 2 であること。

    window_size: int, default=512
        フレーム分けするサンプルサイズ

    stride: int, default=None
        None は window_size となる。

    ftrim: int, default=5
        最初の {ftrim} サンプルをとばす

    btrim: int, default=5
        最後の {btrim} サンプルをとばす

    return_error_value: None
        エラーの時の返り値

    Returns
    -------
    frames: np.ndarray
        フレーム分けした結果。失敗したら None を返す
    """

    # 引数処理
    assert len(segment.shape) == 2, "Segment's shape is (segment_size, ch). This segment shape is {}".format(segment.shape)
    window_size = options.pop('window_size', 512)
    stride = options.pop('stride', None)
    ftrim = options.pop('ftrim', 5)
    btrim = options.pop('btrim', 5)
    return_error_value = options.pop('return_error_value', None)
    assert not bool(options), "args error: key {} is not exist.".format(list(options.keys()))
    # segment が短いときの処理
    if len(segment) < ftrim + btrim:
        return return_error_value
    seg = segment[ftrim: btrim].copy()
    if len(seg) < window_size:
        return return_error_value
    # 分割処理
    if stride is None:
        ch = segment.shape[1]
        num_frames = len(seg) // window_size
        ret = seg[:(num_frames * window_size)]
        return ret.reshape(-1, window_size, ch)
    else:
        num_frames = (len(seg) - window_size) // stride + 1
        idx = np.arange(window_size).reshape(-1, window_size).repeat(num_frames, axis=0) + np.arange(num_frames).reshape(num_frames, 1) * stride
        return seg[idx]


def split_from_target(src:np.ndarray, target:np.ndarray) -> np.ndarray:
    """target のデータを元に src の分割を行う。

    ```python
    tgt = np.array([0, 0, 1, 1, 2, 2, 1])
    src = np.array([1, 2, 3, 4, 5, 6, 7])
    assert split_from_target(src, tgt) == {0: [np.array([1, 2]), np.array([7])], 1: [np.array([3, 4])], 2: [np.array([5, 6])]}
    ```

    Parameters
    ----------
    src: np.ndarray
        分割するデータ

    target: np.ndarray
        ラベルデータ（一次元配列）

    Returns
    -------
    dict:
        key はラベル、value はデータのリスト。
    """
    from collections import defaultdict

    rshifted = np.roll(target, 1)
    diff = target - rshifted
    diff[0] = 1
    idxes = np.where(diff != 0)[0]

    ret = defaultdict(list)
    for i in range(1, len(idxes)):
        ret[target[idxes[i-1]]].append(src[idxes[i-1]:idxes[i]].copy())
    return dict(ret)