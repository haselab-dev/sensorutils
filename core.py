"""
sensor utils。
とりあえずインポートしたら使える。
便利な関数や分類しにくい関数をとりあえず入れておく。

分離できそうなら分離すること。
"""

import pickle
import numpy as np
import scipy
import scipy.interpolate
import typing


def split_by_sliding_window(segment:np.ndarray, **options) -> np.ndarray:
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
    assert type(window_size) is int, "type(window_size) is int: {}".format(type(window_size))
    assert ftrim >= 0 and btrim >= 0, "ftrim >= 0 and btrim >= 0: ftrim={}, btrim={}".format(ftrim, btrim)
    if type(segment) is not np.ndarray:
        return return_error_value
    # segment が短いときの処理
    if len(segment) < ftrim + btrim:
        return return_error_value
    if btrim == 0:
        seg = segment[ftrim:].copy()
    else:
        seg = segment[ftrim: -btrim].copy()
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


def window(src: np.ndarray, window_size: int, stride: int):
    """
    np.ndarray をフレーム分けするプリミティブな実装。
    stride が window_size 以外のとき split_by_sliding_window 関数より速く分割を行う。
    また、src のシェープはどのような次元数でも行える。

    Parameters
    ----------
    src: np.ndarray
        splited source.

    window_size: int
        sliding window size.

    stride: int,
        stride is int more than 0.

    Returns
    -------
    frames: np.ndarray
        a shape of frames is `(num_frames, window_size, *src.shape[1:])`, where num_frames is `(src.shape[0] - window_size) // stride + 1`.
    """
    assert stride > 0
    num_frames = (src.shape[0] - window_size) // stride + 1
    if stride == window_size:
        ret = src[:(num_frames * window_size)]
        return ret.reshape(-1, window_size, *src.shape[1:])
    else:
        ret_shape = (num_frames, window_size, *src.shape[1:])
        strides = (stride * src.strides[0], *src.strides)
        return np.lib.stride_tricks.as_strided(src, shape=ret_shape, strides=strides)


def split_from_target(src:np.ndarray, target:np.ndarray) -> typing.Dict[int, typing.List[np.ndarray]]:
    """target のデータを元に src の分割を行う。

    ```python
    tgt = np.array([0, 0, 1, 1, 2, 2, 1])
    src = np.array([1, 2, 3, 4, 5, 6, 7])
    assert split_from_target(src, tgt) == {0: [np.array([1, 2])], 1: [np.array([3, 4]), np.array([7])], 2: [np.array([5, 6])]}
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

    # idxes = np.append(idxes, len(target)) # 最後の部分を含めるため

    ret = defaultdict(list)
    for i in range(1, len(idxes)):
        ret[target[idxes[i-1]]].append(src[idxes[i-1]:idxes[i]].copy())
    ret[target[idxes[-1]]].append(src[idxes[-1]:].copy()) # 最後の部分を含めるため
    return dict(ret)


def interpolate(src:np.ndarray, rate:int, kind:str='linear', axis:int=-1) -> np.ndarray:
    """interpolation function.
    (use scipy.interpolate.interp1d)

    example: (linear interpolation)
    ```text
    [0, 2, 4] - x2 -> [0, 1, 2, 3, 4]
    [0, 3, 6] - x3 -> [0, 1, 2, 3, 4, 5, 6]
    ```

    Parameters
    ----------
    src: np.ndarray
        interpolation source.

    rate: int
        rate.

    kind: str, default='linear'
        Specifies the kind of interpolation as a string (‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, ‘next’, where ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of zeroth, first, second or third order; ‘previous’ and ‘next’ simply return the previous or next value of the point) or as an integer specifying the order of the spline interpolator to use. Default is ‘linear’.

    axis: int
        Specifies the axis of y along which to interpolate. Interpolation defaults to the last axis of y.

    Returns
    -------
    np.ndarray:
        shape[axis] - 1 == rate * ft.shape[axis]
    """
    N = src.shape[axis]
    x_low = np.linspace(0, 1, N)
    x_target = np.linspace(0, 1, N + (N-1) * (rate-1))
    f = scipy.interpolate.interp1d(x_low, src, kind=kind, axis=axis)
    return f(x_target)


def pickle_dump(obj:typing.Any, path:typing.Union[str,pathlib.Path]) -> None:
    """object dump using pickle.

    Parameters
    ----------
    obj: Any
        any object.

    path: Union[str, pathlib.Path]
        save path.
    """
    with open(path, mode='wb') as f:
        pickle.dump(obj, f)
    return


def pickle_load(path:pathlib.Path) -> typing.Any:
    """object load using pickle.

    Parameters
    ----------
    path: Union[str, pathlib.Path]
        a saved object pickle path.
    """
    with open(path, mode='rb') as f:
        data = pickle.load(f)
    return data
