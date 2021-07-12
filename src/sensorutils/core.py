"""
sensor utils
とりあえずインポートしたら使える．
便利な関数や分類しにくい関数をとりあえず入れておく．

分離できそうなら分離すること
"""

import pickle
import numpy as np
import scipy
import scipy.interpolate
import pathlib
import typing


def to_frames(src: np.ndarray, window_size: int, stride: int, stride_mode: str='index') -> np.ndarray:
    """
    np.ndarrayをフレーム分けするプリミティブな実装で，`stride_mode`で分割アルゴリズムを指定することが可能．

    `to_frames`関数は`to_frames_using_index`関数，`to_frames_using_nptricks`関数，`to_frames_using_reshape`関数を適応的に使い分ける．

    使い分けは以下の通り．

    - `window_size == stride` -> to_frames_using_reshape
    - `window_size != stride and stride_mode == 'index'` -> to_frames_using_index
    - `window_size != stride and stride_mode == 'nptrick'` -> to_frames_using_nptricks

    Parameters
    ----------
    src: np.ndarray
        splited source.

    window_size: int
        sliding window size.

    stride: int,
        stride is int more than 0.

    stride_mode: str
        'index' or 'nptrick'.
        
        it is used `to_frames_*` method when window_size != stride.

    Returns
    -------
    frames: np.ndarray
        a shape of frames is `(num_frames, window_size, *src.shape[1:])`, where num_frames is `(src.shape[0] - window_size) // stride + 1`.
    """
    assert stride > 0, 'ストライドは正の整数である必要がある. stride={}'.format(stride)
    assert stride_mode in ['index', 'nptrick'], "stride_mode is 'index' or 'nptrick'. stride_mode={}".format(stride_mode)
    if stride == window_size:
        return to_frames_using_reshape(src, window_size)
    elif stride_mode == 'index':
        return to_frames_using_index(src, window_size, stride)
    else:
        return to_frames_using_nptricks(src, window_size, stride)


def to_frames_using_reshape(src: np.ndarray, window_size: int) -> np.ndarray:
    """
    np.ndarrayをフレーム分けするプリミティブな実装で，ウィンドウサイズとストライド幅が同じ場合に利用することが可能．

    分割に`np.reshape`を使用しており，非常に高速なsliding-window処理を実行可能．

    Parameters
    ----------
    src: np.ndarray
        splited source.

    window_size: int
        sliding window size. stride = window_size.

    Returns
    -------
    frames: np.ndarray
        a shape of frames is `(num_frames, window_size, *src.shape[1:])`, where num_frames is `(src.shape[0] - window_size) // window_size + 1`.
    """
    num_frames = (src.shape[0] - window_size) // window_size + 1
    ret = src[:(num_frames * window_size)]
    return ret.reshape(-1, window_size, *src.shape[1:])


def to_frames_using_index(src: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """
    np.ndarray をフレーム分けするプリミティブな実装で，分割にindexingを使用している．

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
    assert stride > 0, 'ストライドは正の整数である必要がある. stride={}'.format(stride)
    num_frames = (len(src) - window_size) // stride + 1
    idx = np.arange(window_size).reshape(-1, window_size).repeat(num_frames, axis=0) + np.arange(num_frames).reshape(num_frames, 1) * stride
    return src[idx]


def to_frames_using_nptricks(src: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """
    np.ndarray をフレーム分けするプリミティブな実装で，分割に`np.lib.stride_tricks.as_strided`関数を使用しており，indexingを使用する`to_frames_using_index`より高速である．

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
    assert stride > 0, 'ストライドは正の整数である必要がある. stride={}'.format(stride)
    num_frames = (src.shape[0] - window_size) // stride + 1
    ret_shape = (num_frames, window_size, *src.shape[1:])
    strides = (stride * src.strides[0], *src.strides)
    return np.lib.stride_tricks.as_strided(src, shape=ret_shape, strides=strides)


def split_using_sliding_window(segment:np.ndarray, **options) -> np.ndarray:
    """
    可変サイズのsegmentからsliding-window方式で一定サイズのフレームを抽出する．
    各shapeは以下のようになる．

    - segment: (segment_size, ch)
    - frames: (num_frames, window_size, ch)

    segmentの第2軸(axis=1)以降のshapeは任意であり，
    例えばshapeが(segment_size, ch1, ch2)のデータをsegmentとして入力すると，
    (num_frames, window_size, ch1, ch2)のデータを取得することができる．

    Parameters
    ----------
    segments: np.ndarray
        分割対象のデータ

    window_size: int, default=512
        フレーム分けするサンプルサイズ

    stride: int, default=None
        strideがNoneはwindow_sizeが指定される．

    ftrim: int
        最初のftrimサンプルをとばす(default=5)．

    btrim: int
        最後のbtrimサンプルをとばす(default=5)．

    return_error_value: None
        エラーの時の返り値

    Returns
    -------
    frames: np.ndarray
        sliding-window方式で抽出下フレーム

        失敗したら`return_error_value`で指定された値を返す．
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
        stride = window_size
    return to_frames(seg, window_size, stride, stride_mode='index')


def split_using_target(src:np.ndarray, target:np.ndarray) -> typing.Dict[int, typing.List[np.ndarray]]:
    """
    targetのデータを元にsrcの分割を行う．

    Parameters
    ----------
    src: np.ndarray
        分割するデータ

    target: np.ndarray
        ラベルデータ(一次元配列)

    Returns
    -------
    dict:
        keyはラベル，valueはデータのリスト．
    
    Examples
    --------
    >>> tgt = np.array([0, 0, 1, 1, 2, 2, 1])
    >>> src = np.array([1, 2, 3, 4, 5, 6, 7])
    >>> splited = split_from_target(src, tgt)
    >>>
    >>> # splited == {
    >>> #    0: [np.array([1, 2])],
    >>> #    1: [np.array([3, 4]), np.array([7])],
    >>> #    2: [np.array([5, 6])]
    >>> # }
    """
    from collections import defaultdict

    rshifted = np.roll(target, 1)
    diff = target - rshifted
    diff[0] = 1
    idxes = np.where(diff != 0)[0]

    ret = defaultdict(list)
    for i in range(1, len(idxes)):
        ret[target[idxes[i-1]]].append(src[idxes[i-1]:idxes[i]].copy())
    ret[target[idxes[-1]]].append(src[idxes[-1]:].copy())
    return dict(ret)


def interpolate(src:np.ndarray, rate:int, kind:str='linear', axis:int=-1) -> np.ndarray:
    """
    interpolation function.

    Use `scipy.interpolate.interp1d` in this function.



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
    
    Examples
    --------
    Linear interpolation
    ```
    [0, 2, 4] - x2 -> [0, 1, 2, 3, 4]
    [0, 3, 6] - x3 -> [0, 1, 2, 3, 4, 5, 6]
    ```
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