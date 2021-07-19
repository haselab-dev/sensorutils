"""
統計的な機能。
最大値や中央値、平均などは numpy にある。
連続で適用させたい場合は numpy.apply_along_axis 関数を使えば良い。
"""

import typing

import numpy as np


#def autocorrelation(data:np.ndarray, k:int):
#    """ラグが k の自己相関を求める。
#    時系列データ `$S = \{s_1, s_2, \dots, s_n\}$' に対して、ラグ `$k$' として
#    ```math
#    平均 \mu = \frac{1}{n} \sum_{i=1}^n s_i \\
#    自己相関 = \frac{\sum_{i=k+1}^{n} (s_i - \mu) (s_{i-k} - \mu)}{\sum_{i=1}^n (s_i - \mu)^2}
#    ```
#    分子が共分散なので 1 / (n - k) が正しいのでは？（要調査）
#
#    Parameters
#    ----------
#    data: np.ndarray
#        一次元データ
#    k: int
#        ラグ
#    """
#    avg = np.mean(data)
#    covariance = [(data[i] - avg) * (data[i-k] - avg) for i in range(k, len(data))]
#    covariance = np.sum(covariance)
#    denominator = [(data[i] - avg)**2 for i in range(len(data))]
#    denominator = np.sum(denominator)
#    return covariance / denominator
def autocorrelation(data:np.ndarray, k:int) -> np.ndarray:
    """ラグがkの自己相関を求める．
    時系列データ$S = \{s_1, s_2, \dots, s_n\}$に対して，ラグ$k$として

    $$
    \\frac{\mathrm{Cov}[s_i, s_{i-k}]}{\sqrt{\mathrm{Var}[s_i]\mathrm{Var}[s_{i-k}]}}
    $$

    Parameters
    ----------
    data: np.ndarray
        一次元データ
    k: int
        ラグ

    Returns
    -------
    : np.ndarray
        ラグkの自己相関
    """
    x1 = data[k:]
    x2 = data[:k]
    cov = np.cov(x1, x2)[0][1]
    v1 = np.var(x1)
    v2 = np.var(x2)
    return cov / np.sqrt(v1 * v2)



def correlation_rate(data:dict) -> float:
    """相関比の計算を行う．

    カテゴリ$c$におけるサンプル数$N_c$，平均値$\mu_c$，i番目の要素$x_{ci}$，また全てのカテゴリを含めた平均値$\mu$として，相関比は次式で計算される．

    $$
    \\frac{\sum_C N_c (\mu_c - \mu)^2}{\sum_C \sum_i^{N_c} (x_{ci} - \mu_c)^2 + \sum_C N_c (\mu_c - \mu)^2}
    $$


    Parameters
    ----------
    data: dict
        key = category

        val = List[src,...]

    Returns
    -------
    : float
        相関比
    """
    all_mean = np.mean(np.concatenate(list(data.values())))
    means = list()
    dsss = list()   # 偏差平方和
    n_samples = list()
    for vals in data.values():
        vals = np.array(vals)
        len_vals = len(vals)
        means.append(np.mean(vals))
        n_samples.append(len_vals)
        dsss.append(len_vals * np.var(vals))
    wcf = np.sum(dsss)
    vbc = np.sum([n_samples[i] * (means[i] - all_mean)**2 for i in range(len(n_samples))])
    return vbc / (wcf + vbc)


def cv(frame:np.ndarray, axis:typing.Optional[int]=None) -> float:
    """変動係数の計算

    Parameters
    ----------
    frame: np.ndarray
        計算対象のデータ

    axis: Optional[int]
        計算対象とする軸

    Returns
    -------
    float:
        変動係数
    """
    return np.sqrt(np.var(frame, axis=axis)) / np.mean(frame, axis=axis)


def abs_mean(x:np.ndarray, axis:typing.Optional[int]=None) -> typing.Union[float,np.ndarray]:
    """大きさの平均

    Parameters
    ----------
    x: np.ndarray
        計算対象

    axis: Optional[int]
        対象の軸

    Returns
    -------
    Union[float, np.ndarray]:
        計算結果
    """
    return np.mean(np.abs(x), axis=axis)


def abs_max(x:np.ndarray, axis:typing.Optional[int]=None) -> typing.Union[float,np.ndarray]:
    """大きさの最大値

    Parameters
    ----------
    x: np.ndarray
        計算対象

    axis: Optional[int]
        対象の軸

    Returns
    -------
    Union[float, np.ndarray]:
        計算結果
    """
    return np.max(np.abs(x), axis=axis)


def abs_min(x:np.ndarray, axis:typing.Optional[int]=None) -> typing.Union[float,np.ndarray]:
    """大きさの最小値

    Parameters
    ----------
    x: np.ndarray
        計算対象

    axis: Optional[int]
        対象の軸

    Returns
    -------
    Union[float, np.ndarray]:
        計算結果
    """
    return np.min(np.abs(x), axis=axis)


def abs_std(x:np.ndarray, axis:typing.Optional[int]=None) -> typing.Union[float,np.ndarray]:
    """大きさの標準偏差

    Parameters
    ----------
    x: np.ndarray
        計算対象

    axis: Optional[int]
        対象の軸

    Returns
    -------
    Union[float, np.ndarray]:
        計算結果
    """
    return np.std(np.abs(x), axis=axis)


def intensity(x:np.ndarray, axis:int=-1) -> typing.Union[float,np.ndarray]:
    """動きの激しさ

    $$
    \frac{1}{n-1}\sum_{i=1}^{n-1}|x_i - x_{i+1}|
    $$

    Parameters
    ----------
    x: np.ndarray
        計算対象

    axis: int
        対象の軸

    Returns
    -------
    Union[float, np.ndarray]:
        計算結果
    """
    src = np.abs(np.diff(x, axis=axis))  # n-1 される
    n = src.shape[axis]
    return np.sum(src, axis=axis) / n


def zcr(x:np.ndarray, axis:int=-1) -> typing.Union[float,np.ndarray]:
    """平均値の出現を考慮しないゼロ交差率

    Parameters
    ----------
    x: np.ndarray
        計算対象

    axis: int
        対象の軸

    Returns
    -------
    Union[float, np.ndarray]:
        計算結果
    """
    src = x - np.mean(x, axis=axis, keepdims=True)
    num = np.count_nonzero(np.diff(np.sign(src), axis=axis), axis=axis)
    return num / x.shape[axis]