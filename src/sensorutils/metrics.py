"""
センサデータに関する評価関数
numpy 実装
"""

import typing
import numpy as np


def mae(true:np.ndarray, pred:np.ndarray, axis:typing.Optional[int]=None) -> typing.Union[float, np.ndarray]:
    """Calc Mean Absolute Error.

    $$
    \\frac{1}{N}\sum_{i=0}^{N} |\hat{y}_i - y_i|
    $$

    ```python
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(2, 3, 4)
    mae(a, b).shape
    #>> ()
    mae(a, b, axis=2).shape
    #>> (2, 3)
    ```

    Parameters
    ----------
    true: np.ndarray
        true data.

    pred: np.ndarray
        predicted data.

    Returns
    -------
    :Union[float, np.ndarray]
        MAE
    """
    return np.abs(true - pred).mean(axis=axis)


def mape(true:np.ndarray, pred:np.ndarray, axis:typing.Optional[int]=None) -> typing.Union[float, np.ndarray]:
    """Calc Mean Absolute Persentage Error.

    $$
    \\frac{100}{N}\sum_{i=0}^{N} \\left| \\frac{\hat{y}_i - y_i}{y_i} \\right|
    $$

    ```python
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(2, 3, 4)
    mape(a, b).shape
    #>> ()
    mape(a, b, axis=2).shape
    #>> (2, 3)
    ```

    Parameters
    ----------
    true: np.ndarray
        true data.

    pred: np.ndarray
        predicted data.

    Returns
    -------
    :Union[float, np.ndarray]
        MAPE
    """
    return mae(np.ones_like(true), pred / true, axis) * 100


def mse(true:np.ndarray, pred:np.ndarray, axis:typing.Optional[int]=None) -> typing.Union[float, np.ndarray]:
    """Calc Mean Squared Error.

    $$
    \\frac{1}{N}\sum_{i=0}^{N} (dst_i - src_i)^2
    $$

    ```python
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(2, 3, 4)
    mse(a, b).shape
    #>> ()
    mse(a, b, axis=2).shape
    #>> (2, 3)
    ```

    Parameters
    ----------
    true: np.ndarray
        true data.

    pred: np.ndarray
        predicted data.

    Returns
    -------
    :Union[float, np.ndarray]
        MSE
    """
    return (np.square(true - pred)).mean(axis=axis)


def rmse(true:np.ndarray, pred:np.ndarray, axis:typing.Optional[int]=None) -> typing.Union[float, np.ndarray]:
    """Calc Root Mean Squared Error.

    $$
    \\left(\\frac{1}{N}\sum_{i=0}^{N} (\hat{y}_i - y_i)^2 \\right)^{\\frac{1}{2}}
    $$

    ```python
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(2, 3, 4)
    rmse(a, b).shape
    #>> ()
    rmse(a, b, axis=2).shape
    #>> (2, 3)
    ```

    Parameters
    ----------
    true: np.ndarray
        true data.

    pred: np.ndarray
        predicted data.

    Returns
    -------
    :Union[float, np.ndarray]
        RMSE
    """
    return np.sqrt(mse(true, pred, axis))


def rmspe(true:np.ndarray, pred:np.ndarray, axis:typing.Optional[int]=None) -> typing.Union[float, np.ndarray]:
    """Calc Root Mean Squared Persentage Error.

    $$
    100 \\left(\\frac{1}{N}\sum_{i=0}^{N} (\\frac{\hat{y}_i - y_i}{y_i})^2 \\right)^{\\frac{1}{2}}
    $$

    ```python
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(2, 3, 4)
    rmspe(a, b).shape
    #>> ()
    rmspe(a, b, axis=2).shape
    #>> (2, 3)
    ```

    Parameters
    ----------
    true: np.ndarray
        true data.

    pred: np.ndarray
        predicted data.

    Returns
    -------
    :Union[float, np.ndarray]
        RMSPE
    """
    return rmse(np.ones_like(true), pred / true, axis) * 100


def rmsle(true:np.ndarray, pred:np.ndarray, axis:typing.Optional[int]=None) -> typing.Union[float, np.ndarray]:
    """Calc root mean squared logarithmic error.

    $$
    \\left(\\frac{1}{N}\sum_{i=0}^{N} (\log (\hat{y}_i + 1) - \log (y_i + 1))^2 \\right)^{\\frac{1}{2}}
    $$

    ```python
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(2, 3, 4)
    rmsle(a, b).shape
    #>> ()
    rmsle(a, b, axis=2).shape
    #>> (2, 3)
    ```

    Parameters
    ----------
    true: np.ndarray
        clean data

    pred: np.ndarray
        with noise

    axis: Optional[int], default=None
        mean axis

    Returns
    -------
    :Union[float, np.ndarray]
        RMSLE
    """
    return rmse(np.log(true + 1), np.log(pred + 1), axis=axis)


def r2(true:np.ndarray, pred:np.ndarray) -> float:
    """Calc r2 score(coefficient of determination).

    $$
     {R^{2}}( \hat{y} ) := 1 - \\frac{ \\frac{1}{N} \sum_{i=1}^{N} { ( {y}_i - \hat{y}_{i} ) }^{2} }{ \\frac{1}{N} \sum_{i=1}^{N} { ( {y}_i - \\bar{y}) }^{2} } = 1 - \\frac{M S E(\hat{y})}{Var(y)}
    $$

    ```python
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(2, 3, 4)
    r2(a, b).shape
    #>> ()
    ```

    Parameters
    ----------
    true: np.ndarray
        clean data

    pred: np.ndarray
        with noise

    Returns
    -------
    :float
        coefficient of determination 
    """
    return 1 - (mse(true, pred) / np.var(true))


def snr(true:np.ndarray, pred:np.ndarray, axis:typing.Optional[int]=None) -> typing.Union[float, np.ndarray]:
    """Calc Signal to Noise Ratio.

    $$
    10 \log_{10} \\left(\\frac{\sum_{i=0}^{N}true_i^2}{\sum_{i=0}^{N}(true_i - pred_i)^2} \\right)
    $$

    ```python
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(2, 3, 4)
    snr(a, b).shape
    #>> ()
    snr(a, b, axis=2).shape
    #>> (2, 3)
    ```

    Parameters
    ----------
    true: np.ndarray
        clean data

    pred: np.ndarray
        with noise

    axis: Optional[int]
        mean axis (default=None)

    Returns
    -------
    :Union[float, np.ndarray]
        SNR
    """
    assert true.shape == pred.shape, 'true.shape ({}) == pred.shape ({})'.format(true.shape, pred.shape)
    noise_mse = (np.square(true - pred)).sum(axis=axis)
    signal_ms = (np.square(true)).sum(axis=axis)
    return 10 * np.log10((signal_ms / noise_mse))


def lsd(true_spec:np.ndarray, pred_spec:np.ndarray, axis:typing.Optional[int]=None) -> typing.Union[float, np.ndarray]:
    """Calc Log Spectral Distance.

    $$
    \mathrm{LSD}(S(\omega),\\tilde{S}(\omega)) =
    \sqrt{\\frac{1}{W}\sum_{\omega}^{W} \\left(20\log_{10}\\left|\\frac{S(\omega)}{\\tilde{S}(\omega)}\\right|\\right)^2}
    $$

    $S(\omega)$と$\\tilde{S}(\omega)$ は，それぞれ原波形と雑音抑圧波形の対数スペクトル．
    
    複数の短時間スペクトルの距離は各スペクトルで距離を算出した後，平均を取ること．

    ```python
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(2, 3, 4)
    lsd(a, b).shape
    #>> ()
    ```

    Parameters
    ----------
    true_spec: np.ndarray
        spectral 1

    pred_spec: np.ndarray
        spectral 2

    axis: Optional[int]
        Not use
    
    Returns
    -------
    : Union[float, np.ndarray]
        LSD
    """
    return np.sqrt(np.mean(20 * np.log10(np.abs(true_spec / (true_spec - pred_spec)))))

