"""
センサデータに関する評価関数。
numpy 実装。

* [x] Mean Absolute Error; MAE
* [x] Mean Absolute Persentage Error; MAPE
* [x] Mean Squared Error; MSE
* [x] Root Mean Squared Error; RMSE
* [x] Root Mean Squared Persentage Error; RMSPE
* [x] Root Mean Squared Logarithmic Error; RMSLE
* [x] R^2 (r2)
* [x] Signal to Noise Ratio; SNR
* [ ] Log Spectral Distance; LSD
"""

import typing

import numpy as np


def mae(true:np.ndarray, pred:np.ndarray, axis:typing.Optional[int]=None) -> typing.Union[float, np.ndarray]:
    """to calc Mean Absolute Error.

    ```math
    \frac{1}{N}\sum_{i=0}^{N} |\hat{y}_i - y_i|
    ```

    Parameters
    ----------
    true: np.ndarray
        true data.

    pred: np.ndarray
        predicted data.

    Returns
    -------
    Union[float, np.ndarray]:
    """
    return np.abs(true - pred).mean(axis=axis)


def mape(true:np.ndarray, pred:np.ndarray, axis:typing.Optional[int]=None) -> typing.Union[float, np.ndarray]:
    """to calc Mean Absolute Persentage Error.

    ```math
    \frac{100}{N}\sum_{i=0}^{N} \left| \frac{\hat{y}_i - y_i}{y_i} \right|
    ```

    Parameters
    ----------
    true: np.ndarray
        true data.

    pred: np.ndarray
        predicted data.

    Returns
    -------
    Union[float, np.ndarray]:
    """
    return mae(np.ones_like(true), pred / true, axis) * 100


def mse(true:np.ndarray, pred:np.ndarray, axis:typing.Optional[int]=None) -> typing.Union[float, np.ndarray]:
    """to calc Mean Squared Error.

    ```math
    \frac{1}{N}\sum_{i=0}^{N} (dst_i - src_i)^2
    ```

    Parameters
    ----------
    true: np.ndarray

    pred: np.ndarray

    Returns
    -------
    Union[float, np.ndarray]:
    """
    return (np.square(true - pred)).mean(axis=axis)


def rmse(true:np.ndarray, pred:np.ndarray, axis:typing.Optional[int]=None) -> typing.Union[float, np.ndarray]:
    """to calc Root Mean Squared Error.

    ```math
    \left(\frac{1}{N}\sum_{i=0}^{N} (\hat{y}_i - y_i)^2 \right)^{\frac{1}{2}}
    ```

    Parameters
    ----------
    true: np.ndarray

    pred: np.ndarray

    Returns
    -------
    Union[float, np.ndarray]:
    """
    return np.sqrt(mse(true, pred, axis))


def rmspe(true:np.ndarray, pred:np.ndarray, axis:typing.Optional[int]=None) -> typing.Union[float, np.ndarray]:
    """to calc Root Mean Squared Persentage Error.

    ```math
    100 \left(\frac{1}{N}\sum_{i=0}^{N} (\frac{\hat{y}_i - y_i}{y_i})^2 \right)^{\frac{1}{2}}
    ```

    Parameters
    ----------
    true: np.ndarray

    pred: np.ndarray

    Returns
    -------
    Union[float, np.ndarray]:
    """
    return rmse(np.ones_like(true), pred / true, axis) * 100


def rmsle(true:np.ndarray, pred:np.ndarray, axis:typing.Optional[int]=None) -> typing.Union[float, np.ndarray]:
    """to calc root mean squared logarithmic error.

    ```math
    \left(\frac{1}{N}\sum_{i=0}^{N} (\log (\hat{y}_i + 1) - \log (y_i + 1))^2 \right)^{\frac{1}{2}}
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
    Union[float, np.ndarray]:
    """
    return rmse(np.log(true + 1), np.log(pred + 1), axis=axis)


def r2(true:np.ndarray, pred:np.ndarray) -> float:
    """to calc r2 score.

    ```math
     {R^{2}}( \hat{y} ) := 1 - \frac{ \frac{1}{N} \sum_{i=1}^{N} { ( {y}_i - \hat{y}_{i} ) }^{2} }{ \frac{1}{N} \sum_{i=1}^{N} { ( {y}_i - \bar{y}) }^{2} } = 1 - \frac{M S E(\hat{y})}{Var(y)}
    ```

    Parameters
    ----------
    true: np.ndarray
        clean data

    pred: np.ndarray
        with noise

    Returns
    -------
    float
    """
    return 1 - (mse(true, pred) / np.var(true))


def snr(true:np.ndarray, pred:np.ndarray, axis:typing.Optional[int]=None) -> typing.Union[float, np.ndarray]:
    """to calc Signal to Noise Ratio.

    ```math
    10 \log_{10} \left(\frac{\sum_{i=0}^{N}true_i^2}{\sum_{i=0}^{N}(true_i - pred_i)^2} \right)
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
    Union[float, np.ndarray]:
        [dB]
    """
    assert true.shape == pred.shape, 'true.shape ({}) == pred.shape ({})'.format(true.shape, pred.shape)
    noise_mse = (np.square(true - pred)).sum(axis=axis)
    signal_ms = (np.square(true)).sum(axis=axis)
    return 10 * np.log10((signal_ms / noise_mse))


#def lsd(true:np.ndarray, pred:np.ndarray, axis:typeing.Optional[int]=None) -> typing.Union[float, np.ndarray]:
#    """to calc Log Spectral Distance.
#
#    ```math
#    ```
#    """
#    return np.mean(np.sqrt(np.mean(np.square(true - pred), axis=0)))
