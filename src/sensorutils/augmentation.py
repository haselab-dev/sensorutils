"""データ拡張ライブラリ

参考サイト
* https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
"""
import numpy as np


def jitter(x:np.ndarray, sigma:float=0.05) -> np.ndarray:
    """jittering

    Parameters
    ----------
    x: np.ndarray
        sensor data

        expected shape: (num_channels, length_of_sequence)

    sigma: float
        Noise scale

    Returns
    -------
    : np.ndarray
        sensor data with jitter.
    """
    noise = np.random.normal(loc=0, scale=sigma, size=x.shape)
    return x + noise


def scaling(x:np.ndarray, sigma:float=0.1) -> np.ndarray:
    """scaling

    Parameters
    ----------
    x: np.ndarray
        sensor data

        expected shape: (num_channels, length_of_sequence)

    sigma: float
        Standard deviation. Non-negative.

    Returns
    -------
    : np.ndarray
        scaled sensor data
    """
    scaling_factor = np.random.normal(loc=1.0, scale=sigma, size=(1, x.shape[1]))
    noise = np.matmul(np.ones((x.shape[0], 1)), scaling_factor)
    return x * noise


def swapping(x:np.ndarray) -> np.ndarray:
    """swapping
    
    Parameters
    ----------
    x: np.ndarray
        sensor data

        expected shape: (num_channels, length_of_sequence)

    Returns
    -------
    : np.ndarray
        sensor data with randomly swapped axes
    """
    idx = np.arange(x.shape[0]).reshape(-1, 3)
    idx = np.take_along_axis(idx, np.random.rand(*(idx.shape)).argsort(axis=1), axis=1).reshape(-1)
    return x[idx, :]


def flipping(x:np.ndarray, overall:bool=True) -> np.ndarray:
    """flipping

    Parameters
    ----------
    x: np.ndarray
        sensor data

        expected shape: (num_channels, length_of_sequence)
    
    overall: bool
        flag whether all axes should be flipped together.

    Returns
    -------
    : np.ndarray
        random flipped sensor data
    """
    if overall:
        x_new = ((-1) ** np.random.randint(0, 1)) * x
    else:
        x_new = np.zeros(x.shape)
        x_new[0, :] = ((-1) ** np.random.randint(0, 1)) * x[0, :]
        x_new[1, :] = ((-1) ** np.random.randint(0, 1)) * x[1, :]
        x_new[2, :] = ((-1) ** np.random.randint(0, 1)) * x[2, :]
    return x_new


def reversing(x:np.ndarray) -> np.ndarray:
    """reversing

    Parameters
    ----------
    x: np.ndarray
        sensor data

        expected shape: (num_channels, length_of_sequence)

    Returns
    -------
    : np.ndarray
        reversed sensor data
    """
    return np.fliplr(x)

