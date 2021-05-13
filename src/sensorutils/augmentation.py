"""データ拡張ライブラリ

参考サイト
* https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
"""
import numpy as np


def jitter(x, sigma=0.05):
    """jittering

    Parameters
    ----------
    x:

    sigma:

    Returns
    -------
    """
    noise = np.random.normal(loc=0, scale=sigma, size=x.shape)
    return x + noise


def scaling(x, sigma=0.1):
    """scaling

    Parameters
    ----------
    x:

    sigma:

    Returns
    -------
    """
    scaling_factor = np.random.normal(loc=1.0, scale=sigma, size=(1, x.shape[1]))
    noise = np.matmul(np.ones((x.shape[0], 1)), scaling_factor)
    return x * noise


def swapping(x):
    """swapping
    
    Parameters
    ----------
    x:

    Returns
    -------
    """
    idx = np.arange(x.shape[0]).reshape(-1, 3)
    idx = np.take_along_axis(idx, np.random.rand(*(idx.shape)).argsort(axis=1), axis=1).reshape(-1)
    return x[idx, :]


def flipping(x, overall=True):
    """flipping

    Parameters
    ----------
    x:

    Returns
    -------
    """
    if overall:
        x_new = ((-1) ** np.random.randint(0, 1)) * x
    else:
        x_new = np.zeros(x.shape)
        x_new[0, :] = ((-1) ** np.random.randint(0, 1)) * x[0, :]
        x_new[1, :] = ((-1) ** np.random.randint(0, 1)) * x[1, :]
        x_new[2, :] = ((-1) ** np.random.randint(0, 1)) * x[2, :]
    return x_new


def reversing(x):
    """reversing

    Parameters
    ----------
    x:

    Returns
    -------
    """
    return np.fliplr(x)

