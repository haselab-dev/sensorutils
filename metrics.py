"""
センサデータに関する評価関数

追加予定
SNR,..
"""

import typing

import numpy as np


def snr(dst:np.ndarray, src:np.ndarray, axis:typing.Optional[int]=None) -> typing.Union[float, np.ndarray]:
    """to calc Signal to Noise Ratio.

    Parameters
    ----------
    dst: np.ndarray
        clean data

    src: np.ndarray
        with noise

    axis: Optional[int], default=None
        mean axis

    Returns
    -------
    Union[float, np.ndarray]:
        [dB]
    """
    noise_mse = (np.square(dst - src)).mean(axis=axis)
    signal_ms = (np.square(dst)).mean(axis=axis)
    return 20 * np.log10((signal_ms / noise_mse))