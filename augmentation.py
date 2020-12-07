"""データ拡張ライブラリ

参考サイト
* https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
"""
import numpy as np
import random
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from scipy import signal


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


def generate_random_curve(x, sigma=0.2, knot=4):
    """generate random curve

    Parameters
    ----------
    x:

    sigma:

    knot:

    Returns
    -------
    """
    xx = (np.ones((x.shape[1], 1)) * (np.arange(0, x.shape[0], (x.shape[0] - 1) / (knot + 1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, x.shape[1]))
    x_range = np.arange(x.shape[0])
    cs_x = CubicSpline(xx[:, 0], yy[:, 0])
    cs_y = CubicSpline(xx[:, 1], yy[:, 1])
    cs_z = CubicSpline(xx[:, 2], yy[:, 2])
    return np.array([cs_x(x_range), cs_y(x_range), cs_z(x_range)]).transpose()


def mag_warp(x, sigma=0.2):
    """magnitude warping

    Parameters
    ----------
    x:

    sigma:

    Returns
    -------
    """
    return x * generate_random_curve(x, sigma)


def distort_timesteps(x, sigma=0.2):
    """distortion timesteps

    Parameters
    ----------
    x:

    sigma:

    Returns
    -------
    """
    tt = generate_random_curve(x, sigma)
    tt_cum = np.cumsum(tt, axis=0)
    t_scale = [(x.shape[0] - 1) / tt_cum[-1, 0],
               (x.shape[0] - 1) / tt_cum[-1, 1],
               (x.shape[0] - 1) / tt_cum[-1, 2]]

    tt_cum[:, 0] = tt_cum[:, 0] * t_scale[0]
    tt_cum[:, 1] = tt_cum[:, 1] * t_scale[1]
    tt_cum[:, 2] = tt_cum[:, 2] * t_scale[2]
    return tt_cum


def time_warp(x, sigma=0.2):
    """time warping

    Parameters
    ----------
    x:

    sigma:

    Returns
    -------
    """
    tt_new = distort_timesteps(x, sigma)
    x_new = np.zeros(x.shape)
    x_range = np.arange(x.shape[0])
    x_new[:, 0] = np.interp(x_range, tt_new[:, 0], x[:, 0])
    x_new[:, 1] = np.interp(x_range, tt_new[:, 1], x[:, 1])
    x_new[:, 2] = np.interp(x_range, tt_new[:, 2], x[:, 2])
    return x_new


def rotation(x):
    """rotation

    Parameters
    ----------
    x:

    Returns
    -------
    """
    axis = np.random.uniform(low=-1, high=1, size=x.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(x, axangle2mat(axis, angle))

def rotation_dev(x):
    """rotation. under developing?

    Parameters
    ----------
    x:

    Returns
    -------
    """
    axis = np.random.uniform(low=-1, high=1, size=x.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    quaterinion = np.array([
        np.cos(angle/2),
        axis[0]*np.sin(angle/2),
        axis[1]*np.sin(angle/2),
        axis[2]*np.sin(angle/2),
    ])
    print(f'axis: {axis}')
    print(f'angle: {angle}')
    print(f'quaterinion: {quaterinion}')
    rot_mat = Rotation.from_quat(quaterinion).as_matrix()
    return np.matmul(x, rot_mat), axis


def permutation(x, n_perm=4, min_seg_length=10):
    """permutation

    Parameters
    ----------
    x:

    n_perm:

    min_seg_length:

    Returns
    -------
    """
    x_new = np.zeros(x.shape)
    idx = np.random.permutation(n_perm)
    b_while = True
    while b_while:
        segs = np.zeros(n_perm + 1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(min_seg_length, x.shape[0] - min_seg_length, n_perm - 1))
        segs[-1] = x.shape[0]
        if np.min(segs[1:] - segs[0:-1]) > min_seg_length:
            b_while = False

    pp = 0
    for ii in range(n_perm):
        x_temp = x[segs[idx[ii]]:segs[idx[ii] + 1], :]
        x_new[pp:pp + len(x_temp), :] = x_temp
        pp += len(x_temp)

    return x_new


def rand_sample_timesteps(x, n_samples=100):
    """random sample timesteps

    Parameters
    ----------
    x:

    n_samples:

    Returns
    -------
    """
    x_new = np.zeros(x.shape)
    tt = np.zeros((n_samples, x.shape[1]), dtype=int)
    tt[1:-1, 0] = np.sort(np.random.randint(1, x.shape[0] - 1, n_samples - 2))
    tt[1:-1, 1] = np.sort(np.random.randint(1, x.shape[0] - 1, n_samples - 2))
    tt[1:-1, 2] = np.sort(np.random.randint(1, x.shape[0] - 1, n_samples - 2))
    tt[-1, :] = x.shape[0] - 1
    return x_new


def random_sampling(x, n_samples=100):
    """random sampling

    Parameters
    ----------
    x:

    n_samples:

    Returns
    -------
    """
    tt = rand_sample_timesteps(x, n_samples)
    x_new = np.zeros(x.shape)
    x_new[:, 0] = np.interp(np.arange(x.shape[0]), tt[:, 0], x[tt[:, 0], 0])
    x_new[:, 1] = np.interp(np.arange(x.shape[0]), tt[:, 1], x[tt[:, 1], 1])
    x_new[:, 2] = np.interp(np.arange(x.shape[0]), tt[:, 2], x[tt[:, 2], 2])
    return x_new


def flipping(x):
    """flipping

    Parameters
    ----------
    x:

    Returns
    -------
    """
    x_new = np.zeros(x.shape)
    x_new[:, 0] = ((-1) ** random.randint(0, 1)) * x[:, 0]
    x_new[:, 1] = ((-1) ** random.randint(0, 1)) * x[:, 1]
    x_new[:, 2] = ((-1) ** random.randint(0, 1)) * x[:, 2]
    return x_new


def reversing(x):
    """reversing

    Parameters
    ----------
    x:

    Returns
    -------
    """
    return np.flipud(x)


def low_pass_filter(x, cutoff=20, fs=50):
    """low pass filter

    Parameters
    ----------
    x:

    cutoff:

    fs:

    Returns
    -------
    """
    filter_ = signal.firwin(numtaps=cutoff // 2, cutoff=cutoff, fs=fs)
    x_new = np.zeros(x.shape)
    x_new[:, 0] = signal.lfilter(filter_, 1, x[:, 0])
    x_new[:, 1] = signal.lfilter(filter_, 1, x[:, 1])
    x_new[:, 2] = signal.lfilter(filter_, 1, x[:, 2])
    return x_new


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.arange(10*3).reshape([10, 3])
    print(x)
    print(x.shape)
    x_rot, rot_axis = rotation_dev(x)
    print(x_rot)
    print(x_rot.shape)

    rot_axis *= 10
    ax.plot([0, rot_axis[0]], [0, rot_axis[1]], [0, rot_axis[2]], color='black')
    # ax.scatter(*(rot_axis.reshape([1, -1]).T), color='green')
    ax.scatter(*(x.T), color='blue')
    ax.scatter(*(x_rot.T), color='red')

    plt.show()
