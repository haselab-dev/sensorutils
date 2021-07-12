"""信号データに対しても拡張できるような transforms

`transforms.Compose` や `transforms.RandomApply`，`transforms.RandomOrder` などは `torchvision` を用いればよい．
"""

import typing
import torch
import torch.nn as nn


__all__ = [
    "Normalize",
    "RandomPermute",
    "RandomSignalFlip",
    "Jittering",
]


def normalize(tensor:torch.Tensor, mean:typing.List[float], std:typing.List[float], inplace:bool=False) -> torch.Tensor:
    """Normalize a float tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.
    See :class:`~torchvision.transforms.Normalize` for more details.
    Args:
        tensor (Tensor): Float tensor image of size (C, W) or (B, C, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.
    Returns:
        Tensor: Normalized Tensor image.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if not tensor.is_floating_point():
        raise TypeError('Input tensor should be a float tensor. Got {}.'.format(tensor.dtype))

    if tensor.ndim < 2:
        raise ValueError('Expected tensor to be a tensor image of size (..., C, W). Got tensor.size() = '
                         '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1)
    tensor.sub_(mean).div_(std)
    return tensor


class Normalize(nn.Module):
    """`torchvision.transforms.Normalize` の処理を信号に対しても行えるように変更した．

    Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Parameters
    ----------
    mean: List[float]
        Sequence of means for each channel.

    std: List[float]
        Sequence of standard deviations for each channel.

    inplace: Optional[bool]
        Bool to make this operation in-place.
    """
    def __init__(self, mean:typing.List[float], std:typing.List[float], inplace:bool=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, x):
        return normalize(x, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class RandomPermute(nn.Module):
    """一括でランダムに軸を入れ替える．

    入力するシェープ: (N, num_channels, length)

    Parameters
    ----------
    num_channels: int
        軸数
    """
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, x):
        if x.ndim == 3:
            raise TypeError("expected shape: (N, channel, length), len(x.shape)={}".format(x.ndim))
        idx = torch.randperm(self.num_channels)
        x = x[:, idx, :]
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomSignalFlip(nn.Module):
    """`p` の確率で入力値の符号を入れ替える．

    Parameters
    ----------
    p: float
        符号を反転させる確率
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if torch.rand(1) < self.p:
            x = x * -1
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Jittering(nn.Module):
    """ノイズを加える処理

    Parameters
    ----------
    mean: float
        ノイズの平均値

    std: float
        ノイズの標準偏差
    """
    def __init__(self, mean:float, std:float):
        super().__init__()
        self.mean = mean
        self.std = std
        self.m = torch.distributions.normal.Normal(mean, std)

    def forward(self, x):
        s = self.m.sample(sample_shape=x.size()[1:]).to(x.device)
        x += s
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(mean={}, std={})'.format(self.mean, self.std)
