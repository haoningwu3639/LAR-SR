import torch
import torch.nn as nn
import torch.nn.functional as F


def unfold(x, scale, H, W):
    '''

    :param x: (N, C, H, W)
    :return:
    '''
    B = x.shape[0]
    x = F.unfold(x, kernel_size=scale, stride=scale, padding=0)
    assert x.shape[-1] == H * W
    x = x.reshape(B, -1, scale * scale, H, W)

    return x

def fold(x, scale, H, W):
    B = x.shape[0]
    x = x.reshape(B, -1, scale, scale, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3)
    x = x.reshape(B, -1, H * scale, W * scale)
    return x