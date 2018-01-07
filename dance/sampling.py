import numpy as np


def subsample_nth(arr, downsample_size):
    downsample_size = int(downsample_size)
    assert downsample_size > 0
    return arr[::downsample_size]


def subsample_mean(arr, downsample_size):
    downsample_size = int(downsample_size)
    assert downsample_size > 0
    if downsample_size > 1:
        return np.mean(arr.reshape(-1, downsample_size, arr.shape[1]), axis=1)
    else:
        return arr
