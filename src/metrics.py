import numpy as np


def rmse(a, b):
    diff = np.asarray(a) - np.asarray(b)
    return float(np.sqrt(np.mean(diff * diff)))


def position_rmse(truth, estimate):
    return rmse(truth[:, 0:3], estimate[:, 0:3])


def gps_rmse(truth, gps):
    return rmse(truth[:, 0:3], gps[:, 0:3])


def axis_rmse(truth, estimate):
    names = ["x", "y", "z"]
    out = {}
    for i, name in enumerate(names):
        out[name] = rmse(truth[:, i], estimate[:, i])
    return out
