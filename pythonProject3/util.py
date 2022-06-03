# utility functions

import numpy as np


def scale_cols(x):
    """ for each col of x, sum(x)=0, sum(x**2) = n """
    ncol = x.shape[1]
    mn = np.mean(x, axis=0)
    # note that python uses 1/n for sd, not 1/(n-1)
    sd = np.std(x, axis=0)
    for j in range(ncol):
        x[:, j] = (x[:, j] - mn[j])/sd[j]


def extremes(x):
    return np.min(x), np.max(x)


def anynan(x):
    test = np.sum(np.isnan(x)) > 0
    return test
