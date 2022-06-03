# -*- coding: utf-8 -*-
"""
Dan Schaid
test git
"""

# should set num threads before doing import numpy (apparently numpy only checks for this at import)
# import os

# os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
# os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
# os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
# os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

import numpy as np
import timeit
import json as js
import util as ut
import logistic_ridge as lr
import h5py
from sklearn.decomposition import PCA

# define input parameters

# define I/O files
infile_name = "data/simsample.json"
outfile_name = "data/logistic_ridge.hdf5"

# define parameters

# size of XWX blocks to store in hdf5 for uses in gibbs sampling updates of betas
xwx_block_size = 1000

# No. PC's for alpha terms
npc = 20

# logistic_pen_ccd paramerters
lambda_pen = 0.001
max_iter = 2000

# begin computations

def make_block_start_stop(nbeta, block_size):
    if nbeta <= 2 * block_size:
        block_start = [0]
        block_stop = [nbeta]
        return block_start, block_stop

    nblock = nbeta // block_size

    block_start = list(range(0, nblock * block_size, block_size))
    block_stop = [ele + block_size for ele in block_start]
    remain = nbeta % block_size
    if remain > 0:
        block_stop[len(block_stop) - 1] = nbeta

    return block_start, block_stop


print("begin loading data", flush=True)
with open(infile_name, 'r') as f_obj:
    dat = js.load(f_obj)

# y = 1 for case, 0 for control
y = np.array(dat["y"])
nsubj = len(y)

xlan = np.array(dat["x"])  # design matrix for local ancestry terms
nlan = xlan.shape[1]

xcov = np.array(dat["xcov"])  # design matrix for adjusting covariates
ncov = xcov.shape[1]

# create design matrices

col_index = np.array(range(nlan)).astype(int)

print("end load data", flush=True)

# center/scale x's in-place
ut.scale_cols(xlan)
ut.scale_cols(xcov)

coef_type = np.array(dat["coef_type"])

alpha_index = col_index[coef_type == "a"]
delta_index = col_index[coef_type == "d"]
eta_index = col_index[coef_type == "i"]

pca = PCA(n_components=npc)
x_alpha = pca.fit_transform(xlan[:, alpha_index])
# center/scale PC's for alpha terms
ut.scale_cols(x_alpha)

nr = x_alpha.shape[0]
ndelta = delta_index.shape[0]
neta = eta_index.shape[0]

# design matrix for penalized terms (PC's for alpha, delta, eta)
npen = npc + ndelta + neta
xpen = np.zeros((nr, npen))

xpen[:, 0:npc] = x_alpha

# interweave delta, eta as pairs (delta1, eta1, delta2, eta2, ...))
index1 = npc + np.arange(0, 2 * ndelta, 2)
index2 = index1 + 1
xpen[:, index1] = xlan[:, delta_index]
xpen[:, index2] = xlan[:, eta_index]

# coef codes
# 0 (alpha, local ancestry PCs)
# 1 (delta, dose of risk allele)
# 2 (eta, interaction of local ancestry with risk allele)

coef_code = np.zeros((npc + ndelta + neta))
coef_code[index1] = 1
coef_code[index2] = 2

print("start logistic_ridge_ccd", flush=True)
starttime = timeit.default_timer()
beta0, beta_cov, beta_pen, lnlike, it, v = lr.logistic_ridge_ccd(y, xcov, xpen, lambda_pen, max_iter, eps=0.00001)
mytime = (timeit.default_timer() - starttime)
print("time for logistic_ridge_ccd: ", mytime, flush=True)


print("start writing hdf5", flush=True)
starttime = timeit.default_timer()
with h5py.File(outfile_name, 'w') as hf:
    # write results to hdf5 file
    group_name = 'dat'
    dat = hf.create_group('dat')

    hf['dat'].create_dataset('coef_code', data=coef_code)

    # adjusting covariate effects
    hf['dat'].create_dataset('bhat_cov', data=beta_cov)

    # bhat for local ancestry terms
    hf['dat'].create_dataset('bhat', data=beta_pen)

    hf['dat'].create_dataset('nsubj', data=nsubj)

    print("begin writing xwx blocks")

    # create and write xwx for each block
    nbeta = len(beta_pen)

    block_start, block_stop = make_block_start_stop(nbeta, xwx_block_size)
    nblock = len(block_start)

    hf['dat'].create_group('block')

    for blk in range(nblock):
        start = block_start[blk]
        stop = block_stop[blk]

        # create xwx for block
        xw = xpen[:, start:stop]
        for j in range(xw.shape[1]):
            xw[:, j] *= np.sqrt(v)

        xwx = np.dot(xw.T, xw)

        # now write xwx to hdf5

        block_subset = 'dat/block/index_' + str(blk)

        hf.create_group(block_subset)
        hf[block_subset].create_dataset('xwx', data=xwx)
        hf[block_subset].create_dataset('start', data=start)
        hf[block_subset].create_dataset('stop', data=stop)

    # after process all blocks, wrap up
    hf.close()

mytime = (timeit.default_timer() - starttime)
print("time to write hdf5: ", mytime, flush=True)
