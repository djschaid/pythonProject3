# should set num threads before doing import numpy (apparently numpy only checks for this at import)

import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

import numpy as np
import timeit
import h5py
import gibbs_models as gm

# Define input parameters

# define I/O files

infile_name = 'data/logistic_ridge.hdf5'
outfile_name = 'data/gibbs_block.out'

# define bayes control parameters

control_tpbn = {"iters": 1,
                "a": 1.0,
                "b": 0.5,
                "tau_init": 1,
                "tau_max": 100000,
                "phi": 1}

# Begin Computaions

print("load data from ", infile_name, flush=True)
dat = h5py.File(infile_name, 'r')

# the following lines with [:] at the end forces hdf5 to load data into arrays.
# without the [:], the objects would be pointers and need to then later
# load data into arrays, such as by slicing. The versions below with [:] are
# inteded for testing, and later will need to use without [:] for large
# arrays. Removing [:] will force more reads from disk, slowing performance

bhat = dat['dat']['bhat'][:]

# coef_code has values:
#   0 for alpha (local ancestry PCs)
#   1 for delta (dose of risk allele),
#   2 for interaction coefficients

# this version without [:] holds pointer to data
coef_code = dat['dat']['coef_code'][:]

# note need [...] to force read and convert to int
nsubj = dat['dat']['nsubj'][...]

# below creating list of np.arrays. If arrays are large, then
# this is efficient approach to use list with pointers to arrays
# because loading arrays with contiguous memory is efficient when
# working on large arrays. However, if arrays are small, then using
# pointers to arrays will result in higher rate of cache misses because
# the data in arrays might not be loaded at the time of computation
# in CPUs. An alternative approach would be to attempt to load all data
# into arrays and use indexing to pull in slices of data

nblock = len(dat['dat/block'])

xwx_list = [np.array(dat['dat/block/index_' + str(blk)]['xwx']) for blk in range(nblock)]
block_start = [np.array(dat['dat/block/index_' + str(blk)]['start']) for blk in range(nblock)]
block_stop = [np.array(dat['dat/block/index_' + str(blk)]['stop']) for blk in range(nblock)]

print("end data load", flush=True)

print("begin gibbs_tpbn_block", flush=True)
print("control iters = ", control_tpbn['iters'], flush=True)
starttime = timeit.default_timer()

beta_mean = gm.gibbs_tpbn_block(nsubj, bhat, xwx_list, block_start, block_stop, control_tpbn)

print("total time:", timeit.default_timer() - starttime)

with open(outfile_name, "w") as f:
    f.write('%s\t%s\t%s\n' % ('bhat', 'beta', 'type'))
    n_beta = bhat.shape[0]
    for ib in range(n_beta):
        f.write('%.6e\t%.6e\t%0.e\n' % (bhat[ib], beta_mean[ib], coef_code[ib]))
    f.close()

# important to close pointer to hdf5 dat AFTER no longer needed.
# If close before attempting to use dat items, get error

dat.close()
