import numpy as np
import gigrnd
from scipy import random

def rinvgauss(mu, lam):
    """ Random variates from inverse Gaussian distribution """
    # Gordon Smyth
    # Created 15 Jan 1998.  Last revised 27 Feb 2017.
    # code obtained from statmod R package 12/30/2021 by DJS

    u = random.uniform()
    Z = random.randn(1) ** 2
    phi = lam / mu
    y1 = 1 - 0.5 * (np.sqrt(Z ** 2 + 4 * phi * Z) - Z) / phi
    if ((1 + y1) * u > 1):
        tmp = 1 / y1
    else:
        tmp = y1

    ans = mu * tmp
    if (mu <= 0):
        ans = np.NaN
    if lam <= 0:
        ans = np.NaN
    return ans


# control_tpbn = {"iters": 5000,
#                "a": 1.0,
#                "b": 0.5,
#                "tau_init": 1,
#                "tau_max": 100000,
#                "phi": 1}


def gibbs_tpbn_block(nsubj, bhat, xwx_list, block_start, block_stop, control_tpbn):
    # hypers for Strawderman-Berger prior: a=1,   b=.5
    # hypers for Horseshoe prior:          a=.5,  b=.5
    # hypers for midway   prior:           a=.75, b=.5

    a_hyper = control_tpbn["a"]
    b_hyper = control_tpbn["b"]
    tau_init = control_tpbn["tau_init"]
    phi = control_tpbn["phi"]
    iters = control_tpbn["iters"]
    tau_max = control_tpbn["tau_max"]

    burnin = iters // 2
    thin = 5
    n_pst = (iters - burnin) / thin

    nbeta = bhat.shape[0]
    beta = np.zeros(nbeta)
    tau = np.ones(nbeta) * tau_init
    gamma = np.zeros(nbeta)
    beta_mean = np.zeros(nbeta)

    nblock = len(block_start)

    for it in range(iters):

        if it % 100 == 0:
            print('--- iter-' + str(it) + ' ---', flush=True)

        for blk in range(nblock):
            start = block_start[blk]
            stop = block_stop[blk]

            # update beta
            update_beta_block(start, stop, xwx_list[blk], nsubj, tau, beta, bhat)

            if (it > burnin) and (it % thin == 0):
                beta_mean[start:stop] += beta[start:stop] / n_pst

            # update gamma
            gamma[start:stop] = np.random.gamma(a_hyper + b_hyper, 1.0 / (tau[start:stop] + phi))

            # update tau
            tau[start:stop] = gigrnd.gigrnd_vec(a_hyper - 0.5, 2.0 * gamma[start:stop], nsubj * beta[start:stop] ** 2)

            # bound tau
            tau[start:stop][tau[start:stop] > tau_max] = tau_max

    return beta_mean


def update_beta_block(start, stop, xwx, nsubj, tau, beta, bhat):
    prec = xwx.copy()
    tau_subset = tau[start:stop]

    for j in range(xwx.shape[0]):
        prec[j, j] += nsubj / tau_subset[j]

    sigma = np.linalg.inv(prec)
    chol = np.linalg.cholesky(sigma)
    mu = np.dot(np.dot(sigma, xwx), bhat[start:stop])
    z = np.random.normal(loc=0.0, scale=1.0, size=len(mu))
    beta[start:stop] = np.dot(chol, z) + mu

    return
