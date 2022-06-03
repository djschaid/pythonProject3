# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 10:32:22 2022

@author: SCHAID
"""
import numpy as np
from scipy import stats


def logistic_ridge_v1(y, x, lambda_pen, max_iter=25, eps=0.001):
    """ Logistic ridge regression with penalty lambda_pen """
    n = x.shape[0]
    m = x.shape[1]
    # scale lambda_pen since xwx scales with n, as does lnlike
    lambda_n = lambda_pen * n
    # add col of 1's for intercept
    xf = np.zeros((n, m + 1))
    xf[:, 0] = 1.0
    xf[:, 1:m + 1] = x.copy()
    prev = y.mean()
    beta0 = np.log(prev / (1 - prev))
    beta = np.zeros(m + 1)
    beta[0] = beta0
    # xw = xf.copy()
    nx = m + 1
    imat = np.diag(np.ones(nx))
    imat[0, 0] = 0  ## don't penalize intercept
    p = np.exp(np.dot(xf, beta))
    p = p / (1 + p)
    lnlike_old = np.sum(np.log(p[y == 1])) + np.sum(np.log(1 - p[y == 0])) - .5 * lambda_n * np.sum(beta ** 2)
    lnlike_change = 10

    it = 0
    while (lnlike_change > eps) and (it < max_iter):
        it = it + 1
        v = p * (1 - p)
        xwx = np.dot(np.dot(xf.T, np.diag(v)), xf)
        vmat = xwx + lambda_n * imat
        Z = np.dot(xf, beta) + np.dot(np.diag(1 / v), (y - p))
        beta = np.dot(np.dot(np.linalg.solve(vmat, xf.T), np.diag(v)), Z)
        p = np.exp(np.dot(xf, beta))
        p = p / (1 + p)
        lnlike = np.sum(np.log(p[y == 1])) + np.sum(np.log(1 - p[y == 0])) - .5 * lambda_n * np.sum(beta ** 2)
        lnlike_change = np.abs(lnlike - lnlike_old) / (np.abs(lnlike_old) + 1.0)
        lnlike_old = lnlike

    return beta, lnlike, it, xwx


def logistic_ridge(y, x, lambda_pen, max_iter=25, eps=0.00001):
    """ Logistic ridge regression with penalty lambda_pen """
    # revised to allow for large n by not using diag(v)
    # where len(v) = n

    n = x.shape[0]
    m = x.shape[1]
    # scale lambda_pen since xwx scales with n, as does lnlike
    lambda_n = lambda_pen * n
    # add col of 1's for intercept
    xf = np.zeros((n, m + 1))
    xf[:, 0] = 1.0
    xf[:, 1:m + 1] = x.copy()
    prev = y.mean()
    beta0 = np.log(prev / (1 - prev))
    beta = np.zeros(m + 1)
    beta[0] = beta0
    nx = m + 1
    imat = np.diag(np.ones(nx))
    imat[0, 0] = 0  # don't penalize intercept
    p = np.exp(np.dot(xf, beta))
    p = p / (1 + p)
    lnlike_old = np.sum(np.log(p[y == 1])) + np.sum(np.log(1 - p[y == 0])) - .5 * lambda_n * np.sum(beta ** 2)
    lnlike_change = 10

    it = 0
    while (lnlike_change > eps) and (it < max_iter):
        it = it + 1
        v = p * (1 - p)
        xwx = xf.copy()
        for j in range(nx):
            xwx[:, j] = xf[:, j] * v
        xwx = np.dot(xf.T, xwx)
        vmat = xwx + lambda_n * imat
        Z = np.dot(xf, beta) + (y - p) / v
        beta = np.linalg.solve(vmat, np.dot(xf.T, v * Z))
        p = np.exp(np.dot(xf, beta))
        p = p / (1 + p)
        lnlike = np.sum(np.log(p[y == 1])) + np.sum(np.log(1 - p[y == 0])) - .5 * lambda_n * np.sum(beta ** 2)
        lnlike_change = np.abs(lnlike - lnlike_old) / (np.abs(lnlike_old) + 1.0)
        lnlike_old = lnlike

    return beta, lnlike, it, xwx


def logistic_ridge_ccd_V1(y, x, lambda_pen, max_iter=500, eps=0.00001):
    """ Logistic ridge regression fit by cyclic coordinate descent """

    n = x.shape[0]
    m = x.shape[1]
    # scale lambda_pen since xwx scales with n, as does lnlike
    lambda_n = lambda_pen * n
    # add col of 1's for intercept
    xf = np.zeros((n, m + 1))
    xf[:, 0] = 1.0
    xf[:, 1:m + 1] = x.copy()
    prev = y.mean()
    beta0 = np.log(prev / (1 - prev))
    beta = np.zeros(m + 1)
    beta[0] = beta0
    nx = m + 1

    z = np.dot(xf, beta)
    p = np.exp(z)
    p = p / (1 + p)
    v = p * (1 - p)
    lnlike_old = np.sum(np.log(p[y == 1])) + np.sum(np.log(1 - p[y == 0])) - .5 * lambda_n * np.sum(beta ** 2)
    lnlike_change = 10

    it = 0
    while (lnlike_change > eps) and (it < max_iter):
        it = it + 1
        print("iter = ", it, flush=True)

        for j in range(nx):
            # update beta[j]
            ssq = np.dot(xf[:, j] ** 2, v)
            ssq_pen = ssq + lambda_n
            sprod = np.dot((y - p), xf[:, j])
            b_old = beta[j]
            b_new = b_old * ssq / ssq_pen + sprod / ssq_pen
            beta[j] = b_new

            # update lin predictor for all subjects for covariate j
            z = z + (b_new - b_old) * xf[:, j]
            p = np.exp(z)
            p = p / (1 + p)
            v = p * (1 - p)

        # after sweep through all cols of x, check convergence

        lnlike = np.sum(np.log(p[y == 1])) + np.sum(np.log(1 - p[y == 0])) - .5 * lambda_n * np.sum(beta ** 2)
        lnlike_change = np.abs(lnlike - lnlike_old) / (np.abs(lnlike_old) + 1.0)
        lnlike_old = lnlike

    return beta, lnlike, it, v


def logistic_ridge_ccd(y, xcov, xpen, lambda_pen, max_iter=500, eps=0.00001):
    """ Logistic ridge regression fit by cyclic coordinate descent """

    # y = 1/0 for case/control
    # xcov = design matrix without intercept, for unpenalized adjusting covariates
    # xpen = design matrix for penalized terms

    if (xcov.shape[0] != xpen.shape[0]):
        print("error in logistic_ridge_ccd: number rows not equal for xcov and xpen")
        exit()

    n = xcov.shape[0]
    ncov = xcov.shape[1]
    npen = xpen.shape[1]

    # scale lambda_pen since xwx scales with n, as does lnlike
    lambda_n = lambda_pen * n

    prev = y.mean()
    beta0 = np.log(prev / (1 - prev))
    beta_cov = np.zeros(ncov)
    beta_pen = np.zeros(npen)

    z = np.ones(n) * beta0
    p = np.exp(z)
    p = p / (1 + p)
    v = p * (1 - p)
    lnlike_old = np.sum(np.log(p[y == 1])) + np.sum(np.log(1 - p[y == 0])) - .5 * lambda_n * np.sum(beta_pen ** 2)
    lnlike_change = 10

    it = 0
    while (lnlike_change > eps) and (it < max_iter):
        it = it + 1
        print("iter = ", it, flush=True)

        # update intercept beta0
        ssq = np.sum(v)
        sprod = np.sum((y - p))
        b_old = beta0
        b_new = b_old + sprod / ssq
        beta0 = b_new
        # update lin predictor for all subjects for intercept
        z = z + (b_new - b_old)
        p = np.exp(z)
        p = p / (1 + p)
        v = p * (1 - p)

        if ncov > 0:
            # update unpenalized adjusting covariates
            for j in range(ncov):
                # update beta[j]
                ssq = np.dot(xcov[:, j] ** 2, v)
                sprod = np.dot((y - p), xcov[:, j])
                b_old = beta_cov[j]
                b_new = b_old + sprod / ssq
                beta_cov[j] = b_new

                # update lin predictor for all subjects for covariate j
                z = z + (b_new - b_old) * xcov[:, j]
                p = np.exp(z)
                p = p / (1 + p)
                v = p * (1 - p)

        # update penalized terms
        if npen > 0:
            for j in range(npen):
                # update beta[j]
                ssq = np.dot(xpen[:, j] ** 2, v)
                ssq_pen = ssq + lambda_n
                sprod = np.dot((y - p), xpen[:, j])
                b_old = beta_pen[j]
                b_new = b_old * ssq / ssq_pen + sprod / ssq_pen
                beta_pen[j] = b_new

                # update lin predictor for all subjects for covariate j
                z = z + (b_new - b_old) * xpen[:, j]
                p = np.exp(z)
                p = p / (1 + p)
                v = p * (1 - p)

        # after sweep through all cols, check convergence
        lnlike = np.sum(np.log(p[y == 1])) + np.sum(np.log(1 - p[y == 0])) - .5 * lambda_n * np.sum(beta_pen ** 2)
        lnlike_change = np.abs(lnlike - lnlike_old) / (np.abs(lnlike_old) + 1.0)
        lnlike_old = lnlike

    return beta0, beta_cov, beta_pen, lnlike, it, v


def logistic(y, x, max_iter=25, eps=0.0000001):
    """ Logistic regression  """

    n = x.shape[0]
    m = x.shape[1]

    # add col of 1's for intercept
    xf = np.zeros((n, m + 1))
    xf[:, 0] = 1.0
    xf[:, 1:m + 1] = x.copy()
    prev = y.mean()
    beta0 = np.log(prev / (1 - prev))
    beta = np.zeros(m + 1)
    beta[0] = beta0

    nx = m + 1
    p = np.exp(np.dot(xf, beta))
    p = p / (1 + p)
    lnlike_old = np.sum(np.log(p[y == 1])) + np.sum(np.log(1 - p[y == 0]))
    lnlike_null = lnlike_old
    lnlike_change = 10
    xw = xf.copy()
    it = 0
    while (lnlike_change > eps) and (it < max_iter):
        print("it = ", it)

        it = it + 1
        v = p * (1 - p)
        for j in range(nx):
            xw[:, j] = xf[:, j] * v
            vmat = np.dot(xf.T, xw)
            Z = np.dot(xf, beta) + (y - p) / v
            beta, _, rank, _ = np.linalg.lstsq(vmat, np.dot(xf.T, v * Z), rcond=None)
            p = np.exp(np.dot(xf, beta))
            p = p / (1 + p)
            lnlike = np.sum(np.log(p[y == 1])) + np.sum(np.log(1 - p[y == 0]))
            lnlike_change = np.abs(lnlike - lnlike_old) / (np.abs(lnlike_old) + 1.0)
            lnlike_old = lnlike

    lrt = 2 * (lnlike - lnlike_null)
    df = rank - 1
    if df > 0:
        pval = 1 - stats.chi2.cdf(lrt, df)
    else:
        pval = 1.0

    return beta, lnlike, it, lrt, df, pval


