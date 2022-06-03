
import numpy as np

# n = 5
# mat = np.random.normal(size=(n,n))
# mat2 = np.dot(mat.T, mat)
# ch = np.linalg.cholesky(mat2)
# ch_inv =  np.linalg.inv(ch)

def gchol(matrix):

    # A symmetric matrix is  decomposed as LDL', where L is a lower triangular matrix
    # return  L * diag(sqrt(D))
    EPSILON = 0.000000001
    n = matrix.shape[0]

    eps = 0
    for i in range(n):
        if matrix[i, i] > eps:
            eps = matrix[i, i]
        for j in range((i+1), n):
            matrix[j, i] = matrix[i, j]
    eps *= EPSILON
    rank = 0
    for i in range(n):
        pivot = matrix[i, i]
        if pivot < eps:
            matrix[i, i] = 0.0
        else:
            rank += 1
            for j in range((i + 1), n):
                temp = matrix[j, i] / pivot
                matrix[j, i] = temp
                matrix[j, j] -= temp * temp * pivot
                for k in range((j + 1), n):
                    matrix[k, j] -= temp * matrix[k, i]

    diag = np.zeros(n)
    for i in range(n):
        diag[i] = np.sqrt(matrix[i, i])
        matrix[i,i] = 1.0
    for i in range(n - 1):
        for j in range((i + 1), n):
            matrix[i, j] = 0.0

    for j in range(n):
        matrix[:, j] = matrix[:,j] * diag[j]


def gchol_inv(matrix):
    # very slow, of order n^3

    n = matrix.shape[0]
    inv = matrix.copy()

    for k in range(n):
        if inv[k, k] > 0.0:
            diag = inv[k, k]
            for i in range(k):
                inv[k, i] = inv[k, i] / diag
            for i in range(k, n):
                inv[i, k] = inv[i, k] / diag
            for i in range(n):
                if i == k:
                    continue
                for j in range(n):
                    if j == k:
                        continue
                    inv[i, j] = inv[i, j] - inv[i, k] * inv[k, j] * diag
            inv[k, k] = -1.0 / diag

    for i in range(n):
        for j in range(i+1):
            inv[i, j] = - inv[i, j]

    return inv

# gchol(mat2)
# inv = gchol_inv(mat2)