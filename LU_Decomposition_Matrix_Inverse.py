# LU_Decomposition_Matrix_Inverse.py

"""
wheels for LU decomposition
"""

import numpy as np


def LU_Decomposition_Matrix_Inverse(A, b):
    B = np.array(A, float)
    n = len(B)
    for k in range(n - 1):
        for i in range(k + 1, n):
            if abs(B[i, k]) > 1.0e-9:
                temp = B[i, k] / B[k, k]
                B[i, k + 1:n] = B[i, k + 1:n] - temp * B[k, k + 1:n]
                B[i, k] = temp

    x = np.array(b, float)
    for k in range(1, n):
        x[k] = x[k] - np.dot(B[k, 0:k], x[0:k])

    x[n - 1] = x[n - 1] / B[n - 1, n - 1]
    for k in range(n - 2, -1, -1):
        x[k] = (x[k] - np.dot(B[k, k + 1:n], x[k + 1:n])) / B[k, k]

    return x
