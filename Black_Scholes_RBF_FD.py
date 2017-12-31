# Black_Scholes_RBF_FD.py

"""
Radial Basis Function Generated Finite Difference Method for European Options
"""

import numpy as np
import math
from scipy.sparse import spdiags
import LU_Decomposition_Matrix_Inverse as lu
import copy


def Black_Scholes_RBF_FD(option_parameters, mesh_setting, RBF, payoff, S_min, S_max):
    K = option_parameters[1]
    T = option_parameters[2]
    r = option_parameters[3]
    q = option_parameters[4]

    x_min = math.log(S_min)
    x_max = math.log(S_max)

    M = mesh_setting[0]
    N = mesh_setting[1]

    delta_tau = T / M

    tau = np.linspace(T, 0, M + 1)
    x = np.linspace(x_min, x_max, N + 1)
    
    # initial value
    initial_value = np.fromiter((payoff(math.exp(n), K) for n in x), np.float)
    if T == 0:
        return copy.deepcopy(initial_value)

    # boundary conditions
    bound_min = np.fromiter(((payoff(math.exp(x_min) * math.exp(-q * m), K * math.exp(-r * m))) for m in tau), np.float)
    bound_max = np.fromiter(((payoff(math.exp(x_max) * math.exp(-q * m), K * math.exp(-r * m))) for m in tau), np.float)

    # calculate the weighting matrix W
    epsilon = 1.0
    
    L_matrix = RBF(epsilon, x, x)[0]
    
    W_matrix = np.array([BS_Wi_RBF(option_parameters, mesh_setting, RBF, L_matrix, i, x, S_min, S_max) for i in range(1, N)], dtype=float)
    w_min = W_matrix[0, 0]
    w_mid = W_matrix[0, 1]
    w_max = W_matrix[-1, -1]
           
    W = spdiags(np.array([[w_mid for a in range(N + 1)], [w_min for a in range(N + 1)], [w_max for a in range(N + 1)]]),
        np.array([0, -1, 1]), N - 1, N - 1).toarray()

    I = np.identity(N - 1)
    A = I - 0.5 * delta_tau * W
    B = I + 0.5 * delta_tau * W

    # solving the ODE by Crank-Nicolson
    v = copy.deepcopy(initial_value)
    for k in range(M):
        v[0] = bound_min[k]
        v[-1] = bound_max[k]
        b = np.zeros(N - 1)
        b[0] = 0.5 * delta_tau * w_min * (bound_min[k] + bound_min[k + 1])
        b[-1] = 0.5 * delta_tau * w_max * (bound_max[k] + bound_max[k + 1])
        temp = np.dot(B, v[1:-1])
        v[1:-1] = lu.LU_Decomposition_Matrix_Inverse(A, temp + b)

    option_price_RBF_FD_G = v
    return option_price_RBF_FD_G


def BS_Wi_RBF(option_parameters, mesh_setting, RBF,L_matrix, i, x, S_min, S_max):
    Li = L_matrix[i-1:i+2, i-1:i+2]
    Lphi = BS_operator_RBF(option_parameters, mesh_setting, RBF, x[i], S_min, S_max)
    Wi = lu.LU_Decomposition_Matrix_Inverse(Li, Lphi)
    return Wi


def BS_operator_RBF(option_parameters, mesh_setting, RBF, x_c, S_min, S_max):
    r = option_parameters[3]
    sig = option_parameters[5]

    x_min = math.log(S_min)
    x_max = math.log(S_max)

    N = mesh_setting[1]

    delta_x = (x_max - x_min) / N
    x = [x_c - delta_x, x_c, x_c + delta_x]

    epsilon = 1
    L = np.fromiter((RBF(epsilon, x_c, b)[0] for b in x), np.float)
    L_x = np.fromiter((RBF(epsilon, x_c, b)[1] for b in x), np.float)
    L_xx = np.fromiter((RBF(epsilon, x_c, b)[2] for b in x), np.float)

    L_operator = (r - 0.5 * sig ** 2) * L_x + 0.5 * sig ** 2 * L_xx - r * L
    return L_operator
