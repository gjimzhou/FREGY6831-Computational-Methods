# Black_Scholes_Implicit_FD.py

"""
Implicit Finite Difference Method for Black-Scholes PDE
"""

import math
import numpy as np
from scipy.sparse import spdiags
import LU_Decomposition_Matrix_Inverse as lu


def Black_Scholes_Implicit_FD(stock_price_max, stock_price_min, N_size_price, M_size_time, T, t, strike_price,
                              interest_rate, dividend_yield, volatility, payoff_function):
    s = np.linspace(stock_price_min, stock_price_max, N_size_price+1)
    tao = np.linspace(T, t, M_size_time+1)
    delta_t = float(T - t) / M_size_time
    delta_s = float(stock_price_max - stock_price_min) / N_size_price
    r = interest_rate - dividend_yield

    l = -(volatility ** 2 * (s ** 2) * delta_t / (2 * (delta_s ** 2)) - r * s * delta_t / (2 * delta_s))
    d = 1 + interest_rate * delta_t + volatility ** 2 * (s ** 2) * delta_t / (delta_s ** 2)
    u = -(volatility ** 2 * (s ** 2) * delta_t / (2 * (delta_s ** 2)) + r * s * delta_t / (2 * delta_s))

    A = spdiags(np.array([d[1:N_size_price], l[2:(N_size_price+1)], u[0:(N_size_price-1)]]), np.array([0, -1, 1]),
                N_size_price-1, N_size_price-1).toarray()
    v = np.fromiter((payoff_function(x, strike_price) for x in s), np.float)
    bound_min = np.fromiter((payoff_function(stock_price_min * math.exp(-dividend_yield * x),
                                             strike_price * math.exp(-interest_rate * x)) for x in tao), np.float)
    bound_max = np.fromiter((payoff_function(stock_price_max * math.exp(-dividend_yield * x),
                                             strike_price * math.exp(-interest_rate * x)) for x in tao), np.float)
    option_prices = [v.tolist()]

    for k in range(1, M_size_time+1):
        v[0] = bound_min[k]
        v[-1] = bound_max[k]
        b = np.zeros(N_size_price-1)
        b[0] = -v[0] * l[1]
        b[-1] = -v[-1] * u[-2]
        v[1:-1] = lu.LU_Decomposition_Matrix_Inverse(A, v[1:-1] + b)
        option_prices.append(v.tolist())

    return option_prices

