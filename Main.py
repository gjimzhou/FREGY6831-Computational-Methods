# Main.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PlotFunction as pf
from ComputationalMethodsForFinance import *

# initialize values
stock_price = 100
stock_price_out = 50
stock_price_at = stock_price
stock_price_in = 150
strike_price = 100
maturity = 1
time = 0
interest_rate = 0.05
dividend_yield = 0.02
volatility = 0.30

N_size_price = 40
M_size_time = 40

stock_price_max = stock_price * math.exp(
    (interest_rate - 0.5 * volatility ** 2) * (maturity - time) + 3 * volatility * math.sqrt(maturity - time))
stock_price_min = stock_price * math.exp(
    (interest_rate - 0.5 * volatility ** 2) * (maturity - time) - 3 * volatility * math.sqrt(maturity - time))

stock_prices, times = np.meshgrid(np.linspace(stock_price_min, stock_price_max, N_size_price+1),
    np.linspace(0, maturity, M_size_time+1))

N_size_prices = np.fromiter((10 * 2 ** x for x in range(5)), np.int)
M_size_times = np.fromiter((10 * 2 ** x for x in range(5)), np.int)

cm = ComputationalMethods(stock_price, strike_price, maturity, interest_rate, dividend_yield, volatility)
fdm = FiniteDifferenceMethods(stock_price, strike_price, maturity, interest_rate, dividend_yield, volatility)
rbfa = RadialBasisFunctionApproaches(stock_price, strike_price, maturity, interest_rate, dividend_yield, volatility)
mcsfsp = MonteCarloSimulationForStockPrice(stock_price, strike_price, maturity, interest_rate, dividend_yield, volatility)


# Black-Scholes 3D plots
# Calls
results = cm.Black_Scholes_European_Call_Dynamic(160, 160)
labels = ['Call Payoff', 'Call Price', 'Call Delta', 'Call Gamma', 'Call Theta', 'Call Vega', 'Call Rho']
values = results[:-2]
stock_prices = results[-2]
times = results[-1]
for i in range(7):
    pf.plot3D(values[i], stock_prices, times, labels[i], i)

# Puts
results = cm.Black_Scholes_European_Put_Dynamic(160, 160)
labels = ['Put Payoff', 'Put Price', 'Put Delta', 'Put Gamma', 'Put Theta', 'Put Vega', 'Put Rho']
values = results[:-2]
stock_prices = results[-2]
times = results[-1]
for i in range(7):
    pf.plot3D(values[i], stock_prices, times, labels[i], i)


# FDM 2D plots
# Calls
results = fdm.Black_Scholes_FD_Call_Dynamic(160, 160)
labels = ['Call Black-Scholes Price', 'Call Explicit Price', 'Call Implicit Price', 'Call Crank-Nicolson Price']
values = results[:-1]
stock_prices = results[-1]
for i in range(4):
    pf.plot2D(values[i], stock_prices, labels[i])

# Puts
results = fdm.Black_Scholes_FD_Put_Dynamic(160, 160)
labels = ['Put Black-Scholes Price', 'Put Explicit Price', 'Put Implicit Price', 'Put Crank-Nicolson Price']
values = results[:-1]
stock_prices = results[-1]
for i in range(4):
    pf.plot2D(values[i], stock_prices, labels[i])


# RBF 2D plots
# Calls
results = rbfa.Black_Scholes_FD_Call_Dynamic(160, 160)
labels = ['Call Black-Scholes Price', 'Call Gaussian Price', 'Call Multi-Quadric Price']
values = results[:-1]
stock_prices = results[-1]
for i in range(3):
    pf.plot2D(values[i], stock_prices, labels[i])

# Puts
results = rbfa.Black_Scholes_FD_Put_Dynamic(160, 160)
labels = ['Put Black-Scholes Price', 'Put Gaussian Price', 'Put Multi-Quadric Price']
values = results[:-1]
stock_prices = results[-1]
for i in range(3):
    pf.plot2D(values[i], stock_prices, labels[i])


# FDM/RBF table values
results = []
for s in [stock_price_out, stock_price_at, stock_price_in]:
    for n in N_size_prices:
        results.append(fdm.Black_Scholes_FD_Error(s, n, n))
    for n in N_size_prices:
        results.append(rbfa.Black_Scholes_FD_Error(s, n, n))
df = pd.DataFrame(results)
df.to_csv('FD_results.csv')


# Monte Carlo paths
# GBM
results = mcsfsp.Geometric_Brownian_Motion(10, 160, 1)
sample_paths = np.transpose(results)
plt.plot(sample_paths)
plt.xlabel('Geometric Brownian Motion')
plt.grid(True)
plt.show()

# GBM w/ J
results = mcsfsp.GBM_With_Jump_At_Fixed_Dates(0.1, 0.1, 0.1, 10, 160, 1)
sample_paths = np.transpose(results)
plt.plot(sample_paths)
plt.xlabel('Geometric Brownian Motion')
plt.grid(True)
plt.show()
