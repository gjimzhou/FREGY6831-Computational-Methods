# BlackScholesEuropeanOption.py

"""
Black-Scholes price for European Options
"""

import math
from scipy.stats import norm


def BlackScholesEuropeanCall(t, T, stock_price, strike_price, interest_rate, dividend_yield, volatility):
    # compute the Black-Scholes European Call
    d1 = (math.log(stock_price / strike_price) + (interest_rate - dividend_yield + 0.5 * volatility ** 2) * (T - t)) \
         / (volatility * math.sqrt(T - t))
    d2 = (math.log(stock_price / strike_price) + (interest_rate - dividend_yield - 0.5 * volatility ** 2) * (T - t)) \
         / (volatility * math.sqrt(T - t))
    BS_european_call_price = stock_price * math.exp(-dividend_yield * (T - t)) * norm.cdf(d1) \
                             - strike_price * math.exp(-interest_rate * (T - t)) * norm.cdf(d2)
    BS_european_call_delta = math.exp(-dividend_yield * (T - t)) * norm.cdf(d1)
    BS_european_call_gamma = (math.exp(-dividend_yield * (T - t)) * norm.pdf(d1))\
                             / (stock_price * volatility * math.sqrt(T - t))
    BS_european_call_theta = -(math.exp(-dividend_yield * (T - t)) * stock_price * norm.pdf(d1) * volatility) \
                             / (2 * math.sqrt(T - t)) \
                             + dividend_yield * stock_price * math.exp(-dividend_yield * (T - t)) * norm.cdf(d1) \
                             - interest_rate * strike_price * math.exp(-interest_rate * (T - t)) * norm.cdf(d2)
    BS_european_call_vega = stock_price * math.exp(-dividend_yield * (T - t)) * math.sqrt(T - t) * norm.pdf(d1)
    BS_european_call_rho = strike_price * (T - t) * math.exp(-interest_rate * (T - t)) * norm.cdf(d2)
    return BS_european_call_price, BS_european_call_delta, BS_european_call_gamma, \
           BS_european_call_theta, BS_european_call_vega, BS_european_call_rho


def BlackScholesEuropeanPut(t, T, stock_price, strike_price, interest_rate, dividend_yield, volatility):
    # compute the Black-Scholes European Put
    d1 = (math.log(stock_price / strike_price) + (interest_rate - dividend_yield + 0.5 * volatility ** 2) * (T - t)) \
         / (volatility * math.sqrt(T - t))
    d2 = (math.log(stock_price / strike_price) + (interest_rate - dividend_yield - 0.5 * volatility ** 2) * (T - t)) \
         / (volatility * math.sqrt(T - t))
    BS_european_put_price = strike_price * math.exp(-interest_rate * (T - t)) * norm.cdf(-d2) \
                            - stock_price * math.exp(-dividend_yield * (T - t)) * norm.cdf(-d1)
    BS_european_put_delta = -math.exp(-dividend_yield * (T - t)) * norm.cdf(-d1)
    BS_european_put_gamma = (math.exp(-dividend_yield * (T - t)) * norm.pdf(d1))\
                             / (stock_price * volatility * math.sqrt(T - t))
    BS_european_put_theta = -(math.exp(-dividend_yield * (T - t)) * stock_price * norm.pdf(d1) * volatility) \
                             / (2 * math.sqrt(T - t)) \
                             + interest_rate * strike_price * math.exp(-interest_rate * (T - t)) * norm.cdf(-d2) \
                             - dividend_yield * stock_price * math.exp(-dividend_yield * (T - t)) * norm.cdf(-d1)
    BS_european_put_vega = stock_price * math.exp(-dividend_yield * (T - t)) * math.sqrt(T - t) * norm.pdf(d1)
    BS_european_put_rho = -strike_price * (T - t) * math.exp(-interest_rate * (T - t)) * norm.cdf(-d2)
    return BS_european_put_price, BS_european_put_delta, BS_european_put_gamma, \
           BS_european_put_theta, BS_european_put_vega, BS_european_put_rho
