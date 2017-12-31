# EuropeanOptionPayoff.py

"""
Payoff of European Options
"""


def EuropeanCallOptionPayoff(stock_price, strike_price):
    # compute the payoff
    european_call_payoff = max(stock_price-strike_price, 0)
    return european_call_payoff


def EuropeanPutOptionPayoff(stock_price, strike_price):
    # compute the payoff
    european_put_payoff = max(strike_price-stock_price, 0)
    return european_put_payoff
