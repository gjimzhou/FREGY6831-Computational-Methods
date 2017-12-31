# ComputationalMethodsForFinance.py

"""
Computational Methods Classes
"""

import math
import numpy as np
import EuropeanOptionPayoff as ep
import BlackScholesEuropeanOption as bs
import BlackScholesExplicitFD as ex
import BlackScholesImplicitFD as im
import BlackScholesCrankNicolsonFD as cn
import RadialBasisFunction as rbf
import BlackScholesRBFFD as rb
import MonteCarloForStockPrice as mc


class ComputationalMethods(object):
    def __init__(self, stock_price, strike_price, maturity, interest_rate, dividend_yield, volatility):
        self.stock_price = stock_price
        self.strike_price = strike_price
        self.maturity = maturity
        self.interest_rate = interest_rate
        self.dividend_yield = dividend_yield
        self.volatility = volatility

    @staticmethod
    def European_Call_Payoff(stock_price, strike_price):
        payoff = ep.EuropeanCallOptionPayoff(stock_price, strike_price)
        return payoff

    @staticmethod
    def European_Put_Payoff(stock_price, strike_price):
        payoff = ep.EuropeanPutOptionPayoff(stock_price, strike_price)
        return payoff

    def Black_Scholes_European_Call(self, time, stock_price):
        strike_price = self.strike_price
        maturity = self.maturity
        interest_rate = self.interest_rate
        dividend_yield = self.dividend_yield
        volatility = self.volatility
        results = bs.BlackScholesEuropeanCall(
            time, maturity, stock_price, strike_price, interest_rate, dividend_yield, volatility)
        return results

    def Black_Scholes_European_Put(self, time, stock_price):
        strike_price = self.strike_price
        maturity = self.maturity
        interest_rate = self.interest_rate
        dividend_yield = self.dividend_yield
        volatility = self.volatility
        results = bs.BlackScholesEuropeanPut(
            time, maturity, stock_price, strike_price, interest_rate, dividend_yield, volatility)
        return results

    def Stock_Price_Max(self):
        stock_price = self.stock_price
        maturity = self.maturity
        interest_rate = self.interest_rate
        dividend_yield = self.dividend_yield
        volatility = self.volatility
        time = 0
        return stock_price * math.exp((interest_rate - dividend_yield - 0.5 * volatility ** 2) * (
            maturity - time) + 3 * volatility * math.sqrt(maturity - time))

    def Stock_Price_Min(self):
        stock_price = self.stock_price
        maturity = self.maturity
        interest_rate = self.interest_rate
        dividend_yield = self.dividend_yield
        volatility = self.volatility
        time = 0
        return stock_price * math.exp((interest_rate - dividend_yield - 0.5 * volatility ** 2) * (
            maturity - time) - 3 * volatility * math.sqrt(maturity - time))

    def Black_Scholes_European_Call_Dynamic(self, N, M):
        strike_price = self.strike_price
        maturity = self.maturity
        interest_rate = self.interest_rate
        dividend_yield = self.dividend_yield
        volatility = self.volatility
        stock_price_max = self.Stock_Price_Max()
        stock_price_min = self.Stock_Price_Min()
        stock_prices = np.linspace(stock_price_min, stock_price_max, N + 1)
        delta_t = maturity / M
        times = np.linspace(maturity - delta_t, 0, M + 1)

        payoffs = []
        prices = []
        deltas = []
        gammas = []
        thetas = []
        vegas = []
        rhos = []

        for t in times:
            tmp = np.array([bs.BlackScholesEuropeanCall(t, maturity, s, strike_price, interest_rate, dividend_yield,
                volatility) for s in stock_prices])
            tmp = tmp.T
            payoffs.append([ep.EuropeanCallOptionPayoff(s, strike_price) for s in stock_prices])
            prices.append(tmp[0])
            deltas.append(tmp[1])
            gammas.append(tmp[2])
            thetas.append(tmp[3])
            vegas.append(tmp[4])
            rhos.append(tmp[5])

        stock_prices_meshed, times_meshed = np.meshgrid(stock_prices, times)
        return payoffs, prices, deltas, gammas, thetas, vegas, rhos, stock_prices_meshed, times_meshed

    def Black_Scholes_European_Put_Dynamic(self, N, M):
        strike_price = self.strike_price
        maturity = self.maturity
        interest_rate = self.interest_rate
        dividend_yield = self.dividend_yield
        volatility = self.volatility
        stock_price_max = self.Stock_Price_Max()
        stock_price_min = self.Stock_Price_Min()
        stock_prices = np.linspace(stock_price_min, stock_price_max, N + 1)
        delta_t = maturity / M
        times = np.linspace(maturity - delta_t, 0, M + 1)

        payoffs = []
        prices = []
        deltas = []
        gammas = []
        thetas = []
        vegas = []
        rhos = []

        for t in times:
            tmp = np.array([bs.BlackScholesEuropeanPut(t, maturity, s, strike_price, interest_rate, dividend_yield,
                volatility) for s in stock_prices])
            tmp = tmp.T
            payoffs.append([ep.EuropeanPutOptionPayoff(s, strike_price) for s in stock_prices])
            prices.append(tmp[0])
            deltas.append(tmp[1])
            gammas.append(tmp[2])
            thetas.append(tmp[3])
            vegas.append(tmp[4])
            rhos.append(tmp[5])

        stock_prices_meshed, times_meshed = np.meshgrid(stock_prices, times)
        return payoffs, prices, deltas, gammas, thetas, vegas, rhos, stock_prices_meshed, times_meshed


class FiniteDifferenceMethods(ComputationalMethods):
    def __init__(self, stock_price, strike_price, maturity, interest_rate, dividend_yield, volatility):
        super().__init__(stock_price, strike_price, maturity, interest_rate, dividend_yield, volatility)

    def Black_Scholes_Explicit_FD(self, N, M, payoff):
        strike_price = self.strike_price
        maturity = self.maturity
        interest_rate = self.interest_rate
        dividend_yield = self.dividend_yield
        volatility = self.volatility
        time = 0
        stock_price_max = self.Stock_Price_Max()
        stock_price_min = self.Stock_Price_Min()
        results = ex.Black_Scholes_Explicit_FD(stock_price_max, stock_price_min, N, M, maturity, time, strike_price,
            interest_rate, dividend_yield, volatility, payoff)[-1]
        return results

    def Black_Scholes_Implicit_FD(self, N, M, payoff):
        strike_price = self.strike_price
        maturity = self.maturity
        interest_rate = self.interest_rate
        dividend_yield = self.dividend_yield
        volatility = self.volatility
        time = 0
        stock_price_max = self.Stock_Price_Max()
        stock_price_min = self.Stock_Price_Min()
        results = im.Black_Scholes_Implicit_FD(stock_price_max, stock_price_min, N, M, maturity, time, strike_price,
            interest_rate, dividend_yield, volatility, payoff)[-1]
        return results

    def Black_Scholes_Crank_Nicolson_FD(self, N, M, payoff):
        strike_price = self.strike_price
        maturity = self.maturity
        interest_rate = self.interest_rate
        dividend_yield = self.dividend_yield
        volatility = self.volatility
        time = 0
        stock_price_max = self.Stock_Price_Max()
        stock_price_min = self.Stock_Price_Min()
        results = cn.Black_Scholes_Crank_Nicolson_FD(stock_price_max, stock_price_min, N, M, maturity, time,
            strike_price, interest_rate, dividend_yield, volatility, payoff)[-1]
        return results

    def Black_Scholes_FD_Call_Dynamic(self, N, M):
        strike_price = self.strike_price
        maturity = self.maturity
        interest_rate = self.interest_rate
        dividend_yield = self.dividend_yield
        volatility = self.volatility
        time = 0
        stock_price_max = self.Stock_Price_Max()
        stock_price_min = self.Stock_Price_Min()
        stock_prices = np.linspace(stock_price_min, stock_price_max, N + 1)

        tmp = np.array([bs.BlackScholesEuropeanCall(time, maturity, s, strike_price, interest_rate, dividend_yield,
                                                    volatility) for s in stock_prices])
        tmp = tmp.T
        BS_prices = tmp[0]
        FD_Ex_prices = self.Black_Scholes_Explicit_FD(N, M, self.European_Call_Payoff)
        FD_Im_prices = self.Black_Scholes_Implicit_FD(N, M, self.European_Call_Payoff)
        FD_CN_prices = self.Black_Scholes_Crank_Nicolson_FD(N, M, self.European_Call_Payoff)

        return BS_prices, FD_Ex_prices, FD_Im_prices, FD_CN_prices, stock_prices

    def Black_Scholes_FD_Put_Dynamic(self, N, M):
        strike_price = self.strike_price
        maturity = self.maturity
        interest_rate = self.interest_rate
        dividend_yield = self.dividend_yield
        volatility = self.volatility
        time = 0
        stock_price_max = self.Stock_Price_Max()
        stock_price_min = self.Stock_Price_Min()
        stock_prices = np.linspace(stock_price_min, stock_price_max, N + 1)

        tmp = np.array([bs.BlackScholesEuropeanPut(time, maturity, s, strike_price, interest_rate, dividend_yield,
                                                    volatility) for s in stock_prices])
        tmp = tmp.T
        BS_prices = tmp[0]
        FD_Ex_prices = self.Black_Scholes_Explicit_FD(N, M, self.European_Put_Payoff)
        FD_Im_prices = self.Black_Scholes_Implicit_FD(N, M, self.European_Put_Payoff)
        FD_CN_prices = self.Black_Scholes_Crank_Nicolson_FD(N, M, self.European_Put_Payoff)

        return BS_prices, FD_Ex_prices, FD_Im_prices, FD_CN_prices, stock_prices

    @staticmethod
    def Black_Scholes_FD_price(stock_price, stock_prices, option_prices):
        left = len(stock_prices[stock_prices < stock_price])
        right = left + 1
        stock_price_left = stock_prices[left]
        stock_price_right = stock_prices[right]
        option_price_left = option_prices[left]
        option_price_right = option_prices[right]
        option_price = (option_price_left * (stock_price_right - stock_price) + option_price_right * (
            stock_price - stock_price_left)) / (stock_price_right - stock_price_left)
        return option_price

    @staticmethod
    def FD_Error(FD_price, BS_price):
        return abs(FD_price - BS_price) / abs(BS_price)

    def Black_Scholes_FD_Error(self, stock_price, N, M):
        time = 0
        stock_price_max = self.Stock_Price_Max()
        stock_price_min = self.Stock_Price_Min()
        stock_prices = np.linspace(stock_price_min, stock_price_max, N + 1)
        results = []

        # Call
        BS_price = self.Black_Scholes_European_Call(time, stock_price)[0]
        FD_Ex_prices = self.Black_Scholes_Explicit_FD(N, M, self.European_Call_Payoff)
        FD_Im_prices = self.Black_Scholes_Implicit_FD(N, M, self.European_Call_Payoff)
        FD_CN_prices = self.Black_Scholes_Crank_Nicolson_FD(N, M, self.European_Call_Payoff)
        FD_Ex_price = self.Black_Scholes_FD_price(stock_price, stock_prices, FD_Ex_prices)
        FD_Im_price = self.Black_Scholes_FD_price(stock_price, stock_prices, FD_Im_prices)
        FD_CN_price = self.Black_Scholes_FD_price(stock_price, stock_prices, FD_CN_prices)
        FD_Ex_error = self.FD_Error(FD_Ex_price, BS_price)
        FD_Im_error = self.FD_Error(FD_Im_price, BS_price)
        FD_CN_error = self.FD_Error(FD_CN_price, BS_price)
        results.append([BS_price, FD_Ex_price, FD_Im_price, FD_CN_price, FD_Ex_error, FD_Im_error, FD_CN_error])

        # Put
        BS_price = self.Black_Scholes_European_Put(time, stock_price)[0]
        FD_Ex_prices = self.Black_Scholes_Explicit_FD(N, M, self.European_Put_Payoff)
        FD_Im_prices = self.Black_Scholes_Implicit_FD(N, M, self.European_Put_Payoff)
        FD_CN_prices = self.Black_Scholes_Crank_Nicolson_FD(N, M, self.European_Put_Payoff)
        FD_Ex_price = self.Black_Scholes_FD_price(stock_price, stock_prices, FD_Ex_prices)
        FD_Im_price = self.Black_Scholes_FD_price(stock_price, stock_prices, FD_Im_prices)
        FD_CN_price = self.Black_Scholes_FD_price(stock_price, stock_prices, FD_CN_prices)
        FD_Ex_error = self.FD_Error(FD_Ex_price, BS_price)
        FD_Im_error = self.FD_Error(FD_Im_price, BS_price)
        FD_CN_error = self.FD_Error(FD_CN_price, BS_price)
        results.append([BS_price, FD_Ex_price, FD_Im_price, FD_CN_price, FD_Ex_error, FD_Im_error, FD_CN_error])

        return results


class RadialBasisFunctionApproaches(ComputationalMethods):
    def __init__(self, stock_price, strike_price, maturity, interest_rate, dividend_yield, volatility):
        super().__init__(stock_price, strike_price, maturity, interest_rate, dividend_yield, volatility)

    @staticmethod
    def Gaussian_Radial_Basis_Function(epsilon, x, xi):
        return rbf.Gaussian_Radial_Basis_Function(epsilon, x, xi)

    @staticmethod
    def Multi_Quadric_Radial_Basis_Function(epsilon, x, xi):
        return rbf.Multi_Quadric_Radial_Basis_Function(epsilon, x, xi)

    def Black_Scholes_RBF_FD(self, N, M, RBF, payoff):
        stock_price = self.stock_price
        strike_price = self.strike_price
        maturity = self.maturity
        interest_rate = self.interest_rate
        dividend_yield = self.dividend_yield
        volatility = self.volatility
        option_parameters = [stock_price, strike_price, maturity, interest_rate, dividend_yield, volatility]
        mesh = [N, M]
        stock_price_max = self.Stock_Price_Max()
        stock_price_min = self.Stock_Price_Min()
        results = rb.Black_Scholes_RBF_FD(option_parameters, mesh, RBF, payoff, stock_price_min, stock_price_max)
        return results

    def Black_Scholes_FD_Call_Dynamic(self, N, M):
        strike_price = self.strike_price
        maturity = self.maturity
        interest_rate = self.interest_rate
        dividend_yield = self.dividend_yield
        volatility = self.volatility
        time = 0
        stock_price_max = self.Stock_Price_Max()
        stock_price_min = self.Stock_Price_Min()
        stock_prices = np.linspace(stock_price_min, stock_price_max, N + 1)

        tmp = np.array([bs.BlackScholesEuropeanCall(time, maturity, s, strike_price, interest_rate, dividend_yield,
                                                    volatility) for s in stock_prices])
        tmp = tmp.T
        BS_prices = tmp[0]
        FD_G_prices = self.Black_Scholes_RBF_FD(
            N, M, self.Gaussian_Radial_Basis_Function, self.European_Call_Payoff)
        FD_MQ_prices = self.Black_Scholes_RBF_FD(
            N, M, self.Multi_Quadric_Radial_Basis_Function, self.European_Call_Payoff)

        return BS_prices, FD_G_prices, FD_MQ_prices, stock_prices

    def Black_Scholes_FD_Put_Dynamic(self, N, M):
        strike_price = self.strike_price
        maturity = self.maturity
        interest_rate = self.interest_rate
        dividend_yield = self.dividend_yield
        volatility = self.volatility
        time = 0
        stock_price_max = self.Stock_Price_Max()
        stock_price_min = self.Stock_Price_Min()
        stock_prices = np.linspace(stock_price_min, stock_price_max, N + 1)

        tmp = np.array([bs.BlackScholesEuropeanPut(time, maturity, s, strike_price, interest_rate, dividend_yield,
                                                    volatility) for s in stock_prices])
        tmp = tmp.T
        BS_prices = tmp[0]
        FD_G_prices = self.Black_Scholes_RBF_FD(
            N, M, self.Gaussian_Radial_Basis_Function, self.European_Put_Payoff)
        FD_MQ_prices = self.Black_Scholes_RBF_FD(
            N, M, self.Multi_Quadric_Radial_Basis_Function, self.European_Put_Payoff)

        return BS_prices, FD_G_prices, FD_MQ_prices, stock_prices

    @staticmethod
    def Black_Scholes_FD_price(stock_price, stock_prices, option_prices):
        left = len(stock_prices[stock_prices < stock_price])
        right = left + 1
        stock_price_left = stock_prices[left]
        stock_price_right = stock_prices[right]
        option_price_left = option_prices[left]
        option_price_right = option_prices[right]
        option_price = (option_price_left * (stock_price_right - stock_price) + option_price_right * (
            stock_price - stock_price_left)) / (stock_price_right - stock_price_left)
        return option_price

    @staticmethod
    def FD_Error(FD_price, BS_price):
        return abs(FD_price - BS_price) / abs(BS_price)

    def Black_Scholes_FD_Error(self, stock_price, N, M):
        time = 0
        stock_price_max = self.Stock_Price_Max()
        stock_price_min = self.Stock_Price_Min()
        stock_prices = np.linspace(stock_price_min, stock_price_max, N)
        results = []

        # Call
        BS_price = self.Black_Scholes_European_Call(time, stock_price)[0]
        FD_G_prices = self.Black_Scholes_RBF_FD(
            N, M, self.Gaussian_Radial_Basis_Function, self.European_Call_Payoff)
        FD_MQ_prices = self.Black_Scholes_RBF_FD(
            N, M, self.Multi_Quadric_Radial_Basis_Function, self.European_Call_Payoff)
        FD_G_price = self.Black_Scholes_FD_price(stock_price, stock_prices, FD_G_prices)
        FD_MQ_price = self.Black_Scholes_FD_price(stock_price, stock_prices, FD_MQ_prices)
        FD_G_error = self.FD_Error(FD_G_price, BS_price)
        FD_MQ_error = self.FD_Error(FD_MQ_price, BS_price)
        results.append([BS_price, FD_G_price, FD_MQ_price, FD_G_error, FD_MQ_error])

        # Put
        BS_price = self.Black_Scholes_European_Put(time, stock_price)[0]
        FD_G_prices = self.Black_Scholes_RBF_FD(
            N, M, self.Gaussian_Radial_Basis_Function, self.European_Put_Payoff)
        FD_MQ_prices = self.Black_Scholes_RBF_FD(
            N, M, self.Multi_Quadric_Radial_Basis_Function, self.European_Put_Payoff)
        FD_G_price = self.Black_Scholes_FD_price(stock_price, stock_prices, FD_G_prices)
        FD_MQ_price = self.Black_Scholes_FD_price(stock_price, stock_prices, FD_MQ_prices)
        FD_G_error = self.FD_Error(FD_G_price, BS_price)
        FD_MQ_error = self.FD_Error(FD_MQ_price, BS_price)
        results.append([BS_price, FD_G_price, FD_MQ_price, FD_G_error, FD_MQ_error])

        return results


class MonteCarloSimulationForStockPrice(ComputationalMethods):
    def __init__(self, stock_price, strike_price, maturity, interest_rate, dividend_yield, volatility):
        super().__init__(stock_price, strike_price, maturity, interest_rate, dividend_yield, volatility)

    def Geometric_Brownian_Motion(self, N, M, seed):
        stock_price = self.stock_price
        maturity = self.maturity
        interest_rate = self.interest_rate
        volatility = self.volatility
        dividend_yield = self.dividend_yield
        paths = mc.Jump_Diffusion_Process_At_Fixed_Dates(stock_price, maturity, interest_rate, volatility,
            dividend_yield, 0, 0, 0, N, M, seed)
        return paths

    def GBM_With_Jump_At_Fixed_Dates(self, lam, a, b, N, M, seed):
        stock_price = self.stock_price
        maturity = self.maturity
        interest_rate = self.interest_rate
        volatility = self.volatility
        dividend_yield = self.dividend_yield
        paths = mc.Jump_Diffusion_Process_At_Fixed_Dates(stock_price, maturity, interest_rate, volatility,
            dividend_yield, lam, a, b, N, M, seed)
        return paths
