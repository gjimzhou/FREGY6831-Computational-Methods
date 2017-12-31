# MTH6831-Computational-Methods
## Final Project: Computational Methods Classes

This final project is to create a Python class that contains different computational methods. 



### ComputationalMethods Class

Base class that is named ComputationalMethods. This class contains:

Functions for European call and put option payoff.

Functions for Black-Scholes formula. The outputs should contain price, delta, gamma, theta, vega, rho.



### FiniteDifferenceMethods Class

Child class that is named FiniteDifferenceMethods. This class contains:

Function Black_Scholes_Explicit_FD, which solves Black-Scholes equation through explicit finite difference method.

Function Black_Scholes_Implicit_FD, which solves Black-Scholes equation through implicit finite difference method.

Function Black_Scholes_Crank_Nicolson_FD, which solves Black-Scholes equation through Crank-Nicolson finite difference method.



### RadialBasisFunctionApproaches Class

Child class that is named RadialBasisFunctionApproaches. This class contains:

Function Gaussian_Radial_Basis_Function. The outputs should contain the value of the function, first order derivative, and second order derivative.

Function Multi_Quadric_Radial_Basis_Function. The outputs should contain the value of the function, first order derivative, and second order derivative.

Function Black_Scholes_RBF_FD, which solves Black-Scholes equation through radial basis function-generated finite difference.



### MonteCarloSimulationForStockPrice Class

Child class that is named MonteCarloSimulationForStockPrice. This class contains:

Function Geometric_Brownian_Motion, which generate trajectories of geometric Brownian motion.

Function GBM_With_Jump_At_Fixed_Dates, which generate trajectories of geometric Brownian motion with jumps.
