# RadialBasisFunction.py

"""
Wheels for RBF
"""

import numpy as np


def Gaussian_Radial_Basis_Function(epsilon, x, xi):
    X,XI = np.meshgrid(x, xi)
    L_G = Gaussian_RBF(epsilon, X, XI) 
    L_x_G = FO_Gaussian_RBF(epsilon, X, XI) 
    L_xx_G = SO_Gaussian_RBF(epsilon, X, XI) 
    return L_G, L_x_G, L_xx_G


def Gaussian_RBF(epsilon, x, xi):
    return np.exp(-epsilon ** 2 * (x - xi) ** 2)


def FO_Gaussian_RBF(epsilon, x, xi):
    return -2 * epsilon ** 2 * (x - xi) * np.exp(-epsilon ** 2 * (x - xi) ** 2)


def SO_Gaussian_RBF(epsilon, x, xi):
    return (-2 * epsilon ** 2 + 4 * epsilon ** 4 * (x - xi) ** 2) * np.exp(-epsilon ** 2 * (x - xi) ** 2)


def Multi_Quadric_Radial_Basis_Function(epsilon, x, xi):
    X,XI = np.meshgrid(x, xi)
    L_G = Multi_Quadric_RBF(epsilon, X, XI) 
    L_x_G = FO_Multi_Quadric_RBF(epsilon, X, XI) 
    L_xx_G = SO_Multi_Quadric_RBF(epsilon, X, XI) 
    return L_G, L_x_G, L_xx_G


def Multi_Quadric_RBF(epsilon, x, xi):
    return np.sqrt(1 + epsilon ** 2 * (x - xi) ** 2)


def FO_Multi_Quadric_RBF(epsilon, x, xi):
    return (epsilon ** 2 * (x - xi)) / np.sqrt(1 + epsilon ** 2 * (x - xi) ** 2)


def SO_Multi_Quadric_RBF(epsilon, x, xi):
    return epsilon ** 2 / np.sqrt(1 + epsilon ** 2 * (x - xi) ** 2) - (epsilon ** 4 * (x - xi) ** 2) / np.sqrt(1 +
        epsilon ** 2 * (x - xi) ** 2) ** 3

