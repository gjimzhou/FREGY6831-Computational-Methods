# MonteCarloForStockPrice.py

"""
Generators for jump diffusion processes
"""

import math
import numpy as np
import RandomNumberGenerator as rng


def Jump_Diffusion_Process_At_Fixed_Dates(S, T, r, sig, q, lam, a, b, N, M, seed):
    standard_gaussian_random_number = rng.Marsaglia_Bray_Generator(2 * N * M, seed)
    dt = T / M

    jump_diffusion_paths = []

    for i in range(N):
        Si = S
        jump_diffusion_path = [Si]

        for j in range(M):
            z1 = standard_gaussian_random_number[2 * (M * i + j)]
            z2 = standard_gaussian_random_number[2 * (M * i + j) + 1]
            n = np.random.poisson(lam)

            Si = Si * math.exp((r - q - sig ** 2) * dt + sig * math.sqrt(dt) * z1 + a * n + b * math.sqrt(n) * z2)
            jump_diffusion_path.append(Si)

        jump_diffusion_paths.append(jump_diffusion_path)

    return jump_diffusion_paths
