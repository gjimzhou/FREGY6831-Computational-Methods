# RandomNumberGenerator.py

"""
Generators for uniform and normal random variables
"""

import math


def Linear_Congruential_Generator(N, seed=1):
    seed = int(seed)
    m = 2147483647
    a = 39373
    q = int(m / a)
    r = m % a

    uniform_random_number = []

    for i in range(N):
        k = int(seed / q)
        seed = a * (seed - k * q) - k * r
        if seed < 0:
            seed = seed + m
        uniform_random_number.append(seed / m)

    return uniform_random_number


def Marsaglia_Bray_Generator(N, seed=1):
    seed = int(seed)

    standard_gaussian_random_number = []
    M = len(standard_gaussian_random_number)

    while M < N:
        uniform_random_number = Linear_Congruential_Generator(N - M, seed)
        seed = int(uniform_random_number[-1] * 2147483647)

        for i in range(int((N - M) / 2)):
            u1 = 2 * uniform_random_number[2 * i] - 1
            u2 = 2 * uniform_random_number[2 * i + 1] - 1
            x = u1 ** 2 + u2 ** 2

            if x <= 1:
                y = math.sqrt(-2 * math.log(x) / x)
                standard_gaussian_random_number.append(u1 * y)
                standard_gaussian_random_number.append(u2 * y)

        M = len(standard_gaussian_random_number)

    return standard_gaussian_random_number
