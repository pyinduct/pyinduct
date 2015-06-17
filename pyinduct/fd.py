from __future__ import division
from sympy.calculus.finite_diff import finite_diff_weights
import numpy as np

__author__ = 'Stefan Ecklebe'
"""
    Module that serves as source for Finite Difference Methods
"""


def _calc_weights(M, alpha, x0):
    """
    Compute weights for generation of Finite Difference.
    Implementation according to Fornberg B -  Generation of finite difference formulas on arbitrarily spaced grids
    :param M: order of highest derivative to approximate
    :param alpha: set of grid points (alpha_0, .., alpha_N) to base on
    :return: weights delta_0 to delta_N
    """

    res = finite_diff_weights(M, alpha, x0)
    return res

    N = len(alpha) - 1

    delta = np.zeros((M+1, N+1, N+1))  # der_order, accuracy_order, element
    delta[0, 0, 0] = 1

    c1 = 1

    for n in range(1, N+1):
        c2 = 1
        for nu in range(n-1):
            c3 = alpha[n] - alpha[nu]
            c2 = c2*c3
            for m in range(0, min(n, M)+1):
                delta[m, n, nu] = ((alpha[n] - x0)*delta[m, n-1, nu] - m*delta[m-1, n-1, nu])/c3
        for m in range(0, min(n, M)+1):
            delta[m, n, n] = c1/c2*(m*delta[m-1, n-1, n-1] - (alpha[n-1] - x0)*delta[m, n-1, n-1])
        c1 = c2

    return delta