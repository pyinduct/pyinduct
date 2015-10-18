from __future__ import division
import numpy as np
import scipy.integrate as si
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from pyinduct import get_initial_functions, register_functions
import pyinduct
import utils as ut
import placeholder as ph
from core import Function, LagrangeFirstOrder, LagrangeSecondOrder, back_project_from_initial_functions
from placeholder import FieldVariable, TestFunction
from visualization import EvalData
import pyqtgraph as pg
from numbers import Number
import warnings
import copy as cp
import pyqtgraph as pg


__author__ = 'marcus'


class TransformedSecondOrderEigenfunction(Function):
    """
    Provide the eigenfunction y to an eigenvalue-problem of the form
    a2(z)y''(z) + a1(z)y'(z) + a0(z)y(z) = w y(z)
    where w is a predefined (potentially complex) eigenvalue and z0 <= z <= z1 is the domain.
    :param target_eigenvalue: w (float)
    :param init_state_vect: y(0) = [Re{y(0)}, Re{y'(0)}, Im{y(0)}, Im{y'(0)}] (list of floats)
    :param dgl_coefficients: [a2(z), a1(z), a0(z)] (list of function handles)
    :param domain: [z0, ..... , z1] (list of floats)
    :return: y(z) = [Re{y(z)}, Re{y'(z)}, Im{y(z)}, Im{y'(z)}] (list of numpy arrays)
    """

    def __init__(self, target_eigenvalue, init_state_vect, dgl_coefficients, domain):

        if not all([isinstance(state, (int, float, long)) for state in init_state_vect]) \
                and len(init_state_vect) == 4 and isinstance(init_state_vect, (list, tuple)):
            raise TypeError
        if not len(dgl_coefficients) == 3 and isinstance(dgl_coefficients, (list, tuple)) \
                and all([callable(coef) or isinstance(coef, (int, float, long)) for coef in dgl_coefficients]):
            raise TypeError
        if not isinstance(domain, (np.ndarray, list)) \
                or not all([isinstance(num, (int, long, float)) for num in domain]):
            raise TypeError

        if isinstance(target_eigenvalue, complex):
            self._eig_val_real = target_eigenvalue.real
            self._eig_val_imag = target_eigenvalue.imag
        elif isinstance(target_eigenvalue, (int, long, float)):
            self._eig_val_real = target_eigenvalue
            self._eig_val_imag = 0.
        else:
            raise TypeError

        self._init_state_vect = init_state_vect
        self._a2, self._a1, self._a0 = [ut._convert_to_function(coef) for coef in dgl_coefficients]
        self._domain = domain

        state_vect = self._transform_eigenfunction()
        self._transf_eig_func_real, self._transf_d_eig_func_real = state_vect[0:2]
        self._transf_eig_func_imag, self._transf_d_eig_func_imag = state_vect[2:4]

        Function.__init__(self, self._phi, nonzero=(domain[0], domain[-1]), derivative_handles=[self._d_phi])

    def _ff(self, y, z):
        a2, a1, a0 = [self._a2, self._a1, self._a0]
        wr = self._eig_val_real
        wi = self._eig_val_imag
        d_y = np.array([y[1],
                        -(a0(z) - wr) / a2(z) * y[0] - a1(z) / a2(z) * y[1] - wi / a2(z) * y[2],
                        y[3],
                        wi / a2(z) * y[0] - (a0(z) - wr) / a2(z) * y[2] - a1(z) / a2(z) * y[3]
                        ])
        return d_y

    def _transform_eigenfunction(self):

        eigenfunction = si.odeint(self._ff, self._init_state_vect, self._domain)

        return [eigenfunction[:, 0], eigenfunction[:, 1], eigenfunction[:, 2], eigenfunction[:, 3]]

    def _phi(self, z):
        return np.interp(z, self._domain, self._transf_eig_func_real)

    def _d_phi(self, z):
        return np.interp(z, self._domain, self._transf_d_eig_func_real)
