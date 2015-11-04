from __future__ import division
import numpy as np
import scipy.integrate as si
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from pyinduct import get_initial_functions, register_functions
import pyinduct
import utils as ut
import placeholder as ph
from core import Function, back_project_from_base
from shapefunctions import LagrangeFirstOrder, LagrangeSecondOrder
from placeholder import FieldVariable, TestFunction
from visualization import EvalData
import pyqtgraph as pg
from numbers import Number
from functools import partial
import warnings
import copy as cp
import pyqtgraph as pg


__author__ = 'marcus'


class AddMulFunction(object):

    def __init__(self, function):
        self.function = function

    def __call__(self, z):
        return self.function(z)

    def __mul__(self, other):
        return AddMulFunction(lambda z: self.function(z)*other)

    def __add__(self, other):
        return AddMulFunction(lambda z: self.function(z) + other(z))


class FiniteTransformFunction(Function):
    """
    Provide a transformed function y(z) = T x(z) for a given matrix T and function y(z)
    """
    def __init__(self, function, M, b, l, scale_func=None, nested_lambda=False):

        if not callable(function):
            raise TypeError
        if not isinstance(M, np.ndarray) or len(M.shape) != 2 or np.diff(M.shape) != 0 or M.shape[0]%1 != 0:
            raise TypeError
        if not all([isinstance(num, (int, long, float)) for num in [b, l]]):
            raise TypeError

        self.function = function
        self.M = M
        self.b = b
        self.l = l
        if scale_func == None:
            self.scale_func = lambda z: 1
        else:
            self.scale_func = scale_func

        self.n = int(M.shape[0]/2)
        self.l0 = l/self.n
        self.z_disc = np.array([(i+1)*self.l0 for i in range(self.n)])

        if not nested_lambda:
            # iteration mode
            Function.__init__(self,
                              self._call_transformed_func,
                              nonzero=(0, l),
                              derivative_handles=[])
        else:
            # nested lambda mode
            self.x_func_vec = list()

            for i in range(self.n):
                self.x_func_vec.append(AddMulFunction(
                    partial(lambda z, k: self.scale_func(k*self.l0 + z)*self.function(k*self.l0 + z), k=i)))
            for i in range(self.n):
                self.x_func_vec.append(AddMulFunction(
                    partial(lambda z, k: self.scale_func(self.l - k*self.l0 - z)*self.function(self.l - k*self.l0 - z), k=i)))

            self.y_func_vec = np.dot(self.x_func_vec, np.transpose(M))

            Function.__init__(self,
                              self._call_transformed_func_vec,
                              nonzero=(0, l),
                              derivative_handles=[])

    def _call_transformed_func_vec(self, z):
        i = int(z/self.l0)
        zz = z%self.l0
        if np.isclose(z, self.l0*i) and not np.isclose(0, zz):
            zz = 0
        return self.y_func_vec[i](zz)

    def _call_transformed_func(self, z):
        i = int(z/self.l0)
        if i < 0 or i > self.n*2-1:
            raise ValueError
        zz = z%self.l0
        if np.isclose(z, self.l0*i) and not np.isclose(0, zz):
            zz = 0
        to_return = 0
        for j in range(self.n*2):
            mat_el = self.M[i, j]
            if mat_el != 0:
                if j <= self.n-1:
                    to_return += mat_el*self.function(j*self.l0 + zz) * self.scale_func(j*self.l0 + zz)
                elif j >= self.n:
                    jj = j-self.n
                    to_return += mat_el*self.function(self.l - jj*self.l0 - zz) * self.scale_func(self.l - jj*self.l0 - zz)
                elif j < 0 or j > 2*self.n-1:
                    raise ValueError
        return to_return


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


class SecondOrderRobinEigenfunction(Function):
    def __init__(self, om, param, spatial_domain, phi_0=1):
        self._om = om
        self._param = param
        self.phi_0 = phi_0
        Function.__init__(self, self._phi, nonzero=spatial_domain, derivative_handles=[self._d_phi, self._dd_phi])

    def _phi(self, z):
        a2, a1, a0, alpha, beta = self._param
        om = self._om
        eta = -a1 / 2. / a2

        cosX_term = np.cos(om * z)
        if not np.isclose(0, np.abs(om), atol=1e-100):
            sinX_term = (alpha - eta) / om * np.sin(om * z)
        else:
            sinX_term = (alpha - eta) * z

        phi_i = np.exp(eta * z) * (cosX_term + sinX_term)

        return return_real_part(phi_i*self.phi_0)

    def _d_phi(self, z):
        a2, a1, a0, alpha, beta = self._param
        om = self._om
        eta = -a1 / 2. / a2

        cosX_term = alpha * np.cos(om * z)
        if not np.isclose(0, np.abs(om), atol=1e-100):
            sinX_term = (eta * (alpha - eta) / om - om) * np.sin(om * z)
        else:
            sinX_term = eta * (alpha - eta) * z - om * np.sin(om * z)

        d_phi_i = np.exp(eta * z) * (cosX_term + sinX_term)

        return return_real_part(d_phi_i*self.phi_0)

    def _dd_phi(self, z):
        a2, a1, a0, alpha, beta = self._param
        om = self._om
        eta = -a1 / 2. / a2

        cosX_term = (eta*(2*alpha-eta)-om**2) * np.cos(om * z)
        if not np.isclose(0, np.abs(om), atol=1e-100):
            sinX_term = ((eta**2 * (alpha - eta) / om - (eta+alpha)*om)) * np.sin(om * z)
        else:
            sinX_term = eta**2 * (alpha - eta) * z - (eta+alpha)*om * np.sin(om * z)

        d_phi_i = np.exp(eta * z) * (cosX_term + sinX_term)

        return return_real_part(d_phi_i*self.phi_0)


class SecondOrderDirichletEigenfunction(Function):
    def __init__(self, omega, param, spatial_domain, norm_fac=1.):
        self._omega = omega
        self._param = param
        self.norm_fac = norm_fac

        a2, a1, a0, _, _ = self._param
        self._eta = -a1 / 2. / a2
        Function.__init__(self, self._phi, nonzero=spatial_domain, derivative_handles=[self._d_phi, self._dd_phi])

    def _phi(self, z):
        eta = self._eta
        om = self._omega

        phi_i = np.exp(eta * z) * np.sin(om * z)

        return return_real_part(phi_i * self.norm_fac)

    def _d_phi(self, z):
        eta = self._eta
        om = self._omega

        d_phi_i = np.exp(eta * z) * (om * np.cos(om * z) + eta * np.sin(om * z))

        return return_real_part(d_phi_i * self.norm_fac)

    def _dd_phi(self, z):
        eta = self._eta
        om = self._omega

        d_phi_i = np.exp(eta * z) * (om * (eta + 1) * np.cos(om * z) + (eta - om ** 2) * np.sin(om * z))

        return return_real_part(d_phi_i * self.norm_fac)

def compute_rad_robin_eigenfrequencies(param, l, n_roots=10, show_plot=False):

    a2, a1, a0, alpha, beta = param
    eta = -a1 / 2. / a2

    def characteristic_equation(om):
        if round(om, 200) != 0.:
            zero = (alpha + beta) * np.cos(om * l) + ((eta + beta) * (alpha - eta) / om - om) * np.sin(om * l)
        else:
            zero = (alpha + beta) * np.cos(om * l) + (eta + beta) * (alpha - eta) * l - om * np.sin(om * l)
        return zero

    def complex_characteristic_equation(om):
        if round(om, 200) != 0.:
            zero = (alpha + beta) * np.cosh(om * l) + ((eta + beta) * (alpha - eta) / om + om) * np.sinh(om * l)
        else:
            zero = (alpha + beta) * np.cosh(om * l) + (eta + beta) * (alpha - eta) * l + om * np.sinh(om * l)
        return zero

    # assume 1 root per pi/l (safety factor = 2)
    om_end = 2 * n_roots * np.pi / l
    om = ut.find_roots(characteristic_equation, 2 * n_roots, om_end, rtol=int(np.log10(l)-6), show_plot=show_plot).tolist()

    # delete all around om = 0
    om.reverse()
    for i in xrange(np.sum(np.array(om) < np.pi / l / 2e1)):
        om.pop()
    om.reverse()

    # if om = 0 is a root then add 0 to the list
    zero_limit = alpha + beta + (eta + beta) * (alpha - eta) * l
    if round(zero_limit, 6 + int(np.log10(l))) == 0.:
        om.insert(0, 0.)

    # regard complex roots
    om_squared = np.power(om, 2).tolist()
    complex_root = fsolve(complex_characteristic_equation, om_end)
    if round(complex_root, 6 + int(np.log10(l))) != 0.:
        om_squared.insert(0, -complex_root[0] ** 2)

    # basically complex eigenfrequencies
    om = np.sqrt(np.array(om_squared).astype(complex))

    if len(om) < n_roots:
        raise ValueError("RadRobinEigenvalues.compute_eigen_frequencies()"
                         "can not find enough roots")


    eig_frequencies = om[:n_roots]
    eig_values = a0 - a2 * eig_frequencies**2 - a1 ** 2 / 4. / a2
    return eig_frequencies, eig_values

def return_real_part(to_return):
    """
    Check if the imaginary part of to_return vanishes
    and return the real part
    :param to_return:
    :return:
    """
    if not isinstance(to_return, (Number, list, np.ndarray)):
        raise TypeError
    if isinstance(to_return, (list, np.ndarray)):
        if not all([isinstance(num, Number) for num in to_return]):
            raise TypeError

    maybe_real = np.real_if_close(to_return)

    if maybe_real.dtype == 'complex':
        raise ValueError("Something goes wrong, imaginary part does not vanish")
    else:
        return maybe_real

def get_adjoint_rad_evp_param(param):
    """
    Return to the eigen value problem of the reaction-advection-diffusion
    equation with robin and/or dirichlet boundary conditions
    a2 y''(z) + a1 y'(z) + a0 y(z) = w y(z)
    y'(0) = alpha y(0) / y(0) = 0, y'(l) = -beta y(l) / y(l) = 0
    the parameters for the adjoint Problem (with the same structure).
    """
    a2, a1, a0, alpha, beta = param

    if alpha == None:
        alpha_n = None
    else:
        alpha_n = a1 / a2 + alpha

    if beta == None:
        beta_n = None
    else:
        beta_n = -a1 / a2 + beta
    a1_n = -a1

    return a2, a1_n, a0, alpha_n, beta_n

def transform2intermediate(param, d_end=None):
    """
    Transformation which eliminate the advection term 'a1 x(z,t)' from the
    reaction-advection-diffusion equation
    d/dt x(z,t) = a2 x''(z,t) + a1(z) x'(z,t) + a0(z) x(z,t)
    with robin
    x'(0,t) = alpha x(0,t), x'(l,t) = -beta x(l,t)
    or dirichlet
    x(0,t) = 0, x(l,t) = 0
    or mixed boundary condition.
    """
    if not isinstance(param, (tuple, list)) or not len(param) == 5:
        raise TypeError("pyinduct.utils.transform_2_intermediate(): argument param must from type tuple or list")

    a2, a1, a0, alpha, beta = param
    if callable(a1) or callable(a0):
        if not len(a1._derivative_handles) >= 1:
            raise TypeError
        a0_z = ut._convert_to_function(a0)
        a0_n = lambda z: a0_z(z) - a1(z)**2/4/a2 - a1.derive(1)(z)/2
    else:
        a0_n = a0 - a1**2/4/a2

    if alpha is None:
        alpha_n = None
    elif callable(a1):
        alpha_n = a1(0) / 2. / a2 + alpha
    else:
        alpha_n = a1 / 2. / a2 + alpha

    if beta is None:
        beta_n = None
    elif callable(a1):
        beta_n = -a1(d_end) / 2. / a2 + beta
    else:
        beta_n = -a1 / 2. / a2 + beta

    a2_n = a2
    a1_n = 0

    return a2_n, a1_n, a0_n, alpha_n, beta_n

