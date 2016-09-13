"""
This modules provides eigenfunctions for a certain set of parabolic problems. Therefore functions for the computation
of the corresponding eigenvalues are included.
The functions which compute the eigenvalues are deliberately separated from the predefined eigenfunctions in
order to handle transformations and reduce effort by the controller implementation.
"""

import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify
import scipy.integrate as si
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from . import utils as ut
from . import placeholder as ph
from .core import Function, back_project_from_base
from .shapefunctions import LagrangeFirstOrder, LagrangeSecondOrder
from .placeholder import FieldVariable, TestFunction
from .visualization import EvalData
from numbers import Number
from functools import partial
import warnings
import copy as cp
import collections
import pyqtgraph as pg
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod


class AddMulFunction(object):
    """
    (Temporary) Function class wich can multiplied with scalars and added with functions.
    Only needed to compute the matrix (of scalars) vector (of functions) product in
    :py:class:`FiniteTransformFunction`. Will be no longer needed when :py:class:`pyinduct.core.Function`
    is overloaded with :code:`__add__` and :code:`__mul__` operator.

    Args:
        function (callable):
    """

    def __init__(self, function):
        self.function = function

    def __call__(self, z):
        return self.function(z)

    def __mul__(self, other):
        return AddMulFunction(lambda z: self.function(z) * other)

    def __add__(self, other):
        return AddMulFunction(lambda z: self.function(z) + other(z))


class FiniteTransformFunction(Function):
    """
    Provide a transformed :py:class:`pyinduct.core.Function` :math:`\\bar x(z)` through the transformation
    :math:`\\bar{\\boldsymbol{\\xi}} = T * \\boldsymbol \\xi`,
    with the function vector :math:`\\boldsymbol \\xi\\in\\mathbb R^{2n}` and
    with a given matrix :math:`T\\in\\mathbb R^{2n\\times 2n}`.
    The operator :math:`*` denotes the matrix (of scalars) vector (of functions)
    product. The interim result :math:`\\bar{\\boldsymbol{\\xi}}` is a vector

    .. math:: \\bar{\\boldsymbol{\\xi}} = (\\bar\\xi_{1,0},...,\\bar\\xi_{1,n-1},\\bar\\xi_{2,0},...,\\bar\\xi_{2,n-1})^T.

    of functions

    .. math::
        &\\bar\\xi_{1,j} = \\bar x(jl_0 + z),\\qquad j=0,...,n-1, \\quad l_0=l/n, \\quad z\\in[0,l_0] \\\\
        &\\bar\\xi_{2,j} = \\bar x(l - jl_0 + z).

    Finally, the provided function :math:`\\bar x(z)` is given through :math:`\\bar\\xi_{1,0},...,\\bar\\xi_{1,n-1}`.

    Note:
        For a more extensive documentation see section 4.2 in:

        - Wang, S. und F. Woittennek: Backstepping-Methode für parabolische Systeme mit punktförmigem inneren
          Eingriff. Automatisierungstechnik, 2015.

          http://dx.doi.org/10.1515/auto-2015-0023

    Args:
        function (callable):
            Function :math:`x(z)` which will subdivided in :math:`2n` Functions

            .. math::
                &\\bar\\xi_{1,j} = x(jl_0 + z),\\qquad j=0,...,n-1, \\quad l_0=l/n, \\quad z\\in[0,l_0] \\\\
                &\\bar\\xi_{2,j} = x(l - jl_0 + z).

            The vector of functions :math:`\\boldsymbol\\xi` consist of these functions:

            .. math:: \\boldsymbol\\xi = (\\xi_{1,0},...,\\xi_{1,n-1},\\xi_{2,0},...,\\xi_{2,n-1})^T) .

        M (numpy.ndarray): Matrix :math:`T\\in\\mathbb R^{2n\\times 2n}` of scalars.
        l (numbers.Number): Length of the domain (:math:`z\in[0,l]`).
    """

    def __init__(self, function, M, l, scale_func=None, nested_lambda=False):

        if not isinstance(function, collections.Callable):
            raise TypeError
        if not isinstance(M, np.ndarray) or len(M.shape) != 2 or np.diff(M.shape) != 0 or M.shape[0] % 1 != 0:
            raise TypeError
        if not all([isinstance(num, (int, float)) for num in [l, ]]):
            raise TypeError

        self.function = function
        self.M = M
        self.l = l
        if scale_func == None:
            self.scale_func = lambda z: 1
        else:
            self.scale_func = scale_func

        self.n = int(M.shape[0] / 2)
        self.l0 = l / self.n
        self.z_disc = np.array([(i + 1) * self.l0 for i in range(self.n)])

        if not nested_lambda:
            # iteration mode
            Function.__init__(self, self._call_transformed_func, nonzero=(0, l), derivative_handles=[])
        else:
            # nested lambda mode
            self.x_func_vec = list()

            for i in range(self.n):
                self.x_func_vec.append(AddMulFunction(
                    partial(lambda z, k: self.scale_func(k * self.l0 + z) * self.function(k * self.l0 + z), k=i)))
            for i in range(self.n):
                self.x_func_vec.append(AddMulFunction(partial(
                    lambda z, k: self.scale_func(self.l - k * self.l0 - z) * self.function(self.l - k * self.l0 - z),
                    k=i)))

            self.y_func_vec = np.dot(self.x_func_vec, np.transpose(M))

            Function.__init__(self, self._call_transformed_func_vec, nonzero=(0, l), derivative_handles=[])

    def _call_transformed_func_vec(self, z):
        i = int(z / self.l0)
        zz = z % self.l0
        if np.isclose(z, self.l0 * i) and not np.isclose(0, zz):
            zz = 0
        return self.y_func_vec[i](zz)

    def _call_transformed_func(self, z):
        i = int(z / self.l0)
        if i < 0 or i > self.n * 2 - 1:
            raise ValueError
        zz = z % self.l0
        if np.isclose(z, self.l0 * i) and not np.isclose(0, zz):
            zz = 0
        to_return = 0
        for j in range(self.n * 2):
            mat_el = self.M[i, j]
            if mat_el != 0:
                if j <= self.n - 1:
                    to_return += mat_el * self.function(j * self.l0 + zz) * self.scale_func(j * self.l0 + zz)
                elif j >= self.n:
                    jj = j - self.n
                    to_return += mat_el * self.function(self.l - jj * self.l0 - zz) * self.scale_func(
                        self.l - jj * self.l0 - zz)
                elif j < 0 or j > 2 * self.n - 1:
                    raise ValueError
        return to_return


class TransformedSecondOrderEigenfunction(Function):
    """
    Provide the eigenfunction :math:`\\varphi(z)` to an eigenvalue problem of the form

    .. math:: a_2(z)\\varphi''(z) + a_1(z)\\varphi'(z) + a_0(z)\\varphi(z) = \\lambda\\varphi(z)

    where :math:`\\lambda` is a predefined (potentially complex) eigenvalue and :math:`[z_0,z_1]\\ni z` is the domain.

    Args:
        target_eigenvalue (numbers.Number): :math:`\\lambda`
        init_state_vect (array_like):
            .. math:: \\Big(\\text{Re}\\{\\varphi(0)\\}, \\text{Re}\\{\\varphi'(0)\\}, \\text{Im}\\{\\varphi(0)\\}, \\text{Im}\\{\\varphi'(0)\\}\\Big)^T
        dgl_coefficients (array_like):
            :math:`\\Big( a2(z), a1(z), a0(z) \\Big)^T`
        domain (array_like):
            :math:`\\Big( z_0, ..... , z_1 \\Big)`
    """

    def __init__(self, target_eigenvalue, init_state_vect, dgl_coefficients, domain):

        if not all([isinstance(state, (int, float)) for state in init_state_vect]) and len(
            init_state_vect) == 4 and isinstance(init_state_vect, (list, tuple)):
            raise TypeError
        if not len(dgl_coefficients) == 3 and isinstance(dgl_coefficients, (list, tuple)) and all(
            [isinstance(coef, collections.Callable) or isinstance(coef, (int, float)) for coef in dgl_coefficients]):
            raise TypeError
        if not isinstance(domain, (np.ndarray, list)) or not all([isinstance(num, (int, float)) for num in domain]):
            raise TypeError

        if isinstance(target_eigenvalue, complex):
            self._eig_val_real = target_eigenvalue.real
            self._eig_val_imag = target_eigenvalue.imag
        elif isinstance(target_eigenvalue, (int, float)):
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
        d_y = np.array([y[1], -(a0(z) - wr) / a2(z) * y[0] - a1(z) / a2(z) * y[1] - wi / a2(z) * y[2], y[3],
                        wi / a2(z) * y[0] - (a0(z) - wr) / a2(z) * y[2] - a1(z) / a2(z) * y[3]])
        return d_y

    def _transform_eigenfunction(self):

        eigenfunction = si.odeint(self._ff, self._init_state_vect, self._domain)

        return [eigenfunction[:, 0], eigenfunction[:, 1], eigenfunction[:, 2], eigenfunction[:, 3]]

    def _phi(self, z):
        return np.interp(z, self._domain, self._transf_eig_func_real)

    def _d_phi(self, z):
        return np.interp(z, self._domain, self._transf_d_eig_func_real)


class LambdifiedSympyExpression(Function):
    """
    Provide a :py:class:`pyinduct.core.Function` :math:`\\varphi(z)` based of a lambdified sympy expression.
    The sympy expression must be provided as first element of the list *sympy_funcs*. In the subsequent elements
    of the list, the sympy expressions for the derivatives of the function take place (with increasing order).

    Args:
        sympy_funcs (array_like): Sympy expressions for the function and the derivatives: :math:`\\varphi(z), \\varphi'(z), ...`.
        z: Sympy symbol for :math:`z`.
        spatial_domain (tuple): Domain on which :math:`\\varphi(z)` is defined (e.g.: :code:`spatial_domain=(0, 1)`).
    """

    def __init__(self, sympy_funcs, z, spatial_domain):
        self._funcs = [lambdify(z, sp_func, 'numpy') for sp_func in sympy_funcs]
        funcs = [self._func_factory(der_ord) for der_ord in range(len(sympy_funcs))]
        Function.__init__(self, funcs[0], nonzero=spatial_domain, derivative_handles=funcs[1:])

    def _func_factory(self, der_order):
        func = self._funcs[der_order]

        def function(z):
            return return_real_part(func(z))

        return function


class SecondOrderEigenfunction(metaclass=ABCMeta):
    """
    Wrapper for all eigenvalue problems from the form

    .. math:: a_2\\varphi''(z) + a_1&\\varphi'(z) + a_0\\varphi(z) = \\lambda\\varphi(z), \\qquad a_2, a_1, a_0, \\lambda \\in \\mathbb R

    with eigenfunctions :math:`\\varphi` and eigenvalues :math:`\\lambda`.
    The roots of the characteristic equation (of the dgl) denoted by

    .. math:: p = \\eta \\pm j\\omega, \\qquad \\eta \\in \\mathbb R, \\quad \\omega \\in \\mathbb C

    .. math:: \\eta = -\\frac{a_1}{2a_2}, \\quad \\omega = \\sqrt{-\\frac{a_1^2}{4 a_2^2} + \\frac{a_0 - \\lambda}{a_2}}

    In the following the variable :math:`\\omega` is called as eigenfrequency.
    """

    @abstractmethod
    def eigfreq_eigval_hint(self):
        """
        Returns:
            tuple: Booth tuple elements are numpy.ndarrays of the same length, one for eigenfrequencies and one for eigenvalues:
                :math:`\\Big(\\big[\\omega_1,...,\\omega_\\text{n\\_roots}\Big], \\Big[\\lambda_1,...,\\lambda_\\text{n\\_roots}\\big]\\Big)`
        """

    @staticmethod
    def eigval_tf_eigfreq(param, eig_val=None, eig_freq=None):
        """
        Calculate/Provide a list of eigenvalues to/from a list of eigenfrequencies with

        .. math:: \\omega = \\sqrt{-\\frac{a_1^2}{4a_2^2}+\\frac{a_0-\\lambda}{a_2}}

        respectively

        .. math:: \\lambda = -\\frac{a_1^2}{4a_2}+a_0 - a_2 \\omega.

        Args:
            param (array_like): Parameters :math:`(a_2, a_1, a_0, None, None)`.
            eig_val (array_like): Eigenvalues :math:`\\lambda`.
            eig_freq (array_like): Eigenfrequencies :math:`\\omega`.

        Returns:
            numpy.array: Eigenfrequencies :math:`\\omega` or eigenvalues :math:`\\lambda`.
        """
        a2, a1, a0, _, _ = param

        if eig_freq is not None and eig_val is not None:
            raise ValueError("You can pass:\n"
                             "\t - eigenvalues through eig_val or\n"
                             "\t - eigenfrequencies through eig_freq\n"
                             "\t - but not booth.\n"
                             "")
        elif eig_val is not None:
            return np.sqrt(-a1 ** 2 / 4 / a2 ** 2 + (a0 - np.array(eig_val, dtype=complex)) / a2)
        elif eig_freq is not None:
            return -a1 ** 2 / 4 / a2 + a0 - a2 * np.array(eig_freq, dtype=complex) ** 2

    @staticmethod
    def get_adjoint_problem(param):
        """
        Return to the considered eigenvalue problem with dirichlet or robin boundary condition by :math:`z=0`

        .. math:: \\varphi(0) = 0 \\quad &\\text{or} \\quad \\varphi'(0) = \\alpha\\varphi(0)

        and dirichlet or robin boundary condition by :math:`z=l`

        .. math:: \\varphi`(l) = 0 \\quad &\\text{or} \\quad \\varphi'(l) = -\\beta\\varphi(l)

        the parameters for the adjoint problem (with the same structure).

        Args:
            param (array_like): Set alpha/beta to None if you have a dirichlet boundary condition on this point. Possible:
                - :math:`\\Big( a_2, a_1, a_0, \\alpha, \\beta \\Big)^T`,
                - :math:`\\Big( a_2, a_1, a_0, None, \\beta \\Big)^T`,
                - :math:`\\Big( a_2, a_1, a_0, \\alpha, None \\Big)^T` or
                - :math:`\\Big( a_2, a_1, a_0, None, None \\Big)^T`.

        Return:
            tuple:
                Parameters :math:`\\big(a_2, \\tilde a_1=-a_1, a_0, \\tilde \\alpha, \\tilde \\beta \\big)` for
                the adjoint problem

                .. math::
                    a_2\\psi''(z) + a_1&\\psi'(z) + a_0\\psi(z) = \\lambda\\psi(z) \\\\
                    \\psi(0) = 0 \\quad &\\text{or} \\quad \\psi'(0) = \\tilde\\alpha\\psi(0) \\\\
                    \\psi`(l) = 0 \\quad &\\text{or} \\quad \\psi'(l) = -\\tilde\\beta\\psi(l).
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

    @classmethod
    def solve_evp_hint(evp_class, param, l, n=None, eig_val=None, eig_freq=None, max_order=2, scale=None):
        """
        Provide the first *n* eigenvalues and eigenfunctions. For the exact formulation of
        the considered eigenvalue problem, have a look at the docstring from the eigenfunction
        class from which you will call this method.

        Args:
            param: Parameters :math:`(a_2, a_1, a_0, ...)` see *evp_class.__doc__*.
            l: End of the domain from the eigenfunctions (start is 0).
            n: Number of eigenvalues/eigenfunctions to be compute.
            eig_freq (array_like): Pass you own choice of eigenfrequencies here.
            eig_val (array_like): Pass you own choice of eigenvalues here.
            max_order: Maximum derivative order which must provided from the eigenfunctions.
            scale (array_like): Here you can pass a list of values to scale the eigenfunctions.

        Returns:
            Tuple with one list for the eigenvalues and one for the eigenfunctions.
        """
        if np.sum([1 for arg in [n, eig_val, eig_freq] if arg is not None]) != 1:
            raise ValueError("You must pass one and only one of the kwargs:\n"
                             "\t - n (Number of eigenvalues/eigenfunctions to be compute)\n"
                             "\t - eig_val (Eigenvalues)\n"
                             "\t - eig_freq (Eigenfrequencies).\n"
                             "")
        elif eig_val is not None:
            eig_freq = evp_class.eigval_tf_eigfreq(param, eig_val=eig_val)
        elif eig_freq is not None:
            eig_val = evp_class.eigval_tf_eigfreq(param, eig_freq=eig_freq)
        else:
            eig_freq, eig_val = evp_class.eigfreq_eigval_hint(param, l, n)

        if scale is None:
            scale = np.ones(eig_freq.shape)

        eig_func = np.array([evp_class(om, param, (0, l), scale=sc, max_der_order=max_order)
                             for om, sc in zip(np.array(eig_freq, dtype=complex), scale)])

        return np.array(eig_val, dtype=complex), eig_func


class SecondOrderRobinEigenfunction(Function, SecondOrderEigenfunction):
    """
    Provide the eigenfunction :math:`\\varphi(z)` to an eigenvalue problem of the form

    .. math::
        a_2\\varphi''(z) + a_1&\\varphi'(z) + a_0\\varphi(z) = \\lambda\\varphi(z) \\\\
        \\varphi'(0) &= \\alpha \\varphi(0) \\\\
        \\varphi'(l) &= -\\beta \\varphi(l).

    The eigenfrequency

    .. math:: \\omega = \\sqrt{-\\frac{a_1^2}{4a_2^2}+\\frac{a_0-\\lambda}{a_2}}

    must be provided (for example with the :py:func:`eigfreq_eigval_hint` of this class).

    Args:
        om (numbers.Number): eigenfrequency :math:`\\omega`
        param (array_like): :math:`\\Big( a_2, a_1, a_0, \\alpha, \\beta \\Big)^T`
        spatial_domain (tuple): Start point :math:`z_0` and end point :math:`z_1` of
            the spatial domain :math:`[z_0,z_1]\\ni z`.
        scale (numbers.Number): Factor to scale the eigenfunctions (correspond :math:`\\varphi(0)=\\text{phi\\_0}`).
        max_der_order (int): Number of derivative handles that are needed.
    """

    def __init__(self, om, param, spatial_domain, scale=1, max_der_order=2):
        self._om = om
        self._param = param
        self._norm_fac = scale
        self._max_der_order = max_der_order

        self._om_is_close = np.isclose(self._om, 0)
        a2_, a1_, a0_, alpha_, beta_ = self._param
        eta_ = - a1_ / a2_ / 2
        l_ = spatial_domain[1]

        alpha, beta, eta, omega, varphi_0, z, c1, c2, c3, c4, l = sp.symbols(
            "alpha beta eta omega varphi_0 z c1 c2 c3 c4 l")
        subs_list = [(varphi_0, scale), (eta, eta_), (omega, om), (alpha, alpha_), (beta, beta_), (l, l_)]

        if om == 0:
            phi = c2 * sp.exp(eta * z) + c1 * z * sp.exp(eta * z)

        else:
            phi = sp.exp(eta * z) * (c1 * sp.sin(omega * z) + c2 * sp.cos(omega * z))

        eq = phi.diff(z).subs(z, 0) - alpha * varphi_0
        c1_ = list(sp.linsolve([eq.subs(c2, varphi_0)], (c1)))[0][0]
        c2_ = varphi_0
        sp_funcs = [phi.subs([(c1, c1_), (c2, c2_)]).subs(subs_list)]

        for _ in np.arange(max_der_order):
            sp_funcs.append(sp_funcs[-1].diff(z))
        self._funcs = LambdifiedSympyExpression(sp_funcs, z, spatial_domain)

        zero_limit_sp_funcs = [sp.limit(sp_func, omega, 0) for sp_func in sp_funcs]
        self._zero_limit_funcs = LambdifiedSympyExpression(zero_limit_sp_funcs, z, spatial_domain)

        funcs = [self._eig_func_factory(der_ord) for der_ord in range(max_der_order + 1)]
        Function.__init__(self, funcs[0], nonzero=spatial_domain, derivative_handles=funcs[1:])

    def _eig_func_factory(self, der_order):
        om_is_close = self._om_is_close
        func = self._funcs.derive(der_order)
        zero_limit_func = self._zero_limit_funcs.derive(der_order)

        def eigenfunction(z):
            if om_is_close:
                res = zero_limit_func(z)
            else:
                res = func(z)
            return return_real_part(res)

        return eigenfunction

    @staticmethod
    def eigfreq_eigval_hint(param, l, n_roots, show_plot=False):
        """
        Return the first *n_roots* eigenfrequencies :math:`\\omega` and eigenvalues :math:`\\lambda`.

        .. math:: \\omega_i = \\sqrt{-\\frac{a_1^2}{4a_2^2}+\\frac{a_0-\\lambda_i}{a_2}} \\quad i = 1,...,\\text{n\\_roots}

        to the considered eigenvalue problem.

        Args:
            param (array_like): :math:`\\Big( a_2, a_1, a_0, \\alpha, \\beta \\Big)^T`
            l (numbers.Number): Right boundary value of the domain :math:`[0,l]\\ni z`.
            n_roots (int): Amount of eigenfrequencies to be compute.
            show_plot (bool): A plot window of the characteristic equation appears if it is :code:`True`.

        Return:
            tuple --> booth tuple elements are numpy.ndarrays of length *nroots*:
                :math:`\\Big(\\big[\\omega_1,...,\\omega_\\text{n\\_roots}\Big], \\Big[\\lambda_1,...,\\lambda_\\text{n\\_roots}\\big]\\Big)`
        """

        a2, a1, a0, alpha, beta = param
        eta = -a1 / 2. / a2

        # characteristic equations for eigenvectors: phi = c1 e^(eta z) + c2 z e^(eta z)
        char_eq = np.polynomial.Polynomial([alpha * beta * l + alpha + beta, alpha * l - beta * l, -l])

        # characteristic equations for eigenvectors: phi = e^(eta z) (c1 cos(om z) + c2 sin(om z))
        def characteristic_equation(om):
            if np.isclose(om, 0):
                return (alpha + beta) * np.cos(om * l) + (eta + beta) * (alpha - eta) * l - om * np.sin(om * l)
            else:
                return (alpha + beta) * np.cos(om * l) + ((eta + beta) * (alpha - eta) / om - om) * np.sin(om * l)

        if show_plot:
            z_real = np.linspace(-15, 15)
            z_imag = np.linspace(-5, 5)
            vec_function = np.vectorize(characteristic_equation)
            plt.plot(z_real, vec_function(z_real))
            plt.plot(z_imag, vec_function(z_imag * 1j))
            plt.plot(z_real, char_eq(z_real))
            plt.show()

        # assume 1 root per pi/l (safety factor = 3)
        search_begin = np.pi / l * .1
        search_end = 3 * n_roots * np.pi / l
        start_values_real = np.linspace(search_begin, search_end, search_end / np.pi * l * 100)
        start_values_imag = np.linspace(search_begin, search_end, search_end / np.pi * l * 20)

        # search imaginary roots
        try:
            om = list(ut.find_roots(characteristic_equation, 100, [np.array([0]), start_values_imag],
                                    rtol=int(np.log10(l) - 3), complex=True, show_plot=show_plot, get_all=True))
        except ValueError:
            om = list()

        # search real roots
        om += ut.find_roots(characteristic_equation, 2 * n_roots, [start_values_real, np.array([0])],
                            rtol=int(np.log10(l) - 3), complex=True, show_plot=show_plot).tolist()

        # only "real" roots and complex roots with imaginary part != 0 and real part == 0 considered
        if any([not np.isclose(root.real, 0) and not np.isclose(root.imag, 0) for root in om]):
            raise NotImplementedError("This case is currently not considered.")

        # read out complex roots
        _complex_roots = [root for root in om if np.isclose(root.real, 0) and not np.isclose(root.imag, 0)]
        complex_roots = list()
        for complex_root in _complex_roots:
            if not any([np.isclose(np.abs(complex_root), _complex_root) for _complex_root in complex_roots]):
                complex_roots.append(complex_root)

        # sort out all complex roots and roots with negative real part
        om = [root.real + 0j for root in om if root.real >= 0 and np.isclose(root.imag, 0)]

        # delete all around om = 0
        for i in [ind for ind, val in enumerate(np.isclose(np.array(om), 0, atol=1e-4)) if val]:
            om.pop(i)

        # if om = 0 is a root and the corresponding characteristic equation is satisfied then add 0 to the list
        if np.isclose(np.abs(characteristic_equation(0)), 0) and any(np.isclose(char_eq.roots(), 0)):
            om.insert(0, 0)

        # add complex root
        for complex_root in complex_roots:
            om.insert(0, complex_root)

        if len(om) < n_roots:
            raise ValueError("RadRobinEigenvalues.compute_eigen_frequencies()"
                             "can not find enough roots")

        eig_frequencies = np.array(om[:n_roots])
        eig_values = a0 - a2 * eig_frequencies ** 2 - a1 ** 2 / 4. / a2

        return eig_frequencies, eig_values


class SecondOrderDirichletEigenfunction(LambdifiedSympyExpression, SecondOrderEigenfunction):
    """
    Provide the eigenfunction :math:`\\varphi(z)` to an eigenvalue problem of the form

    .. math::
        a_2\\varphi''(z) + a_1&\\varphi'(z) + a_0\\varphi(z) = \\lambda\\varphi(z) \\\\
        \\varphi(0) &= 0 \\\\
        \\varphi(l) &= 0.

    The eigenfrequency

    .. math:: \\omega = \\sqrt{-\\frac{a_1^2}{4a_2^2}+\\frac{a_0-\\lambda}{a_2}}

    must be provided.

    Args:
        om (numbers.Number): eigenfrequency :math:`\\omega`
        param (array_like): :math:`\\Big( a_2, a_1, a_0, None, None \\Big)^T`
        spatial_domain (tuple): Start point :math:`z_0` and end point :math:`z_1` of
            the spatial domain :math:`[z_0,z_1]\\ni z`.
        scale (numbers.Number): Factor to scale the eigenfunctions.
        max_der_order (int): Number of derivative handles that are needed.
    """

    def __init__(self, om, param, spatial_domain, scale=1, max_der_order=2):
        self._om = om
        self._param = param
        self._norm_fac = scale
        self._max_der_order = max_der_order

        a2_, a1_, a0_, _, _ = self._param
        eta_ = -a1_ / 2. / a2_

        eta, omega, scale_, z = sp.symbols("eta omega scale_ z")
        subs_list = [(scale_, scale), (eta, eta_), (omega, om)]
        sp_funcs = [(scale * sp.exp(eta * z) * sp.sin(omega * z)).subs(subs_list)]
        for _ in np.arange(max_der_order):
            sp_funcs.append(sp_funcs[-1].diff(z))

        LambdifiedSympyExpression.__init__(self, sp_funcs, z, spatial_domain)

    @staticmethod
    def eigfreq_eigval_hint(param, l, n_roots):
        """
        Return the first *n_roots* eigenfrequencies :math:`\\omega` and eigenvalues :math:`\\lambda`.

        .. math:: \\omega_i = \\sqrt{-\\frac{a_1^2}{4a_2^2}+\\frac{a_0-\\lambda_i}{a_2}} \\quad i = 1,...,\\text{n\\_roots}

        to the considered eigenvalue problem.

        Args:
            l (numbers.Number): Right boundary value of the domain :math:`[0,l]\\ni z`.
            n_roots (int): Amount of eigenfrequencies to be compute.

        Return:
            tuple --> booth tuple elements are numpy.ndarrays of length *n_roots*:
                :math:`\\Big(\\big[\\omega_1,...,\\omega_\\text{n\\_roots}\Big], \\Big[\\lambda_1,...,\\lambda_\\text{n\\_roots}\\big]\\Big)`
        """
        a2, a1, a0, _, _ = param
        eig_frequencies = np.array([i * np.pi / l for i in np.arange(1, n_roots + 1)])
        eig_values = a0 - a2 * eig_frequencies ** 2 - a1 ** 2 / 4. / a2

        return eig_frequencies, eig_values


def return_real_part(to_return):
    """
    Check if the imaginary part of :code:`to_return` vanishes
    and return the real part.

    Args:
        to_return (numbers.Number or array_like): Variable to check.

    Raises:
        ValueError: If (all) imaginary part(s) not vanishes.

    Return:
        numbers.Number or array_like: Real part of :code:`to_return`.
    """
    if not isinstance(to_return, (Number, list, np.ndarray)):
        raise TypeError
    if isinstance(to_return, (list, np.ndarray)):
        if not all([isinstance(num, Number) for num in to_return]):
            raise TypeError

    maybe_real = np.atleast_1d(np.real_if_close(to_return))

    if maybe_real.dtype == 'complex':
        raise ValueError("Something goes wrong, imaginary part does not vanish")
    else:
        if maybe_real.shape == (1,):
            maybe_real = maybe_real[0]
        return maybe_real


def transform2intermediate(param: object, d_end: object = None) -> object:
    """
    Transformation :math:`\\tilde x(z,t)=x(z,t)e^{\\int_0^z \\frac{a_1(\\bar z)}{2 a_2}\,d\\bar z}`
    which eliminate the advection term :math:`a_1 x(z,t)` from the
    reaction-advection-diffusion equation

    .. math:: \\dot x(z,t) = a_2 x''(z,t) + a_1(z) x'(z,t) + a_0(z) x(z,t)

    with robin

    .. math:: x'(0,t) = \\alpha x(0,t), \\quad x'(l,t) = -\\beta x(l,t)

    or dirichlet

    .. math:: x(0,t) = 0, \\quad x(l,t) = 0

    or mixed boundary condition.

    Args:
        param (array_like): :math:`\\Big( a_2, a_1, a_0, \\alpha, \\beta \\Big)^T`

    Raises:
        TypeError: If :math:`a_1(z)` is callable but no derivative handle is defined for it.

    Return:
        tuple:
            Parameters :math:`\\big(a_2, \\tilde a_1=0, \\tilde a_0(z), \\tilde \\alpha, \\tilde \\beta \\big)` for
            the transformed system

            .. math:: \\dot{\\tilde{x}}(z,t) = a_2 \\tilde x''(z,t) + \\tilde a_0(z) \\tilde x(z,t)

            and the corresponding boundary conditions (:math:`\\alpha` and/or :math:`\\beta` set to None by dirichlet
            boundary condition).

    """
    if not isinstance(param, (tuple, list)) or not len(param) == 5:
        raise TypeError("pyinduct.utils.transform_2_intermediate(): argument param must from type tuple or list")

    a2, a1, a0, alpha, beta = param
    if isinstance(a1, collections.Callable) or isinstance(a0, collections.Callable):
        if not len(a1._derivative_handles) >= 1:
            raise TypeError
        a0_z = ut._convert_to_function(a0)
        a0_n = lambda z: a0_z(z) - a1(z) ** 2 / 4 / a2 - a1.derive(1)(z) / 2
    else:
        a0_n = a0 - a1 ** 2 / 4 / a2

    if alpha is None:
        alpha_n = None
    elif isinstance(a1, collections.Callable):
        alpha_n = a1(0) / 2. / a2 + alpha
    else:
        alpha_n = a1 / 2. / a2 + alpha

    if beta is None:
        beta_n = None
    elif isinstance(a1, collections.Callable):
        beta_n = -a1(d_end) / 2. / a2 + beta
    else:
        beta_n = -a1 / 2. / a2 + beta

    a2_n = a2
    a1_n = 0

    return a2_n, a1_n, a0_n, alpha_n, beta_n

def eigfreq_eigval_hint():
    pass
