"""
This modules provides eigenfunctions for a certain set of parabolic problems. Therefore functions for the computation
of the corresponding eigenvalues are included.
The functions which compute the eigenvalues are deliberately separated from the predefined eigenfunctions in
order to handle transformations and reduce effort within the controller implementation.
"""

import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify
import scipy.integrate as si
from numbers import Number
from functools import partial
import collections
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractstaticmethod

from .core import Base, Function, normalize_base, find_roots, real
from .visualization import visualize_roots

__all__ = ["SecondOrderEigenVector", "SecondOrderDirichletEigenfunction"]


class SecondOrderEigenVector(Function):
    r"""
    This class provides the eigen vectors corresponding to a
    linear second order spatial operator denoted by
        :math:`(Ax)(z) = a_2 x''(z) + a_1x'(z) + a_0x(z)` .

    With the boundary conditions:
        :math:`\alpha_1 x'(z_1) + \alpha_0 x(z_1) = 0
        :math:`\beta_1 x'(z_2) + \beta_0 x(z_2) = 0 .

    The calculate the corresponding eigenvectors, the problem
        :math:`(Ax)(z) = \lambda x(z)
    is solved for the eigen values :math:`\lambda` .

    Note:
        To easily instantiate a set of eigenvectors for a certain
        system, use the :py:func:`cure_hint` of this class or even
        better the helper-function
        :py:func:`pyinduct.shapefunctions.cure_interval` from the
        :py:module:`shapefunction` module.

    Warn:
        Due to their algebraic multiplicity the eigen vectors for
        conjugate complex eigenvalue pairs are identical. Therefore
        pay attention to pass only one member of these pairs to
        obtain the orthonormal properties of the generated
        eigenvectors.

    Parameters:
        char_root (complex): Characteristic root, corresponding to the
            eigenvalue :math:`\lambda` for which the eigenvector is
            to be determined.
            (Can be obtained by :py:func:`convert_to_characteristic_root`)
        kappa (tuple): Constants of the exponential ansatz solution.

    """
    def __init__(self, char_root, kappa, domain, derivative_order=2):
        # build generic solution
        z, nu, eta, kappa1, kappa2 = sp.symbols("z nu eta kappa1 kappa2")
        gen_sols = [sp.exp(eta*z) * (kappa1*sp.cos(nu*z) + kappa2*sp.sin(nu*z))]

        gen_sols[0] = gen_sols[0].subs([(kappa1, kappa[0]),
                                        (kappa2, kappa[0]),
                                        (eta, np.real(char_root)),
                                        (nu, np.imag(char_root)),
                                        ])
        # derive
        for d in range(derivative_order + 1):
            gen_sols.append(gen_sols[-1].diff(z))

        # generate numeric handles
        num_handles = [sp.lambdify(z, sol, modules="numpy")
                       for sol in gen_sols]

        super().__init__(num_handles[0],
                         domain=domain,
                         derivative_handles=num_handles[1:])

    @staticmethod
    def cure_hint(domain, params, count, derivative_order, **kwargs):
        r"""
        Helper to cure an interval with eigenvectors.

        Parameters:
            domain (:py:class:`core.domain`): Domain of the
                spatial problem.
            params (bunch-like): Parameters of the system, see
                class docstring for details. Must somehow contain
                :math:`a_2, a_1, a_0, \alpha_0, \alpha_1, \beta_0, \beta_1` .
            count (int): Amount of eigenvectors to generate.
            derivative_order (int): Amount of derivative handles to provide.

        Keyword Arguments:
            debug (bool): If provided, this parameter will cause several debug
                windows to open.
        """
        if (params.alpha0 == 0 and params.alpha1 == 0
                or params.beta0 == 0 and params.beta1 == 0):
            raise ValueError("Provided boundary conditions are useless.")

        bounds = domain.bounds

        # again, build generic solution
        z, nu, eta, kappa1, kappa2 = sp.symbols("z nu eta kappa1 kappa2")
        gen_sol = sp.exp(eta*z) * (kappa1*sp.cos(nu*z) + kappa2*sp.sin(nu*z))
        gen_sol = gen_sol.subs([(eta, -params.a1/(2*params.a2))])

        kappa = np.zeros((count, 2))

        # check special case of dirichlet boundary, defined at zero
        if params.alpha1 == 0 and params.alpha0 != 0 and 0 in domain.bounds:
            # since kappa1 is equal to sol evaluated at z=0
            gen_sol = gen_sol.subs(kappa2, 1)
            kappa[:, 0] = 0
            # kappa2 is the arbitrary scaling factor
            gen_sol = gen_sol.subs(kappa1, 0)
            kappa[:, 1] = 1
            settled_bc = domain.bounds.index(0)

        else:
            # choose the arbitrary scaling to be one
            gen_sol = gen_sol.subs(kappa1, 1)
            kappa[:, 0] = 1

            # incorporate the first boundary condition
            bc1 = (params.alpha0 * gen_sol.subs(z, bounds[0])
                   + params.alpha1 * gen_sol.diff(z).subs(z, bounds[0]))
            kappa2_sol = sp.solve(bc1, kappa2)[0]
            gen_sol = gen_sol.subs(kappa2, kappa2_sol)
            settled_bc = 0

        # resolve 2nd boundary condition to extract char. function
        if settled_bc == 1:
            c0 = params.alpha0
            c1 = params.alpha1
        elif settled_bc == 0:
            c0 = params.beta0
            c1 = params.beta1
        else:
            raise ValueError

        char_eq = (c0 * gen_sol.subs(z, bounds[1 - settled_bc])
                   + c1 * gen_sol.diff(z).subs(z, bounds[1 - settled_bc]))

        if kwargs.get("debug", False):
            sp.init_printing()
            print("characteristic equation:")
            sp.pretty_print(char_eq)
            sp.plot(char_eq)

        # lambdify
        char_num = sp.lambdify(nu, char_eq, modules="numpy")

        def char_func(_z):
            """
            Characteristic function of the spatial eigenvalue problem.
            If the limit exists it is used to lift poles of the
            function.
            """
            try:
                return char_num(_z)
            except FloatingPointError:
                lim_p = np.float(sp.limit(char_eq, nu, _z, dir="+"))
                lim_m = np.float(sp.limit(char_eq, nu, _z, dir="-"))
                if np.isclose(lim_p, lim_m):
                    return lim_m
                else:
                    # gained by dice roll, guaranteed to be fair.
                    return 5

        # search roots
        iter_limit = count * 10
        nu_num = find_roots(char_func,
                            n_roots=count,
                            grid=[np.linspace(0, iter_limit, 1e2)])

        if kwargs.get("debug", False):
            visualize_roots(nu_num,
                            [np.linspace(0, iter_limit, 1e3)],
                            char_func)

        # resolve kappa2
        if kappa[0, 1] == np.nan:
            kappa[:, 1] = kappa2_sol.evalf(nu_num)

        # reconstruct eigenvalues
        eta_num = np.ones_like(nu_num) * (-params.a1/(2*params.a2))
        char_roots = np.array([eta + 1j * nu
                               for eta, nu in zip(eta_num, nu_num)])

        eig_vectors = Base([SecondOrderEigenVector(char_root=r,
                                                   kappa=k,
                                                   domain=domain.bounds,
                                                   derivative_order=derivative_order)
                            for r, k in zip(char_roots, kappa)])

        return eig_vectors
        return normalize_base(eig_vectors)

    @staticmethod
    def convert_to_eigenvalue(params, char_root):
        """
        Converts a given characteristic into an
        eigenvalue.

        Parameters:
            params (bunch): system parameters, see TODO.
            char_root (complex): characteristic_root
        """
        return real(params.a2 / params.a0 * (
            char_root**2
            + char_root*params.a1/params.a2
            + .5 * (params.a1/params.a2)**2
        ))

    @staticmethod
    def convert_to_characteristic_root(params, eigenvalue):
        r"""
        Converts a given characteristic into an
        eigenvalue.

        Parameters:
            params (bunch): system parameters, see TODO.
            eigenvalue (real): eigenvalue :math:`\lamda`
        """
        return (-params.a1(2*params.a2)
                + 1j*np.sqrt(
                    (params.a1/(2*params.a2))**2
                    - params.a0/params.a2 * eigenvalue
                )
        )


class LambdifiedSympyExpression(Function):
    """
    This class provides a :py:class:`pyinduct.core.Function` :math:`\\varphi(z)` based on a lambdified sympy expression.
    The sympy expressions for the function and it's spatial derivatives must be provided as the list *sympy_funcs*.
    The expressions must be provided with increasing derivative order, starting with order 0.

    Args:
        sympy_funcs (array_like): Sympy expressions for the function and the derivatives:
            :math:`\\varphi(z), \\varphi'(z), ...`.
        spat_symbol: Sympy symbol for the spatial variable :math:`z`.
        spatial_domain (tuple): Domain on which :math:`\\varphi(z)` is defined (e.g.: :code:`spatial_domain=(0, 1)`).
    """

    def __init__(self, sympy_funcs, spat_symbol, spatial_domain):
        self._funcs = [lambdify(spat_symbol, sp_func, 'numpy') for sp_func in sympy_funcs]
        funcs = [self._func_factory(der_ord) for der_ord in range(len(sympy_funcs))]
        Function.__init__(self, funcs[0], nonzero=spatial_domain, derivative_handles=funcs[1:])

    def _func_factory(self, der_order):
        func = self._funcs[der_order]

        def function(z):
            return real(func(z))

        return function


class SecondOrderEigenfunction(metaclass=ABCMeta):
    # TODO is lambda really element of R in the following docstring?
    """
    Wrapper for all eigenvalue problems of the form

    .. math:: a_2\\varphi''(z) + a_1&\\varphi'(z) + a_0\\varphi(z) = \\lambda\\varphi(z), \\qquad a_2, a_1, a_0, \\lambda \\in \\mathbb R

    with eigenfunctions :math:`\\varphi` and eigenvalues :math:`\\lambda`.
    The roots of the characteristic equation (belonging to the ode) are denoted by

    .. math:: p = \\eta \\pm j\\omega, \\qquad \\eta \\in \\mathbb R, \\quad \\omega \\in \\mathbb C

    .. math:: \\eta = -\\frac{a_1}{2a_2}, \\quad \\omega = \\sqrt{-\\frac{a_1^2}{4 a_2^2} + \\frac{a_0 - \\lambda}{a_2}}

    In the following the variable :math:`\\omega` is called an eigenfrequency.
    """

    @abstractstaticmethod
    def eigfreq_eigval_hint(param, l, n_roots):
        """
        Args:
            param (array_like): Parameters :math:`(a_2, a_1, a_0, None, None)`.
            l: End of the domain :math:`z\\in[0, 1]`.
            n_roots (int): Number of eigenfrequencies/eigenvalues to be compute.

        Returns:
            tuple: Booth tuple elements are numpy.ndarrays of the same length, one for eigenfrequencies and one for eigenvalues.

                :math:`\\Big(\\big[\\omega_1,...,\\omega_\\text{n\\_roots}\Big], \\Big[\\lambda_1,...,\\lambda_\\text{n\\_roots}\\big]\\Big)`
        """

    @staticmethod
    def eigval_tf_eigfreq(param, eig_val=None, eig_freq=None):
        """
        Provide corresponding of eigenvalues/eigenfrequencies for given eigenfreqeuncies/eigenvalues, depending on which
        type is given.

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
        Return the parameters of the adjoint eigenvalue problem for the given parameter set.
        Hereby, dirichlet or robin boundary condition at :math:`z=0`

        .. math:: \\varphi(0) = 0 \\quad &\\text{or} \\quad \\varphi'(0) = \\alpha\\varphi(0)

        and dirichlet or robin boundary condition at :math:`z=l`

        .. math:: \\varphi`(l) = 0 \\quad &\\text{or} \\quad \\varphi'(l) = -\\beta\\varphi(l)

        can be imposed.

        Args:
            param (array_like): To define a homogeneous dirichlet boundary condition set alpha or beta to `None` at the
                corresponding side. Possibilities: \n
                - :math:`\\Big( a_2, a_1, a_0, \\alpha, \\beta \\Big)^T`,
                - :math:`\\Big( a_2, a_1, a_0, None, \\beta \\Big)^T`,
                - :math:`\\Big( a_2, a_1, a_0, \\alpha, None \\Big)^T` or
                - :math:`\\Big( a_2, a_1, a_0, None, None \\Big)^T`.

        Return:
            tuple:
                Parameters :math:`\\big(a_2, \\tilde a_1, a_0, \\tilde \\alpha, \\tilde \\beta \\big)` for
                the adjoint problem

                .. math::
                    a_2\\psi''(z) + \\tilde a_1&\\psi'(z) + a_0\\psi(z) = \\lambda\\psi(z) \\\\
                    \\psi(0) = 0 \\quad &\\text{or} \\quad \\psi'(0) = \\tilde\\alpha\\psi(0) \\\\
                    \\psi`(l) = 0 \\quad &\\text{or} \\quad \\psi'(l) = -\\tilde\\beta\\psi(l)

                with

                .. math:: \\tilde a_1 = -a_1, \\quad \\tilde\\alpha = \\frac{a_1}{a_2}\\alpha, \\quad \\tilde\\beta = -\\frac{a_1}{a_2}\\beta.
        """
        a2, a1, a0, alpha, beta = param

        if alpha is None:
            alpha_n = None
        else:
            alpha_n = a1 / a2 + alpha

        if beta is None:
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

        You must call this *classmethod* with one and only one of the kwargs: \n
            - *n* (*eig_val* and *eig_freq* will be computed with the :py:func:`eigfreq_eigval_hint`)
            - *eig_val* (*eig_freq* will be calculated with :py:func:`eigval_tf_eigfreq`)
            - *eig_freq* (*eig_val* will be calculated with :py:func:`eigval_tf_eigfreq`),\n
        or (and) pass the kwarg scale (then n is set to len(scale)).
        If you have the kwargs *eig_val* and *eig_freq* already calculated then these are preferable,
        in the sense of performance.


        Args:
            param: Parameters :math:`(a_2, a_1, a_0, ...)` see *evp_class.__doc__*.
            l: End of the eigenfunction domain (start is 0).
            n: Number of eigenvalues/eigenfunctions to compute.
            eig_freq (array_like): Pass your own choice of eigenfrequencies here.
            eig_val (array_like): Pass your own choice of eigenvalues here.
            max_order: Maximum derivative order which must provided by the eigenfunctions.
            scale (array_like): Here you can pass a list of values to scale the eigenfunctions.

        Returns:
            Tuple with one list for the eigenvalues and one for the eigenfunctions.
        """
        if np.sum([1 for arg in [n, eig_val, eig_freq] if arg is not None]) != 1 and scale is None:
            raise ValueError("You must pass one and only one of the kwargs:\n"
                             "\t - n (Number of eigenvalues/eigenfunctions to be compute)\n"
                             "\t - eig_val (Eigenvalues)\n"
                             "\t - eig_freq (Eigenfrequencies),\n"
                             "or (and) pass the kwarg scale (then n is set to len(scale)).")
        elif eig_val is not None:
            eig_freq = evp_class.eigval_tf_eigfreq(param, eig_val=eig_val)
            _n = len(eig_val)
        elif eig_freq is not None:
            eig_val = evp_class.eigval_tf_eigfreq(param, eig_freq=eig_freq)
            _n = len(eig_freq)
        else:
            if n is None:
                __n = len(scale)
            else:
                __n = n
            eig_freq, eig_val = evp_class.eigfreq_eigval_hint(param, l, __n)
            _n = n

        if scale is not None and _n is not None and len(scale) != _n:
            raise ValueError("Length of scale must match {n, len(eig_val), len(eig_freq}.")
        elif scale is None:
            scale = np.ones(eig_freq.shape)

        eig_func = np.array([evp_class(om, param, l, scale=sc, max_der_order=max_order) for om, sc in
                             zip(np.array(eig_freq, dtype=complex), scale)])

        return np.array(eig_val, dtype=complex), eig_func


class SecondOrderDirichletEigenfunction(LambdifiedSympyExpression, SecondOrderEigenfunction):
    """
    This class provides an eigenfunction :math:`\\varphi(z)` to eigenvalue problems of the form

    .. math::
        a_2\\varphi''(z) + a_1&\\varphi'(z) + a_0\\varphi(z) = \\lambda\\varphi(z) \\\\
        \\varphi(0) &= 0 \\\\
        \\varphi(l) &= 0.

    The eigenfrequency

    .. math:: \\omega = \\sqrt{-\\frac{a_1^2}{4a_2^2}+\\frac{a_0-\\lambda}{a_2}}

    must be provided (for example with the :py:func:`eigfreq_eigval_hint` of this class).

    Args:
        om (numbers.Number): eigenfrequency :math:`\\omega`
        param (array_like): :math:`\\Big( a_2, a_1, a_0, None, None \\Big)^T`
        l (numbers.Number): End of the domain :math:`z\\in [0,l]`.
        scale (numbers.Number): Factor to scale the eigenfunctions.
        max_der_order (int): Number of derivative handles that are needed.
    """

    def __init__(self, om, param, l, scale=1, max_der_order=2):
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

        LambdifiedSympyExpression.__init__(self, sp_funcs, z, (0, l))

    @staticmethod
    def eigfreq_eigval_hint(param, l, n_roots):
        """
        Return the first *n_roots* eigenfrequencies :math:`\\omega` and eigenvalues :math:`\\lambda`.

        .. math:: \\omega_i = \\sqrt{-\\frac{a_1^2}{4a_2^2}+\\frac{a_0-\\lambda_i}{a_2}} \\quad i = 1,...,\\text{n\\_roots}

        to the considered eigenvalue problem.

        Args:
            param (array_like): :math:`\\Big( a_2, a_1, a_0, None, None \\Big)^T`
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


class SecondOrderRobinEigenfunction(Function, SecondOrderEigenfunction):
    """
    This class provides an eigenfunction :math:`\\varphi(z)` to the eigenvalue problem given by

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
        l (numbers.Number): End of the domain :math:`z\\in [0,l]`.
        scale (numbers.Number): Factor to scale the eigenfunctions (corresponds to :math:`\\varphi(0)=\\text{phi\\_0}`).
        max_der_order (int): Number of derivative handles that are needed.
    """

    def __init__(self, om, param, l, scale=1, max_der_order=2):
        self._om = om
        self._param = param
        self._norm_fac = scale
        self._max_der_order = max_der_order

        self._om_is_close = np.isclose(self._om, 0)
        a2_, a1_, a0_, alpha_, beta_ = self._param
        eta_ = - a1_ / a2_ / 2

        alpha, beta, eta, omega, varphi_0, z, c1, c2, c3, c4, ll = sp.symbols(
            "alpha beta eta omega varphi_0 z c1 c2 c3 c4 l")
        subs_list = [(varphi_0, scale), (eta, eta_), (omega, om), (alpha, alpha_), (beta, beta_), (ll, l)]

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
        self._funcs = LambdifiedSympyExpression(sp_funcs, z, (0, l))

        zero_limit_sp_funcs = [sp.limit(sp_func, omega, 0) for sp_func in sp_funcs]
        self._zero_limit_funcs = LambdifiedSympyExpression(zero_limit_sp_funcs, z, (0, 1))

        funcs = [self._eig_func_factory(der_ord) for der_ord in range(max_der_order + 1)]
        Function.__init__(self, funcs[0], nonzero=(0, l), derivative_handles=funcs[1:])

    def _eig_func_factory(self, der_order):
        om_is_close = self._om_is_close
        func = self._funcs.derive(der_order)
        zero_limit_func = self._zero_limit_funcs.derive(der_order)

        def eigenfunction(z):
            if om_is_close:
                res = zero_limit_func(z)
            else:
                res = func(z)
            return real(res)

        return eigenfunction

    @staticmethod
    def eigfreq_eigval_hint(param, l, n_roots, show_plot=False):
        r"""
        Return the first *n_roots* eigenfrequencies :math:`\omega` and eigenvalues :math:`\lambda`.

        .. math:: \omega_i = \sqrt{
            - \frac{a_1^2}{4a_2^2}
            + \frac{a_0 - \lambda_i}{a_2}}
            \quad i = 1, \dotsc, \text{n\_roots}

        to the considered eigenvalue problem.

        Args:
            param (array_like): :math:`\big( a_2, a_1, a_0, \alpha, \beta \big)^T`
            l (numbers.Number): Right boundary value of the domain :math:`[0,l]\ni z`.
            n_roots (int): Amount of eigenfrequencies to compute.
            show_plot (bool): Show a plot window of the characteristic equation.

        Return:
            tuple --> booth tuple elements are numpy.ndarrays of length *nroots*:
            .. math:: \Big(\big[\omega_1, \dotsc, \omega_{\text{n\_roots}}\Big],
                \Big[\lambda_1, \dotsc, \lambda_{\text{n\_roots}}\big]\Big)
        """

        a2, a1, a0, alpha, beta = param
        eta = -a1 / 2. / a2

        # characteristic equations for eigen vectors: phi = c1 e^(eta z) + c2 z e^(eta z)
        char_eq = np.polynomial.Polynomial([alpha * beta * l + alpha + beta, alpha * l - beta * l, -l])

        # characteristic equations for eigen vectors: phi = e^(eta z) (c1 cos(om z) + c2 sin(om z))
        def characteristic_equation(omega):
            if np.isclose(omega, 0):
                return (alpha + beta) + (eta + beta) * (alpha - eta) * l
                # return ((alpha + beta) * np.cos(omega * l)
                #         + (eta + beta) * (alpha - eta) * l
                #         - omega * np.sin(omega * l))
            else:
                return (alpha + beta) * np.cos(omega * l) \
                       + ((eta + beta) * (alpha - eta) / omega - omega) * np.sin(omega * l)

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
            om = list(
                find_roots(characteristic_equation, 100, [np.array([0]), start_values_imag], rtol=int(np.log10(l) - 3),
                           cmplx=True))  #, get_all=True))
        except ValueError:
            om = list()

        # search real roots
        om += find_roots(characteristic_equation, 2 * n_roots, [start_values_real, np.array([0])],
                         rtol=int(np.log10(l) - 3), cmplx=True).tolist()

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


class TransformedSecondOrderEigenfunction(Function):
    """
    This class provides an eigenfunction :math:`\\varphi(z)` to the eigenvalue problem given by

    .. math:: a_2(z)\\varphi''(z) + a_1(z)\\varphi'(z) + a_0(z)\\varphi(z) = \\lambda\\varphi(z) \\quad,

    where :math:`\\lambda \\in \\mathbb{C}` denotes an eigenvalue and :math:`z \\in [z_0, \dotsc, z_n]` the domain.

    Args:
        target_eigenvalue (numbers.Number): :math:`\\lambda`
        init_state_vect (array_like):
            .. math:: \\Big(\\text{Re}\\{\\varphi(0)\\}, \\text{Re}\\{\\varphi'(0)\\}, \\text{Im}\\{\\varphi(0)\\}, \\text{Im}\\{\\varphi'(0)\\}\\Big)^T
        dgl_coefficients (array_like): Function handles
            :math:`\\Big( a2(z), a1(z), a0(z) \\Big)^T` .
        domain (array_like):
            :math:`\\Big( z_0, ..... , z_n \\Big)`
    """

    def __init__(self, target_eigenvalue, init_state_vect, dgl_coefficients, domain):

        if not all([isinstance(state, (int, float)) for state in init_state_vect]) and len(init_state_vect) == 4 \
            and isinstance(init_state_vect, (list, tuple)):
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
        self._a2, self._a1, self._a0 = [coef  # Function.from_constant(coef, domain=domain)
                                        for coef in dgl_coefficients]
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


class AddMulFunction(object):
    """
    (Temporary) Function class which can multiplied with scalars and added with functions.
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
    This class provides a transformed :py:class:`pyinduct.core.Function` :math:`\\bar x(z)` through the transformation
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
            Function :math:`x(z)` that will act as start for the generation of :math:`2n` Functions :math:`\\xi_{i,j}`

            .. math::
                &\\bar\\xi_{1,j} = x(z + jl_0),\\qquad j=0,...,n-1, \\quad l_0=l/n, \\quad z\\in[0,l_0] \\\\
                &\\bar\\xi_{2,j} = x(z + l - jl_0 ).

            The vector of functions :math:`\\boldsymbol\\xi` will then be constituted as follows:

            .. math:: \\boldsymbol\\xi = (\\xi_{1,0},...,\\xi_{1,n-1},\\xi_{2,0},...,\\xi_{2,n-1})^T .

        M (numpy.ndarray): Matrix :math:`T\\in\\mathbb R^{2n\\times 2n}` of scalars.
        l (numbers.Number): Length of the domain (:math:`z\\in [0,l]`).
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


# def real(to_return):
#     """
#     Check if the imaginary part of :code:`to_return` vanishes
#     and return the real part.
#
#     Args:
#         to_return (numbers.Number or array_like): Variable to check.
#
#     Raises:
#         ValueError: If (all) imaginary part(s) not vanishes.
#
#     Return:
#         numbers.Number or array_like: Real part of :code:`to_return`.
#     """
#     if not isinstance(to_return, (Number, list, np.ndarray)):
#         raise TypeError
#     if isinstance(to_return, (list, np.ndarray)):
#         if not all([isinstance(num, Number) for num in to_return]):
#             raise TypeError
#
#     maybe_real = np.atleast_1d(np.real_if_close(to_return))
#
#     if maybe_real.dtype == 'complex':
#         raise ValueError("Something goes wrong, imaginary part does not vanish")
#     else:
#         if maybe_real.shape == (1,):
#             maybe_real = maybe_real[0]
#         return maybe_real


def transform_to_intermediate(param, l=None):
    """
    Apply a transformation :math:`\\tilde x(z,t)=x(z,t)e^{\\int_0^z \\frac{a_1(\\bar z)}{2 a_2}\,d\\bar z}`
    which eliminates the advection term :math:`a_1 x(z,t)` from the reaction-advection-diffusion equation

    .. math:: \\dot x(z,t) = a_2 x''(z,t) + a_1(z) x'(z,t) + a_0(z) x(z,t)

    with robin

    .. math:: x'(0,t) = \\alpha x(0,t), \\quad x'(l,t) = -\\beta x(l,t) \\quad,

    dirichlet

    .. math:: x(0,t) = 0, \\quad x(l,t) = 0 \\quad

    or mixed boundary conditions.

    Note:
        To successfully transform the system, the first spatial derivative of :math`a_1(z)` is needed.

    Args:
        param (array_like): :math:`\\Big( a_2, a_1, a_0, \\alpha, \\beta \\Big)^T`
        l (numbers.Number): End of the domain (start is 0).

    Raises:
        TypeError: If :math:`a_1(z)` is callable but no derivative handle is defined for it.

    Return:
        tuple:
            Parameters :math:`\\big(a_2, \\tilde a_1=0, \\tilde a_0(z), \\tilde \\alpha, \\tilde \\beta \\big)` for
            the transformed system

            .. math:: \\dot{\\tilde{x}}(z,t) = a_2 \\tilde x''(z,t) + \\tilde a_0(z) \\tilde x(z,t)

            and the corresponding boundary conditions (:math:`\\alpha` and/or :math:`\\beta` set to None for dirichlet
            boundary conditions).

    """
    if not isinstance(param, (tuple, list)) or not len(param) == 5:
        raise TypeError("pyinduct.utils.transform_2_intermediate(): argument param must from type tuple or list")

    a2, a1, a0, alpha, beta = param
    if isinstance(a1, collections.Callable) or isinstance(a0, collections.Callable):
        if not len(a1._derivative_handles) >= 1:
            raise TypeError
        a0_z = ut.function_wrapper(a0)
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
        beta_n = -a1(l) / 2. / a2 + beta
    else:
        beta_n = -a1 / 2. / a2 + beta

    a2_n = a2
    a1_n = 0

    return a2_n, a1_n, a0_n, alpha_n, beta_n
