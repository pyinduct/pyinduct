"""
This modules provides eigenfunctions for a certain set of second order spatial
operators. Therefore functions for the computation of the corresponding
eigenvalues are included.
The functions which compute the eigenvalues are deliberately separated from
the predefined eigenfunctions in order to handle transformations and reduce
effort within the controller implementation.
"""

import collections
from abc import ABCMeta, abstractstaticmethod
from functools import partial
from numbers import Number

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as si
import sympy as sp
from sympy.utilities.lambdify import lambdify

from .core import (Domain, Base, Function, generic_scalar_product,
                   calculate_scalar_product_matrix, dot_product_l2,
                   normalize_base, find_roots, real)
from .visualization import visualize_roots

__all__ = ["SecondOrderOperator", "SecondOrderEigenVector", "SecondOrderEigenfunction",
           "SecondOrderDirichletEigenfunction", "SecondOrderRobinEigenfunction",
           "TransformedSecondOrderEigenfunction", "AddMulFunction",
           "LambdifiedSympyExpression", "FiniteTransformFunction"]


class SecondOrderOperator:
    r"""
    Interface class to collect all important parameters that describe
    a second order ordinary differential equation.

    Args:
        a2(Number or callable): coefficient :math:`a_2`.
        a1(Number or callable): coefficient :math:`a_1`.
        a0(Number or callable): coefficient :math:`a_0`.
        alpha1(Number): coefficient :math:`\alpha_1`.
        alpha0(Number): coefficient :math:`\alpha_0`.
        beta1(Number): coefficient :math:`\beta_1`.
        beta0(Number): coefficient :math:`\beta_0`.
    """
    def __init__(self, a2=0, a1=0, a0=0,
                 alpha1=0, alpha0=0,
                 beta1=0, beta0=0,
                 domain=(-np.inf, np.inf)):

        self.a2 = a2
        self.a1 = a1
        self.a0 = a0

        self.alpha1 = alpha1
        self.alpha0 = alpha0

        self.beta1 = beta1
        self.beta0 = beta0

        self.domain = domain

    @staticmethod
    def from_list(param_list, domain=None):
        return SecondOrderOperator(*param_list, domain=domain)

    @staticmethod
    def from_dict(param_dict, domain=None):
        return SecondOrderOperator(**param_dict, domain=domain)

    def get_adjoint_problem(self):
        r"""
        Return the parameters of the operator :math:`A^*` describing the
        the problem

        .. math:: (\textup{A}^*\psi)(z) = \bar a_2 \partial_z^2 \psi(z)
                                            + \bar a_1 \partial_z \psi(z)
                                            + \bar a_0 \psi(z) \:,

        where the :math:`\bar a_i` are constant and whose boundary conditions
        are given by

        .. math:: \bar\alpha_1 \partial_z \psi(z_1) + \bar\alpha_0 \psi(z_1)
                    &= 0 \\
                  \bar\beta_1 \partial_z \psi(z_2) + \bar\beta_0 \psi(z_2)
                    &= 0 .

        The following mapping is used:

        .. math:: \bar a_2 = a_2,  \quad\bar a_1 = -a_1, \quad\bar a_0 = a_0, \\
                \bar\alpha_1 = -1, \quad
                     \bar\alpha_0 = \frac{a_1}{a_2} - \frac{\alpha_0}{\alpha_1}
                     , \\
                \bar\beta_1 = -1, \quad
                    \bar\beta_0 = \frac{a_1}{a_2} - \frac{\beta_0}{\beta_1} \:.

        Return:
            :py:class:`.SecondOrderOperator` : Parameter set describing
            :math:`A^*` .
        """
        return SecondOrderOperator(a2=self.a2,
                                   a1=-self.a1,
                                   a0=self.a0,
                                   alpha1=-1,
                                   alpha0=(self.a1/self.a2
                                           - self.alpha0 / self.alpha1),
                                   beta1=-1,
                                   beta0=(self.a1/self.a2
                                          - self.beta0 / self.beta1),
                                   )


class SecondOrderEigenVector(Function):
    r"""
    This class provides eigenvectors of the form

    .. math:: \varphi(z) =  e^{\eta z} \big(\kappa_1 \cos(\nu z)
                            + \sin(\nu z) \big) \:,

    of a linear second order spatial operator :math:`\textup{A}` denoted by

    .. math:: (\textup{A}\varphi)(z) = a_2 \partial_z^2 \varphi(z)
                                        + a_1 \partial_z \varphi(z)
                                        + a_0 \varphi(z)

    where the :math:`a_i` are constant and whose boundary conditions are given
    by

    .. math:: \alpha_1 \partial_z x(z_1) + \alpha_0 x(z_1) &= 0 \\
              \beta_1 \partial_z x(z_2) + \beta_0 x(z_2) &= 0 .

    To calculate the corresponding eigenvectors, the problem

    .. math:: (\textup{A}\varphi)(z) = \lambda \varphi(z)

    is solved for the eigenvalues :math:`\lambda` , making use of the
    characteristic roots :math:`p` given by

    .. math:: p = \underbrace{-\frac{a_1}{a_2}}_{=:\eta}
                    + j\underbrace{\sqrt{ \frac{a_0 - \lambda}{a_2}
                            - \left(\frac{a_1}{2a_2}\right)^2 }}_{=:\nu}

    Note:
        To easily instantiate a set of eigenvectors for a certain
        system, use the :py:func:`.cure_hint` of this class or even
        better the helper-function
        :py:func:`.cure_interval` .

    Warn:
        Since an eigenvalue corresponds to a pair of conjugate complex
        characteristic roots, latter are only calculated for the positive
        half-plane since the can be mirrored.
        To obtain the orthonormal properties of the generated
        eigenvectors, the eigenvalue corresponding to the characteristic
        root 0+0j is ignored, since it leads to the zero function.

    Parameters:
        char_pair (tuple of complex): Characteristic root, corresponding to the
            eigenvalue :math:`\lambda` for which the eigenvector is
            to be determined.
            (Can be obtained by :py:meth:`.convert_to_characteristic_root`)
        coefficients (tuple): Constants of the exponential ansatz solution.

    Returns:
        :py:class:`.SecondOrderEigenVector` : The eigenvector.
    """

    def __init__(self, char_pair, coefficients, domain, derivative_order):
        # build generic solution
        z, p1, p2, c1, c2 = sp.symbols("z p1 p2 c1 c2")
        gen_sols = [c1 * sp.exp(p1 * z) + c2 * sp.exp(p2 * z)]
        gen_sols[0] = gen_sols[0].subs([(c1, coefficients[0]),
                                        (c2, coefficients[1]),
                                        (p1, char_pair[0]),
                                        (p2, char_pair[1]),
                                        ])
        # derive
        for d in range(derivative_order):
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
            domain (:py:class:`.Domain`): Domain of the
                spatial problem.
            params (:py:class:`.SecondOrderOperator`): Parameters of the system,
                see :py:func:`__init__` for details on their definition.
                Long story short, it must contain
                :math:`a_2, a_1, a_0, \alpha_0, \alpha_1, \beta_0 \text{ and }
                \beta_1` .
            count (int): Amount of eigenvectors to generate.
            derivative_order (int): Amount of derivative handles to provide.
            kwargs: will be passed to :py:meth:`.calculate_eigenvalues`

        Keyword Arguments:
            debug (bool): If provided, this parameter will cause several debug
                windows to open.

        Returns:
            tuple of (array, :py:class:`.Base`): An array holding the
            eigenvalues paired with a basis spanned by the eigenvectors.
        """
        diff = 1
        while diff > 0:
            eig_values, char_roots, coefficients = \
                SecondOrderEigenVector.calculate_eigenvalues(
                    domain,
                    params,
                    count + diff,
                    extended_output=True,
                    **kwargs)

            fractions = []
            for root_set, coeffs in zip(char_roots, coefficients):
                frac = SecondOrderEigenVector(char_pair=root_set,
                                              coefficients=coeffs,
                                              domain=domain.bounds,
                                              derivative_order=derivative_order)
                fractions.append(frac)

            eig_base = Base(fractions)

            # if we can't normalize it, we can't use it
            norm = generic_scalar_product(eig_base)
            orthogonal_sols = np.where(np.isclose(norm, 0))
            eig_values = np.delete(eig_values, orthogonal_sols)[:count]
            eig_vectors = np.delete(eig_base.fractions, orthogonal_sols)[:count]

            diff = max(0, count - len(eig_vectors))

        base = normalize_base(Base(eig_vectors[:count]))
        return eig_values, base

    @staticmethod
    def calculate_eigenvalues(domain, params, count, extended_output=False,
                              **kwargs):
        r"""
        Determine the eigenvalues of the problem given by *parameters*
        defined on *domain* .

        Parameters:
            domain (:py:class:`.Domain`): Domain of the
                spatial problem.
            params (bunch-like): Parameters of the system, see
                :py:func:`__init__` for details on their definition.
                Long story short, it must contain
                :math:`a_2, a_1, a_0, \alpha_0, \alpha_1,
                \beta_0 \text{ and } \beta_1` .
            count (int): Amount of eigenvalues to generate.
            extended_output (bool): If true, not only eigenvalues but also
                the corresponding characteristic roots and coefficients
                of the eigenvectors are returned. Defaults to False.

        Keyword Arguments:
            debug (bool): If provided, this parameter will cause several debug
                windows to open.

        Returns:
            array or tuple of arrays: :math:`\lambda` , ordered in increasing
            order or tuple of (:math:`\lambda, p, \boldsymbol{\kappa}` )
            if *extended_output* is True.
        """
        if (params.alpha0 == 0 and params.alpha1 == 0
                or params.beta0 == 0 and params.beta1 == 0):
            raise ValueError("Provided boundary conditions are useless.")

        bounds = domain.bounds

        # again, build generic solution
        z, nu, c1, c2 = sp.symbols("z nu c_1 c_2")
        mu = -.5*params.a1 / params.a2
        p1 = mu + nu
        p2 = mu - nu
        gen_sol = (c1 * sp.exp(p1 * z) + c2 * sp.exp(p2 * z))

        # check special case of a dirichlet boundary defined at zero
        if (params.alpha1 == 0 and params.alpha0 != 0 and bounds[0] == 0
                or params.beta1 == 0 and params.beta0 != 0 and bounds[1] == 0):
            # since c1 + c2 is equal to sol evaluated at z=0 (c1 + c2 = 0)
            gen_sol = gen_sol.subs(c2, -c1)
            # choose arbitrary scaling to be one
            gen_sol = gen_sol.subs(c1, 1)
            dirichlet_zero = True
            settled_bc = domain.bounds.index(0)
        else:
            # choose the arbitrary scaling phi(0) to be one -> c1 + c2 = 1
            gen_sol = gen_sol.subs(c2, 1 - c1)
            dirichlet_zero = False

            # incorporate the first boundary condition
            bc1 = (params.alpha0 * gen_sol.subs(z, bounds[0])
                   + params.alpha1 * gen_sol.diff(z).subs(z, bounds[0]))
            c1_sol = sp.solve(bc1, c1)[0]
            gen_sol = gen_sol.subs(c1, c1_sol)
            settled_bc = 0

        if settled_bc == 1:
            ci = c1
            g0 = params.alpha0
            g1 = params.alpha1
        elif settled_bc == 0:
            ci = c2
            g0 = params.beta0
            g1 = params.beta1
        else:
            raise ValueError

        char_eq = (g0 * gen_sol.subs(z, bounds[1 - settled_bc])
                   + g1 * gen_sol.diff(z).subs(z, bounds[1 - settled_bc]))

        # lambdify
        char_func = sp.lambdify(nu, char_eq, modules="numpy")

        if kwargs.get("debug", False):
            sp.init_printing()
            print("characteristic equation:")
            sp.pretty_print(char_eq)

        cache = {}

        def patched_func(_z):
            """
            A patched version of a sympy expression.
            If the limit exists it is used to lift poles of the
            function.
            """
            # TODO export into own wrapper class (see LambdifiedSympyExpression)

            try:
                return char_func(_z)
            except FloatingPointError:
                if _z in cache:
                    return cache[_z]
                else:
                    lim_p = np.float(sp.limit(char_eq, nu, _z, dir="+"))
                    lim_m = np.float(sp.limit(char_eq, nu, _z, dir="-"))
                    if np.isclose(lim_p, lim_m):
                        cache[_z] = lim_m
                        return lim_m
                    else:
                        print("Unsteady function")
                        # chosen by fair dice roll. guaranteed to be random.
                        return 4

        if 0:
            # extract numerator and denominator
            numer, denom = char_eq.as_numer_denom()

            freqs = []

            def extract_trig(expr):
                if expr.func == sp.sin or expr.func == sp.cos:
                    freqs.append(sp.collect(expr.args[0], nu, evaluate=False)[nu])

                for arg in expr.args:
                    extract_trig(arg)

            extract_trig(numer)
            try:
                max_freq = float(max(freqs))
                root_dist = np.pi/max_freq
            except ValueError:
                root_dist = .1

            if kwargs.get("debug", False):
                sp.init_printing()
                print("characteristic equation:")
                sp.pretty_print(char_eq)
                # sp.plot(char_eq, (z, *domain.bounds))
                print("numerator:")
                sp.pretty_print(numer)
                # sp.plot(numer, (z, *domain.bounds))
                print("denominator:")
                sp.pretty_print(denom)
                # sp.plot(denom, (z, *domain.bounds))
                print("estimated root distance of: {}".format(root_dist))

            numer_func = sp.lambdify(nu, numer, modules="numpy")
            denom_func = sp.lambdify(nu, denom, modules="numpy")

            diff = 1
            while diff > 0:
                # search roots
                # an upper limit is hard to guess
                iter_limit = 10 * root_dist * (count + diff)
                search_grid = Domain(bounds=(0, iter_limit),
                                     step=root_dist)
                num_roots = find_roots(numer_func, grid=[search_grid.points],
                                       n_roots=count + diff,
                                       rtol=root_dist*1e-2)

                # check for common roots of numerator and denominator
                nu_num = []
                for root in num_roots:
                    if not np.isclose(denom_func(root), 0):
                        nu_num.append(root)

                diff = max(0, count - len(nu_num))

            nu_num = np.array(nu_num[:count])

        # TODO introduce step heuristic, again.
        limit_real = count * np.pi
        limit_imag = 4 * count * np.pi
        roots = find_roots(function=patched_func,
                           grid=[np.linspace(-limit_real, limit_real, num=100),
                                 np.linspace(-limit_imag, limit_imag, num=100)],
                           cmplx=True,
                           sort_mode="norm")
        np.testing.assert_almost_equal(char_func(roots), 0, verbose=True)

        if kwargs.get("debug", False):
            visualize_roots(roots,
                            grid=[np.linspace(-2, 2), np.linspace(-30, 30)],
                            func=patched_func,
                            cmplx=True)

        # reconstruct characteristic pairs p1 and p2
        char_pairs = np.hstack([np.atleast_2d(mu + roots).T,
                                np.atleast_2d(mu - roots).T])

        # reconstruct eigenvalues
        eig_values = SecondOrderEigenVector.convert_to_eigenvalue(params,
                                                                  char_pairs)

        # sort out duplicates
        _unique_entries = np.unique(np.round(eig_values, decimals=5))

        # order by abs and
        sort_idx = np.argsort(np.abs(_unique_entries))
        unique_values = _unique_entries[sort_idx][:count]

        # recreate corresponding characteristic pairs
        unique_pairs = SecondOrderEigenVector.convert_to_characteristic_root(
            params,
            unique_values)

        # resolve coefficients
        c = np.zeros((len(unique_values), 2), dtype=complex)
        if dirichlet_zero:
            # c1 has been set to one, determine c2 = -c1
            c[:, 0] = 1
            c[:, 1] = -c[:, 0]
        else:
            # c1 + c2 has been set to one, determine c2 = 1 - c1
            c1_handle = sp.lambdify(nu, c1_sol, modules="numpy")

            def patched_c_handle(x):
                try:
                    return c1_handle(x)
                except FloatingPointError:
                    return np.NaN

            c[:, 0] = [patched_c_handle(pair[0]) for pair in unique_pairs]
            c[:, 1] = 1 - c[:, 0]

            # remove invalid entries (c1 or c2 is NaN)
            invalid_idx = np.where(np.logical_and(*np.isnan(c).T))
            unique_values = np.delete(unique_values, invalid_idx)
            unique_pairs = np.delete(unique_pairs, invalid_idx, axis=0)
            c = np.delete(c, invalid_idx, axis=0)

        if kwargs.get("debug", False):
            print("roots: {}".format(unique_pairs))
            print("eig_vals: {}".format(unique_values))
            print("coefficients: {}".format(c))

        if extended_output:
            return unique_values, unique_pairs, c
        else:
            return unique_values

    @staticmethod
    def convert_to_eigenvalue(params, char_roots):
        r"""
        Converts a pair of characteristic roots :math:`p_{1,2}` into an
        eigenvalue :math:`\lambda` by using the provided parameters.
        The relation is given by

        .. math:: \lambda = a_2 p^2 + a_1 p + a_0

        Parameters:
            params (:py:class:`.SecondOrderOperator`): System parameters.
            char_roots (tuple or array of tuples): Characteristic roots
        """
        char_roots = np.atleast_2d(char_roots)
        l1, l2 = (params.a2 * char_roots[:, 0] ** 2
                  + params.a1 * char_roots[:, 0]
                  + params.a0,
                  params.a2 * char_roots[:, 1] ** 2
                  + params.a1 * char_roots[:, 1]
                  + params.a0)

        # TODO: Is actually np.testing.assert_array_almost_equal() needed?
        if not np.allclose(l1, l2):
            raise ValueError("Given characteristic root pair must resolve to"
                             "a single eigenvalue.")
        return l1

    @staticmethod
    def convert_to_characteristic_root(params, eigenvalue):
        r"""
        Converts a given eigenvalue :math:`\lambda` into a
        characteristic root :math:`p` by using the provided
        parameters. The relation is given by

        .. math:: p = -\frac{a_1}{a_2}
                    + j\sqrt{ \frac{a_0 - \lambda}{a_2}
                            - \left(\frac{a_1}{2a_2}\right)^2 }

        Parameters:
            params (bunch): system parameters, see :py:func:`.cure_hint` .
            eigenvalue (real): eigenvalue :math:`\lambda`

        Returns:
            complex number: characteristic root :math:`p`
        """
        eig_values = np.atleast_1d(eigenvalue)
        res = np.array([(-params.a1 / (2 * params.a2)
                         + np.sqrt((params.a1 / 2 / params.a2) ** 2
                                   - (params.a0 - val) / params.a2),
                         -params.a1 / (2 * params.a2)
                         - np.sqrt((params.a1 / 2 / params.a2) ** 2
                                   - (params.a0 - val) / params.a2))
                        for val in eig_values])

        return res


class LambdifiedSympyExpression(Function):
    r"""
    This class provides a :py:class:`.Function` :math:`\varphi(z)` based on a
    lambdified sympy expression. The sympy expressions for the function and it's
    spatial derivatives must be provided as the list *sympy_funcs*. The
    expressions must be provided with increasing derivative order, starting with
    order 0.

    Args:
        sympy_funcs (array_like): Sympy expressions for the function and the
            derivatives: :math:`\varphi(z), \varphi'(z), ...`.
        spat_symbol: Sympy symbol for the spatial variable :math:`z`.
        spatial_domain (tuple): Domain on which :math:`\varphi(z)` is defined
            (e.g.: :code:`spatial_domain=(0, 1)`).
    """

    def __init__(self, sympy_funcs, spat_symbol, spatial_domain):
        self._funcs = [lambdify(spat_symbol, sp_func, 'numpy')
                       for sp_func in sympy_funcs]
        funcs = [self._func_factory(der_ord)
                 for der_ord in range(len(sympy_funcs))]
        Function.__init__(self, funcs[0],
                         domain=spatial_domain,
                         nonzero=spatial_domain,
                         derivative_handles=funcs[1:])

    def _func_factory(self, der_order):
        func = self._funcs[der_order]

        def function(z):
            return real(func(z))

        return function


class SecondOrderEigenfunction(metaclass=ABCMeta):
    r"""
    Wrapper for all eigenvalue problems of the form

    .. math:: a_2\varphi''(z) + a_1&\varphi'(z) + a_0\varphi(z) =
        \lambda\varphi(z), \qquad a_2, a_1, a_0, \lambda \in \mathbb C

    with eigenfunctions :math:`\varphi` and eigenvalues :math:`\lambda`.
    The roots of the characteristic equation (belonging to the ode) are denoted
    by

    .. math:: p = \eta \pm j\omega, \qquad \eta \in \mathbb R,
        \quad \omega \in \mathbb C

    .. math:: \eta = -\frac{a_1}{2a_2}, \quad
        \omega = \sqrt{-\frac{a_1^2}{4 a_2^2} + \frac{a_0 - \lambda}{a_2}}

    In the following the variable :math:`\omega` is called an eigenfrequency.
    """

    @abstractstaticmethod
    def eigfreq_eigval_hint(param, l, n_roots):
        r"""
        Args:
            param (array_like): Parameters :math:`(a_2, a_1, a_0, None, None)`.
            l: End of the domain :math:`z\in[0, 1]`.
            n_roots (int): Number of eigenfrequencies/eigenvalues to be compute.

        Returns:
            tuple: Booth tuple elements are numpy.ndarrays of the same length,
            one for eigenfrequencies and one for eigenvalues.

            .. math:: \Big(\big[\omega_1,...,\omega_\text{n\_roots}\Big],
                \Big[\lambda_1,...,\lambda_\text{n\_roots}\big]\Big)
        """

    @staticmethod
    def eigval_tf_eigfreq(param, eig_val=None, eig_freq=None):
        r"""
        Provide corresponding of eigenvalues/eigenfrequencies for given
        eigenfreqeuncies/eigenvalues, depending on which
        type is given.

        .. math:: \omega = \sqrt{-\frac{a_1^2}{4a_2^2}+\frac{a_0-\lambda}{a_2}}

        respectively

        .. math:: \lambda = -\frac{a_1^2}{4a_2}+a_0 - a_2 \omega.

        Args:
            param (array_like): Parameters :math:`(a_2, a_1, a_0, None, None)`.
            eig_val (array_like): Eigenvalues :math:`\lambda`.
            eig_freq (array_like): Eigenfrequencies :math:`\omega`.

        Returns:
            numpy.array: Eigenfrequencies :math:`\omega` or eigenvalues
            :math:`\lambda`.
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
        r"""
        Return the parameters of the adjoint eigenvalue problem for the given
        parameter set. Hereby, dirichlet or robin boundary condition at
        :math:`z=0`

        .. math:: \varphi(0) = 0 \quad &\text{or} \quad
            \varphi'(0) = \alpha\varphi(0)

        and dirichlet or robin boundary condition at :math:`z=l`

        .. math:: \varphi`(l) = 0 \quad &\text{or} \quad
            \varphi'(l) = -\beta\varphi(l)

        can be imposed.

        Args:
            param (array_like): To define a homogeneous dirichlet boundary
                condition set alpha or beta to `None` at the corresponding side.
                Possibilities:

                - :math:`\Big( a_2, a_1, a_0, \alpha, \beta \Big)^T`,
                - :math:`\Big( a_2, a_1, a_0, None, \beta \Big)^T`,
                - :math:`\Big( a_2, a_1, a_0, \alpha, None \Big)^T` or
                - :math:`\Big( a_2, a_1, a_0, None, None \Big)^T`.

        Return:
            tuple:
            Parameters :math:`\big(a_2, \tilde a_1, a_0, \tilde \alpha, \tilde \beta \big)` for
            the adjoint problem

            .. math::
                a_2\psi''(z) + \tilde a_1&\psi'(z) + a_0\psi(z) = \lambda\psi(z) \\
                \psi(0) = 0 \quad &\text{or} \quad \psi'(0) = \tilde\alpha\psi(0) \\
                \psi`(l) = 0 \quad &\text{or} \quad \psi'(l) = -\tilde\beta\psi(l)

            with

            .. math:: \tilde a_1 = -a_1, \quad \tilde\alpha = \frac{a_1}{a_2}\alpha, \quad \tilde\beta = -\frac{a_1}{a_2}\beta.
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
        Provide the first *n* eigenvalues and eigenfunctions (wraped inside a
        pyinduct base). For the exact formulation of the considered eigenvalue
        problem, have a look at the docstring from the eigenfunction class from
        which you will call this method.

        You must call this *classmethod* with one and only one of the kwargs:

            - *n* (*eig_val* and *eig_freq* will be computed with the
              :py:meth:`.eigfreq_eigval_hint`)
            - *eig_val* (*eig_freq* will be calculated with
              :py:meth:`.eigval_tf_eigfreq`)
            - *eig_freq* (*eig_val* will be calculated with
              :py:meth:`.eigval_tf_eigfreq`),\n
        or (and) pass the kwarg scale (then n is set to len(scale)).
        If you have the kwargs *eig_val* and *eig_freq* already calculated then
        these are preferable, in the sense of performance.


        Args:
            param: Parameters :math:`(a_2, a_1, a_0, ...)` see
                *evp_class.__doc__*.
            l: End of the eigenfunction domain (start is 0).
            n: Number of eigenvalues/eigenfunctions to compute.
            eig_freq (array_like): Pass your own choice of eigenfrequencies
                here.
            eig_val (array_like): Pass your own choice of eigenvalues here.
            max_order: Maximum derivative order which must provided by the
                eigenfunctions.
            scale (array_like): Here you can pass a list of values to scale the
                eigenfunctions.

        Returns:
            Tuple with one list for the eigenvalues and one base which fractions are the
            eigenfunctions.
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

        return np.array(eig_val, dtype=complex), Base(eig_func)


class SecondOrderDirichletEigenfunction(LambdifiedSympyExpression, SecondOrderEigenfunction):
    r"""
    This class provides an eigenfunction :math:`\varphi(z)` to eigenvalue
    problems of the form

    .. math::
        a_2\varphi''(z) + a_1&\varphi'(z) + a_0\varphi(z) = \lambda\varphi(z) \\
        \varphi(0) &= 0 \\
        \varphi(l) &= 0.

    The eigenfrequency

    .. math:: \omega = \sqrt{-\frac{a_1^2}{4a_2^2}+\frac{a_0-\lambda}{a_2}}

    must be provided (for example with the :py:meth:`.eigfreq_eigval_hint`
    of this class).

    Args:
        om (numbers.Number): eigenfrequency :math:`\omega`
        param (array_like): :math:`\Big( a_2, a_1, a_0, None, None \Big)^T`
        l (numbers.Number): End of the domain :math:`z\in [0,l]`.
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
        r"""
        Return the first *n_roots* eigenfrequencies :math:`\omega` and
        eigenvalues :math:`\lambda`.

        .. math:: \omega_i = \sqrt{-\frac{a_1^2}{4a_2^2}+
            \frac{a_0-\lambda_i}{a_2}} \quad i = 1,...,\text{n\_roots}

        to the considered eigenvalue problem.

        Args:
            param (array_like): :math:`\Big( a_2, a_1, a_0, None, None \Big)^T`
            l (numbers.Number): Right boundary value of the domain
                :math:`[0,l]\ni z`.
            n_roots (int): Amount of eigenfrequencies to be compute.

        Return:
            tuple --> two numpy.ndarrays of length *n_roots*:

            .. math:: \Big(\big[\omega_1,...,\omega_\text{n\_roots}\Big],
                \Big[\lambda_1,...,\lambda_\text{n\_roots}\big]\Big)
        """
        a2, a1, a0, _, _ = param
        eig_frequencies = np.array([i * np.pi / l for i in np.arange(1, n_roots + 1)])
        eig_values = a0 - a2 * eig_frequencies ** 2 - a1 ** 2 / 4. / a2

        return eig_frequencies, eig_values


class SecondOrderRobinEigenfunction(Function, SecondOrderEigenfunction):
    r"""
    This class provides an eigenfunction :math:`\varphi(z)` to the eigenvalue
    problem given by

    .. math::
        a_2\varphi''(z) + a_1&\varphi'(z) + a_0\varphi(z) = \lambda\varphi(z) \\
        \varphi'(0) &= \alpha \varphi(0) \\
        \varphi'(l) &= -\beta \varphi(l).

    The eigenfrequency

    .. math:: \omega = \sqrt{-\frac{a_1^2}{4a_2^2}+\frac{a_0-\lambda}{a_2}}

    must be provided (for example with the :py:meth:`.eigfreq_eigval_hint` of
    this class).

    Args:
        om (numbers.Number): eigenfrequency :math:`\omega`
        param (array_like): :math:`\Big( a_2, a_1, a_0, \alpha, \beta \Big)^T`
        l (numbers.Number): End of the domain :math:`z\in [0,l]`.
        scale (numbers.Number): Factor to scale the eigenfunctions (corresponds
            to :math:`\varphi(0)=\text{phi\_0}`).
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

        zero_limit_sp_funcs = [sp.limit(sp_func, omega, 0)
                               for sp_func in sp_funcs]
        self._zero_limit_funcs = LambdifiedSympyExpression(
            zero_limit_sp_funcs, z, (0, l))

        funcs = [self._eig_func_factory(der_ord)
                 for der_ord in range(max_der_order + 1)]
        Function.__init__(self, funcs[0], domain=(0, l), nonzero=(0, l),
                          derivative_handles=funcs[1:])

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
        Return the first *n_roots* eigenfrequencies :math:`\omega` and
        eigenvalues :math:`\lambda`.

        .. math:: \omega_i = \sqrt{
            - \frac{a_1^2}{4a_2^2}
            + \frac{a_0 - \lambda_i}{a_2}}
            \quad i = 1, \dotsc, \text{n\_roots}

        to the considered eigenvalue problem.

        Args:
            param (array_like): Parameters
                :math:`\big( a_2, a_1, a_0, \alpha, \beta \big)^T`
            l (numbers.Number): Right boundary value of the domain
                :math:`[0,l]\ni z`.
            n_roots (int): Amount of eigenfrequencies to compute.
            show_plot (bool): Show a plot window of the characteristic equation.

        Return:
            tuple --> booth tuple elements are numpy.ndarrays of length *nroots*:
            .. math:: \Big(\big[\omega_1, \dotsc, \omega_{\text{n\_roots}}\Big],
                \Big[\lambda_1, \dotsc, \lambda_{\text{n\_roots}}\big]\Big)
        """

        a2, a1, a0, alpha, beta = param
        eta = -a1 / 2. / a2

        # characteristic equations for eigen vectors:
        # phi = c1 e^(eta z) + c2 z e^(eta z)
        char_eq = np.polynomial.Polynomial([alpha * beta * l + alpha + beta,
                                            alpha * l - beta * l,
                                            -l])

        # characteristic equations for eigen vectors:
        # phi = e^(eta z) (c1 cos(om z) + c2 sin(om z))
        def characteristic_equation(omega):
            if np.isclose(omega, 0):
                return alpha + beta + (eta + beta) * (alpha - eta) * l
                # return ((alpha + beta) * np.cos(omega * l)
                #         + (eta + beta) * (alpha - eta) * l
                #         - omega * np.sin(omega * l))
            else:
                return ((alpha + beta) * np.cos(omega * l)
                        + ((eta + beta) * (alpha - eta) / omega
                           - omega) * np.sin(omega * l))

        # assume 1 root per pi/l (safety factor = 3)
        search_begin = 0
        search_end = 3 * n_roots * np.pi / l
        start_values_real = np.linspace(search_begin,
                                        search_end,
                                        search_end / np.pi * l * 100)
        start_values_imag = np.linspace(search_begin,
                                        search_end,
                                        search_end / np.pi * l * 20)

        if show_plot:
            vec_function = np.vectorize(characteristic_equation)
            plt.plot(start_values_real,
                     np.real(vec_function(start_values_real)))
            plt.plot(start_values_real,
                     np.imag(vec_function(start_values_real)))
            plt.show()
            plt.plot(start_values_imag * 1j,
                     np.real(vec_function(start_values_imag * 1j)))
            plt.plot(start_values_imag * 1j,
                     np.imag(vec_function(start_values_imag * 1j)))
            plt.show()

        # search imaginary roots
        try:
            om = list(find_roots(characteristic_equation,
                                 [np.array([0]), start_values_imag],
                                 rtol=1e-3 / l, cmplx=True))
        except ValueError:
            om = list()

        # search real roots
        om += find_roots(characteristic_equation, [start_values_real],
                         2 * n_roots, rtol=1e-3 / l,
                         cmplx=False).tolist()

        # only "real" roots and complex roots with imaginary part != 0
        # and real part == 0 considered
        if any([not np.isclose(root.real, 0)
                and not np.isclose(root.imag, 0) for root in om]):
            raise NotImplementedError("This case is currently not considered.")

        # read out complex roots
        _complex_roots = [root for root in om if np.isclose(root.real, 0)
                          and not np.isclose(root.imag, 0)]
        complex_roots = list()
        for complex_root in _complex_roots:
            if not any([np.isclose(np.abs(complex_root), _complex_root)
                        for _complex_root in complex_roots]):
                complex_roots.append(complex_root)

        # sort out all complex roots and roots with negative real part
        om = [root.real + 0j for root in om
              if root.real >= 0 and np.isclose(root.imag, 0)]

        # delete all around om = 0
        om = [val for val in om if not np.isclose(val * l, 0, atol= 1e-4)]

        # if om = 0 is a root and the corresponding characteristic equation
        # is satisfied then add 0 to the list
        if (np.isclose(np.abs(characteristic_equation(0)), 0)
                and any(np.isclose(char_eq.roots(), 0))):
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
    r"""
    This class provides an eigenfunction :math:`\varphi(z)` to the eigenvalue
    problem given by

    .. math:: a_2(z)\varphi''(z) + a_1(z)\varphi'(z) + a_0(z)\varphi(z)
        = \lambda\varphi(z) \quad,

    where :math:`\lambda \in \mathbb{C}` denotes an eigenvalue and
    :math:`z \in [z_0, \dotsc, z_n]` the domain.

    Args:
        target_eigenvalue (numbers.Number): :math:`\lambda`
        init_state_vector (array_like):
            .. math:: \Big(\text{Re}\{\varphi(0)\}, \text{Re}\{\varphi'(0)\},
                \text{Im}\{\varphi(0)\}, \text{Im}\{\varphi'(0)\}\Big)^T
        dgl_coefficients (array_like): Function handles
            :math:`\Big( a2(z), a1(z), a0(z) \Big)^T` .
        domain (:py:class:`.Domain`): Spatial domain of the
            problem.
    """

    def __init__(self, target_eigenvalue, init_state_vector, dgl_coefficients,
                 domain):

        if (not all([isinstance(state, (int, float))
                     for state in init_state_vector])
                and len(init_state_vector) == 4
                and isinstance(init_state_vector, (list, tuple))):
            print(init_state_vector)
            raise TypeError

        if (not len(dgl_coefficients) == 3
            and isinstance(dgl_coefficients, (list, tuple))
            and all([isinstance(coef, collections.Callable)
                     or isinstance(coef, (int, float))
                     for coef in dgl_coefficients])):
            raise TypeError

        # if (not isinstance(domain, (np.ndarray, list)) or
        #         not all([isinstance(num, (int, float)) for num in domain])):
        #     raise TypeError

        if isinstance(target_eigenvalue, complex):
            self._eig_val_real = target_eigenvalue.real
            self._eig_val_imag = target_eigenvalue.imag
        elif isinstance(target_eigenvalue, (int, float)):
            self._eig_val_real = target_eigenvalue
            self._eig_val_imag = 0.
        else:
            raise TypeError

        self._init_state_vector = init_state_vector
        self._a2, self._a1, self._a0 = dgl_coefficients
        self._domain = domain

        state_vector = self._transform_eigenfunction()
        self._transf_eig_func_real = state_vector[0]
        self._transf_d_eig_func_real = state_vector[1]
        self._transf_eig_func_imag = state_vector[2]
        self._transf_d_eig_func_imag = state_vector[3]

        Function.__init__(self, self._phi,
                          domain=domain.bounds,
                          derivative_handles=[self._d_phi])

    def _ff(self, y, z):
        a2, a1, a0 = [self._a2, self._a1, self._a0]
        wr = self._eig_val_real
        wi = self._eig_val_imag
        d_y = np.array([y[1],
                        -(a0(z) - wr) / a2(z) * y[0]
                        - a1(z) / a2(z) * y[1] - wi / a2(z) * y[2],
                        y[3],
                        wi / a2(z) * y[0] - (a0(z) - wr) / a2(z) * y[2]
                        - a1(z) / a2(z) * y[3]])
        return d_y

    def _transform_eigenfunction(self):
        eigenfunction = si.odeint(self._ff,
                                  self._init_state_vector,
                                  self._domain.points)
        return eigenfunction.T

    def _phi(self, z):
        return np.interp(z, self._domain, self._transf_eig_func_real)

    def _d_phi(self, z):
        return np.interp(z, self._domain, self._transf_d_eig_func_real)


class AddMulFunction(object):
    """
    (Temporary) Function class which can multiplied with scalars and added with
    functions. Only needed to compute the matrix (of scalars) vector
    (of functions) product in :py:class:`.FiniteTransformFunction`. Will be no
    longer needed when :py:class:`.Function` is overloaded with
    :code:`__add__` and :code:`__mul__` operator.

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
    r"""
    This class provides a transformed :py:class:`.Function` :math:`\bar x(z)`
    through the transformation
    :math:`\bar{\boldsymbol{\xi}} = T * \boldsymbol \xi`, with the function
    vector :math:`\boldsymbol \xi\in\mathbb R^{2n}` and with a given matrix
    :math:`T\in\mathbb R^{2n\times 2n}`. The operator :math:`*` denotes the
    matrix (of scalars) vector (of functions) product. The interim result
    :math:`\bar{\boldsymbol{\xi}}` is a vector

    .. math:: \bar{\boldsymbol{\xi}} = (\bar\xi_{1,0},...,\bar\xi_{1,n-1},
        \bar\xi_{2,0},...,\bar\xi_{2,n-1})^T.

    of functions

    .. math::
        &\bar\xi_{1,j} = \bar x(jl_0 + z),
        \qquad j=0,...,n-1, \quad l_0=l/n, \quad z\in[0,l_0] \\
        &\bar\xi_{2,j} = \bar x(l - jl_0 + z).

    Finally, the provided function :math:`\bar x(z)` is given through
    :math:`\bar\xi_{1,0},...,\bar\xi_{1,n-1}`.

    Note:
        For a more extensive documentation see section 4.2 in:

        - Wang, S. und F. Woittennek: Backstepping-Methode für parabolische
          Systeme mit punktförmigem inneren Eingriff. Automatisierungstechnik,
          2015.
          http://dx.doi.org/10.1515/auto-2015-0023

    Args:
        function (callable):
            Function :math:`x(z)` that will act as start for the generation of
            :math:`2n` Functions :math:`\xi_{i,j}`

            .. math::
                &\bar\xi_{1,j} = x(z + jl_0),
                \qquad j=0,...,n-1, \quad l_0=l/n, \quad z\in[0,l_0] \\
                &\bar\xi_{2,j} = x(z + l - jl_0 ).

            The vector of functions :math:`\boldsymbol\xi` will then be
            constituted as follows:

            .. math:: \boldsymbol\xi = (\xi_{1,0},...,\xi_{1,n-1},
                \xi_{2,0},...,\xi_{2,n-1})^T .

        M (numpy.ndarray): Matrix :math:`T\in\mathbb R^{2n\times 2n}` of
            scalars.
        l (numbers.Number): Length of the domain (:math:`z\in [0,l]`).
    """

    def __init__(self, function, M, l, scale_func=None, nested_lambda=False):

        if not isinstance(function, collections.Callable):
            raise TypeError
        if (not isinstance(M, np.ndarray) or len(M.shape) != 2
                or np.diff(M.shape) != 0 or M.shape[0] % 1 != 0):
            raise TypeError
        if not all([isinstance(num, (int, float)) for num in [l, ]]):
            raise TypeError

        self.function = function
        self.M = M
        self.l = l
        if scale_func is None:
            self.scale_func = lambda z: 1
        else:
            self.scale_func = scale_func

        self.n = int(M.shape[0] / 2)
        self.l0 = l / self.n
        self.z_disc = np.array([(i + 1) * self.l0 for i in range(self.n)])

        if not nested_lambda:
            # iteration mode
            Function.__init__(self, self._call_transformed_func,
                              domain=(0, l), derivative_handles=[])
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
