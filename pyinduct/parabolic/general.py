import collections
import warnings

import numpy as np
from scipy.optimize import fsolve

from ..core import Domain, Function, find_roots
from ..eigenfunctions import SecondOrderOperator
from ..placeholder import (ScalarFunction, TestFunction, FieldVariable, ScalarTerm,
                           IntegralTerm, Input, Product)
from ..simulation import WeakFormulation

__all__ = ["compute_rad_robin_eigenfrequencies", "eliminate_advection_term", "get_parabolic_dirichlet_weak_form",
           "get_parabolic_robin_weak_form", "get_in_domain_transformation_matrix", "get_adjoint_rad_evp_param"]


def compute_rad_robin_eigenfrequencies(param, l, n_roots=10, show_plot=False):
    """
    Return the first :code:`n_roots` eigenfrequencies :math:`\\omega` (and eigenvalues :math:`\\lambda`)

    .. math:: \\omega = \\sqrt{-\\frac{a_1^2}{4a_2^2}+\\frac{a_0-\\lambda}{a_2}}

    to the eigenvalue problem

    .. math::
        a_2\\varphi''(z) + a_1&\\varphi'(z) + a_0\\varphi(z) = \\lambda\\varphi(z) \\\\
        \\varphi'(0) &= \\alpha\\varphi(0) \\\\
        \\varphi'(l) &= -\\beta\\varphi(l).

    Args:
        param (array_like): :math:`\\Big( a_2, a_1, a_0, \\alpha, \\beta \\Big)^T`
        l (numbers.Number): Right boundary value of the domain :math:`[0,l]\\ni z`.
        n_roots (int): Amount of eigenfrequencies to be compute.
        show_plot (bool): A plot window of the characteristic equation appears if it is :code:`True`.

    Return:
        tuple --> booth tuple elements are numpy.ndarrays of length :code:`nroots`:
            :math:`\\Big(\\big[\\omega_1,...,\\omega_\\text{n\\_roots}\Big], \\Big[\\lambda_1,...,\\lambda_\\text{n\\_roots}\\big]\\Big)`
    """

    a2, a1, a0, alpha, beta = param
    eta = -a1 / 2. / a2

    def characteristic_equation(om):
        if np.round(om, 200) != 0.:
            zero = (alpha + beta) * np.cos(om * l) + ((eta + beta) * (alpha - eta) / om - om) * np.sin(om * l)
        else:
            zero = (alpha + beta) * np.cos(om * l) + (eta + beta) * (alpha - eta) * l - om * np.sin(om * l)
        return zero

    def complex_characteristic_equation(om):
        if np.round(om, 200) != 0.:
            zero = (alpha + beta) * np.cosh(om * l) + ((eta + beta) * (alpha - eta) / om + om) * np.sinh(om * l)
        else:
            zero = (alpha + beta) * np.cosh(om * l) + (eta + beta) * (alpha - eta) * l + om * np.sinh(om * l)
        return zero

    # assume 1 root per pi/l (safety factor = 3)
    om_end = 3 * n_roots * np.pi / l
    start_values = np.arange(0, om_end, .1)
    om = find_roots(characteristic_equation, start_values, 2 * n_roots,
                    rtol=int(np.log10(l) - 6)).tolist()

    # delete all around om = 0
    om.reverse()
    for i in range(np.sum(np.array(om) < np.pi / l / 2e1)):
        om.pop()
    om.reverse()

    # if om = 0 is a root then add 0 to the list
    zero_limit = alpha + beta + (eta + beta) * (alpha - eta) * l
    if np.round(zero_limit, 6 + int(np.log10(l))) == 0.:
        om.insert(0, 0.)

    # regard complex roots
    om_squared = np.power(om, 2).tolist()
    complex_root = fsolve(complex_characteristic_equation, om_end)
    if np.round(complex_root, 6 + int(np.log10(l))) != 0.:
        om_squared.insert(0, -complex_root[0] ** 2)

    # basically complex eigenfrequencies
    om = np.sqrt(np.array(om_squared).astype(complex))

    if len(om) < n_roots:
        raise ValueError("RadRobinEigenvalues.compute_eigen_frequencies()"
                         "can not find enough roots")

    eig_frequencies = om[:n_roots]
    eig_values = a0 - a2 * eig_frequencies ** 2 - a1 ** 2 / 4. / a2
    return eig_frequencies, eig_values


def get_adjoint_rad_evp_param(param):
    """
    Calculates the parameters for the adjoint eigen value problem of the
    reaction-advection-diffusion equation:

    .. math::
        a_2\\varphi''(z) + a_1&\\varphi'(z) + a_0\\varphi(z) = \\lambda\\varphi(z) \\\\
        \\varphi(0) = 0 \\quad &\\text{or} \\quad \\varphi'(0) = \\alpha\\varphi(0) \\\\
        \\varphi`(l) = 0 \\quad &\\text{or} \\quad \\varphi'(l) = -\\beta\\varphi(l)

    with robin and/or dirichlet boundary conditions.

    Args:
        param (array_like): :math:`\\Big( a_2, a_1, a_0, \\alpha, \\beta \\Big)^T`

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


def eliminate_advection_term(param, domain_end=None):
    """
    This method performs a transformation

    .. math:: \\tilde x(z,t)=x(z,t)
            e^{\\int_0^z \\frac{a_1(\\bar z)}{2 a_2}\,d\\bar z} ,

    on the system, which eliminates the advection term :math:`a_1 x(z,t)` from a
    reaction-advection-diffusion equation of the type:

    .. math:: \\dot x(z,t) = a_2 x''(z,t) + a_1(z) x'(z,t) + a_0(z) x(z,t) .

    The boundary can be given by robin

    .. math:: x'(0,t) = \\alpha x(0,t), \\quad x'(l,t) = -\\beta x(l,t) ,

    dirichlet

    .. math:: x(0,t) = 0, \\quad x(l,t) = 0

    or mixed boundary conditions.

    Args:
        param (array_like): :math:`\\Big( a_2, a_1, a_0, \\alpha, \\beta \\Big)^T`
        domain_end (float): upper bound of the spatial domain

    Raises:
        TypeError: If :math:`a_1(z)` is callable but no derivative handle is defined for it.

    Return:
        SecondOrderOperator:

        or

        tuple:
            Parameters :math:`\\big(a_2, \\tilde a_1=0, \\tilde a_0(z), \\tilde \\alpha, \\tilde \\beta \\big)` for
            the transformed system

            .. math:: \\dot{\\tilde{x}}(z,t) = a_2 \\tilde x''(z,t) + \\tilde a_0(z) \\tilde x(z,t)

            and the corresponding boundary conditions (:math:`\\alpha` and/or :math:`\\beta` set to None by dirichlet
            boundary condition).

    """
    # TODO remove this compatibility wrapper and promote use of new Operator
    # class over the entire toolbox.
    if isinstance(param, SecondOrderOperator):
        a2 = param.a2
        a1 = param.a1
        a0 = param.a0
        alpha = -param.alpha0
        beta = param.beta0

    else:
        if not isinstance(param, (tuple, list)) or not len(param) == 5:
            raise TypeError("pyinduct.utils.transform_2_intermediate(): "
                            "argument param must from type tuple or list")

        a2, a1, a0, alpha, beta = param

    if isinstance(a1, Function):
        if not isinstance(a0, collections.Callable):
            a0_z = Function.from_constant(a0)
        else:
            a0_z = a0

        def a0_n(z):
            return a0_z(z) - a1(z) ** 2 / 4 / a2 - a1.derive(1)(z) / 2
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
    elif isinstance(a1, Function):
        beta_n = -a1(domain_end) / 2. / a2 + beta
    else:
        beta_n = -a1 / 2. / a2 + beta

    a2_n = a2
    a1_n = 0

    # TODO see above.
    if isinstance(param, SecondOrderOperator):
        return SecondOrderOperator(a2=a2_n, a1=0, a0=a0_n,
                                   alpha1=param.beta1, alpha0=-alpha_n,
                                   beta1=param.beta1, beta0=beta_n)
    else:
        return a2_n, a1_n, a0_n, alpha_n, beta_n


def get_parabolic_dirichlet_weak_form(init_func_label,
                                      test_func_label,
                                      input_handle,
                                      param,
                                      spatial_domain):
    """
    Return the weak formulation of a parabolic 2nd order system, using an
    inhomogeneous dirichlet boundary at both sides.

    Args:
        init_func_label(str): Label of shape base to use.
        test_func_label(str): Label of test base to use.
        input_handle(:py:class:`pyinduct.simulation.SimulationInput`): Input.
        param(tuple): Parameters of the spatial operator.
        spatial_domain(tuple): Spatial domain of the problem.
        # spatial_domain(:py:class:`pyinduct.core.Domain`): Spatial domain of the
        #  problem.

    Returns:
        :py:class:`pyinduct.simulation.WeakFormulation`: Weak form of the system.
    """
    a2, a1, a0, alpha, beta = param
    l = spatial_domain[1]

    x = FieldVariable(init_func_label)
    x_dt = x.derive(temp_order=1)
    x_dz = x.derive(spat_order=1)
    x_ddz = x.derive(spat_order=2)

    psi = TestFunction(test_func_label)
    psi_dz = psi.derive(1)
    psi_ddz = psi.derive(2)

    # integral terms
    int1 = IntegralTerm(Product(x_dt, psi), spatial_domain)
    int2 = IntegralTerm(Product(x, psi_ddz), spatial_domain, -a2)
    int2h = IntegralTerm(Product(x_ddz, psi), spatial_domain, -a2)
    int3 = IntegralTerm(Product(x, psi_dz), spatial_domain, a1)
    int4 = IntegralTerm(Product(x, psi), spatial_domain, -a0)

    if input_handle is None:
        # homogeneous case
        return WeakFormulation([int1, int2h, int3, int4],
                               name="parabolic_dirichlet_hom")

    # scalar terms
    s1 = ScalarTerm(Product(Input(input_handle), psi_dz(l)), a2)
    s2 = ScalarTerm(Product(Input(input_handle), psi(l)), -a1)
    s3 = ScalarTerm(Product(x_dz(l), psi(l)), -a2)
    s4 = ScalarTerm(Product(x_dz(0), psi(0)), a2)

    return WeakFormulation([int1, int2, int3, int4, s1, s2, s3, s4],
                           name="parabolic_dirichlet")


def get_parabolic_robin_weak_form(shape_base_label, test_base_label, input_handle, param, spatial_domain,
                                  actuation_type_point=None):
    """

    :param shape_base_label:
    :param test_base_label:
    :param input_handle:
    :param param:
    :param spatial_domain:
    :param actuation_type_point:
    :return:
    """
    # TODO What is happening here? -> add documentation.

    if actuation_type_point is None:
        actuation_type_point = spatial_domain[1]

    a2, a1, a0, alpha, beta = param
    l = spatial_domain[1]

    # init ScalarFunction for a1 and a0 to handle spatially varying coefficients
    created_base_labels = ["a0_z", "a1_z"]
    a0_z = ScalarFunction.from_scalars(a0, created_base_labels[0])
    a1_z = ScalarFunction.from_scalars(a1, created_base_labels[1])

    x = FieldVariable(shape_base_label)
    x_dt = x.derive(temp_order=1)
    x_dz = x.derive(spat_order=1)

    psi = TestFunction(test_base_label, order=0)
    psi_dz = psi.derive(1)

    # integral terms
    int1 = IntegralTerm(Product(x_dt, psi), spatial_domain)
    int2 = IntegralTerm(Product(x_dz, psi_dz), spatial_domain, a2)
    int3 = IntegralTerm(Product(Product(x_dz, a1_z), psi), spatial_domain, -1)
    int4 = IntegralTerm(Product(Product(x, a0_z), psi), spatial_domain, -1)

    # scalar terms
    s1 = ScalarTerm(Product(x(0), psi(0)), a2 * alpha)
    s2 = ScalarTerm(Product(x(l), psi(l)), a2 * beta)
    s3 = ScalarTerm(Product(Input(input_handle), psi(actuation_type_point)), -a2)

    # derive state-space system
    weak_form = WeakFormulation([int1, int2, int3, int4, s1, s2, s3], name="parabolic_robin_{}".format(param))
    return weak_form, created_base_labels


def get_in_domain_transformation_matrix(k1, k2, mode='n_plus_1'):
    """
    Returns the transformation matrix M. M is one part of a transformation

    .. :math::
        `x = My + Ty` ,

    where x is the field variable of an interior point controlled parabolic system
    and y is the field variable of an boundary controlled parabolic system.
    T is a (Fredholm-) integral transformation (which can be approximated with M).

    Args:
        k1:
        k2:
        mode: Available modes:

            - 'n_plus_1': M.shape = (n+1,n+1), w = (w(0),...,w(n))^T, w \in {x,y}

            - '2n': M.shape = (2n,2n), w = (w(0),...,w(n),...,w(1))^T, w \in {x,y}

    Return:
        numpy.array: Transformation matrix M.
    """
    if not all(isinstance(i, (int, float)) for i in [k1, k2]):
        raise TypeError("TypeErrorMessage")
    if not all(i % 1 == 0 for i in [k1, k2]):
        raise TypeError("TypeErrorMessage")
    n = k1 + k2
    if k1 + k2 != n or n < 2 or k1 < 0 or k2 < 0:
        raise ValueError("The sum of two positive integers k1 and k2 must be n.")
    if (k1 != 0 and k2 != 0) and n % 2 == 0:
        warnings.warn("Transformation matrix M is not invertible.")

    mod_diag = lambda n, k: np.diag(np.ones(n - np.abs(k)), k)

    if mode == 'n_plus_1':
        M = np.zeros((n + 1, n + 1))
        if k2 < n:
            M += mod_diag(n + 1, k2) + mod_diag(n + 1, -k2)
        if k2 != 0:
            M += np.fliplr(mod_diag(n + 1, n - k2) + mod_diag(n + 1, -n + k2))
    elif mode == '2n':
        M = mod_diag(2 * n, k2) + mod_diag(2 * n, -k2) + mod_diag(2 * n, n + k1) + mod_diag(2 * n, -n - k1)
    else:
        raise ValueError("String in variable 'mode' not understood.")
    return M * 0.5
