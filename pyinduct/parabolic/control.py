import collections
import copy as cp
from numbers import Number

import numpy as np

from ..control import Controller
from ..placeholder import ScalarTerm, IntegralTerm
from ..simulation import SimulationInput, SimulationInputSum, WeakFormulation

__all__ = ["get_parabolic_robin_backstepping_controller", "split_domain"]


def split_domain(n, a_desired, l, mode='coprime'):
    """
    Consider a domain :math:`[0,l]` which is divided into the two sub domains
    :math:`[0,a]` and :math:`[a,l]` with the discretization :math:`l_0 = l/n`
    and a partition :math:`a + b = l`.

    Calculate two numbers :math:`k_1` and :math:`k_2` with :math:`k_1 + k_2 = n`
    such that :math:`n` is odd and :math:`a = k_1l_0` is close to ``a_desired``.

    Args:
        n (int): Number of sub-intervals to create (must be odd).
        a_desired (float): Desired partition size :math:`a` .
        l (float): Length :math:`l` of the interval.
        mode (str): Operation mode to use:

            - 'coprime': :math:`k_1` and :math:`k_2` are coprime (default) .
            - 'force_k2_as_prime_number': :math:`k_2` is a prime number
              (:math:`k_1` and :math:`k_2` are coprime)
            - 'one_even_one_odd': One is even and one is odd.
    """

    if not isinstance(n, (int, float)):
        raise TypeError("Integer excepted.")
    if not int(n) - n == 0:
        raise TypeError("n must be a natural number")
    if n % 2 == 0:
        raise ValueError("n must be odd.")
    else:
        n = int(n)
    if l <= 0:
        raise ValueError("l can not be <= 0")
    if not 0. <= a_desired <= l:
        raise ValueError("a_desired not in interval (0,l).")
    elif a_desired == 0.:
        return 0, n, 0, None, None
    elif a_desired == l:
        return n, 0, a_desired, None, None

    def get_candidate_tuple(n, num):
        k1 = (n - num)
        k2 = num
        ratio = k1 / k2
        a = (ratio * l / (1 + ratio))
        b = l - a
        diff = np.abs(a_desired - a)
        return k1, k2, a, b, ratio, diff

    cand = list()
    for num in range(1, 3):
        cand.append(get_candidate_tuple(n, num))

    if mode == 'force_k2_as_prime_number':
        for k2_prime_num in range(3, n, 2):
            if all(k2_prime_num % i != 0 for i in range(2, int(np.sqrt(k2_prime_num)) + 1)):
                cand.append(get_candidate_tuple(n, k2_prime_num))
    elif mode == 'coprime':
        for k2_coprime_to_k1 in range(3, n):
            if all(not (k2_coprime_to_k1 % i == 0 and (n - k2_coprime_to_k1) % i == 0)
                   for i in range(3, min(k2_coprime_to_k1, n - k2_coprime_to_k1) + 1, 2)):
                cand.append(get_candidate_tuple(n, k2_coprime_to_k1))
    elif mode == 'one_even_one_odd':
        for k2_num in range(1, n):
            cand.append(get_candidate_tuple(n, k2_num))
    else:
        raise ValueError("String in variable 'mode' not understood.")

    diffs = [i[5] for i in cand]
    diff_min = min(diffs)
    min_index = diffs.index(diff_min)

    return cand[min_index]


def scale_equation_term_list(eqt_list, factor):
    """
    Temporary function, as long :py:class:`.EquationTerm`
    can only be scaled individually.

    Args:
        eqt_list (list):
            List  of :py:class:`.EquationTerm`'s
        factor (numbers.Number): Scale factor.

    Return:
        Scaled copy of :py:class:`.EquationTerm`'s (eqt_list).
    """
    if not isinstance(eqt_list, list):
        raise TypeError
    if not all([isinstance(term, (ScalarTerm, IntegralTerm)) for term in eqt_list]):
        raise TypeError
    if not isinstance(factor, Number):
        raise TypeError

    eqt_list_copy = cp.deepcopy(eqt_list)
    for term in eqt_list_copy:
        term.scale *= factor

    return eqt_list_copy


def get_parabolic_robin_backstepping_controller(state,
                                                approx_state,
                                                d_approx_state,
                                                approx_target_state,
                                                d_approx_target_state,
                                                integral_kernel_zz,
                                                original_beta,
                                                target_beta,
                                                scale=None):
    r"""
    Build a modal approximated backstepping controller
    :math:`u(t)=(Kx)(t)`, for the (open loop-) diffusion system with reaction
    and advection term, robin boundary condition and robin actuation

    .. math::
        :nowrap:

        \begin{align*}
            \dot x(z,t) &= a_2 x''(z,t) + a_1 x'(z,t) + a_0 x(z,t),
             && z\in (0, l) \\
            x'(0,t) &= \alpha x(0,t) \\
            x'(l,t) &= -\beta x(l,t) + u(t)
        \end{align*}

    such that the closed loop system has the desired dynamic of the
    target system

    .. math::
        :nowrap:

        \begin{align*}
            \dot{\bar{x}}(z,t) &= a_2 \bar x''(z,t) + \bar a_1 \bar x'(z,t) +
            \bar a_0 \bar x(z,t), && z\in (0, l) \\
            \bar x'(0,t) &= \bar\alpha \bar x(0,t) \\
            \bar x'(l,t) &= -\bar\beta x(l,t)
        \end{align*}

    where :math:`\bar a_1,\, \bar a_0,\, \bar\alpha,\, \bar\beta` are controller
    parameters.

    The control design is performed using the backstepping method, whose
    integral transform

    .. math:: \bar x(z) = x(z) + \int_0^z k(z, \bar z) x(\bar z) \, d\bar z

    maps from the original system to the target system.

    Note:
        For more details see the example script
        :py:mod:`pyinduct.examples.rad_eq_const_coeff` that
        implements the example from [WoiEtAl17]_ .

    Args:
        state (list of :py:class:`.ScalarTerm`'s): Measurement / value from
            simulation of :math:`x(l)`.
        approx_state (list of :py:class:`.ScalarTerm`'s): Modal approximated
            :math:`x(l)`.
        d_approx_state (list of :py:class:`.ScalarTerm`'s): Modal approximated
            :math:`x'(l)`.
        approx_target_state (list of :py:class:`.ScalarTerm`'s): Modal
            approximated :math:`\bar x(l)`.
        d_approx_target_state (list of :py:class:`.ScalarTerm`'s): Modal
            approximated :math:`\bar x'(l)`.
        integral_kernel_zz (:py:class:`numbers.Number`): Integral kernel
            evaluated at :math:`\bar z = z = l` :

            .. math:: k(l, l) = \bar\alpha
                                - \alpha
                                + \frac{a_0-\bar a_0}{a_2} l \:.
        original_beta (:py:class:`numbers.Number`): Coefficient :math:`\beta`
            of the original system.
        target_beta (:py:class:`numbers.Number`): Coefficient :math:`\bar\beta`
            of the target system.
        scale (:py:class:`numbers.Number`): A constant :math:`c \in \mathbb R`
            to scale the control law: :math:`u(t) = c \, (Kx)(t)`.

    Returns:
        :py:class:`.Controller`: :math:`(Kx)(t)`

    .. [WoiEtAl17] Frank Woittennek, Marcus Riesmeier and Stefan Ecklebe;
              On approximation and implementation of transformation based
              feedback laws for distributed parameter systems;
              IFAC World Congress, 2017, Toulouse
    """
    args = [state, approx_state, d_approx_state, approx_target_state, d_approx_target_state]
    if not all([isinstance(arg, list) for arg in args]):
        raise TypeError
    terms = state + approx_state + d_approx_state + approx_target_state + d_approx_target_state
    if not all([isinstance(term, (ScalarTerm, IntegralTerm)) for term in terms]):
        raise TypeError
    if not all([isinstance(num, Number) for num in [original_beta, target_beta, integral_kernel_zz]]):
        raise TypeError
    if not isinstance(scale, (Number, type(None))):
        raise TypeError

    beta = original_beta
    beta_t = target_beta

    unsteady_term = scale_equation_term_list(state, beta - beta_t
                                             - integral_kernel_zz)

    first_sum_1st_term = scale_equation_term_list(approx_target_state, -beta_t)
    first_sum_2nd_term = scale_equation_term_list(approx_state, beta_t)

    second_sum_1st_term = scale_equation_term_list(d_approx_target_state, -1)
    second_sum_2nd_term = scale_equation_term_list(d_approx_state, 1)
    second_sum_3rd_term = scale_equation_term_list(approx_state,
                                                   integral_kernel_zz)

    control_law = (unsteady_term
                   + first_sum_1st_term
                   + first_sum_2nd_term
                   + second_sum_1st_term
                   + second_sum_2nd_term
                   + second_sum_3rd_term)

    if scale is not None:
        scaled_control_law = scale_equation_term_list(control_law, scale)
    else:
        scaled_control_law = control_law

    c_name = "parabolic_robin_backstepping_controller"
    return Controller(WeakFormulation(scaled_control_law, name=c_name))


# TODO: change to factory, rename: function_wrapper
# def _convert_to_function(coef):
#     if not isinstance(coef, collections.Callable):
#         return lambda z: coef
#     else:
#         return coef
