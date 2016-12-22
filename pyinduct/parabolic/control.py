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
    Consider a domain [0,l] which is divided into the two sub domains [0,a]
    and [a,l] with:

    -the discretization l_0 = l/n

    -and a partition a+b=l.

    respectively k1+k2=n is calculated so that n is odd and a=k1*l_0 is close
    to a_desired modes:

    - 'force_k2_as_prime_number': k2 is an prime number (k1, k2 are coprime)

    - 'coprime': k1, k2 are coprime

    - 'one_even_one_odd': just meet the specification from the doc (default)

    Args:
        n:
        a_desired:
        l:
        mode:
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
        """
        TODO docstring

        Args:
            n:
            num:
        """
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
    Temporary function, as long :py:class:`pyinduct.placeholder.EquationTerm`
    can only be scaled individually.

    Args:
        eqt_list (list):
            List  of :py:class:`pyinduct.placeholder.EquationTerm` s
        factor (numbers.Number): Scale factor.

    Return:
        Scaled copy of :py:class:`pyinduct.placeholder.EquationTerm` s (eqt_list).
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
    # TODO add docstring for this method
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
