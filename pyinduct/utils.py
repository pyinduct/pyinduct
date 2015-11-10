from __future__ import division
from numbers import Number
import warnings
import copy as cp

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve, root
import pyqtgraph as pg

from pyinduct import get_initial_functions, register_functions
import placeholder as ph
from core import back_project_from_base
from shapefunctions import LagrangeFirstOrder
from placeholder import FieldVariable, TestFunction
from visualization import EvalData

__author__ = 'Stefan Ecklebe'


def find_roots(function, n_roots, area_end, rtol, points_per_root=10, atol=1e-7, show_plot=False, cmplx=False):
    """
    Searches roots of the given function in the interval [0, area_end] and checks them with aid of rtol for uniqueness.
    It will return the exact amount of roots given by n_roots or raise ValueError.

    It is assumed that functions roots are distributed approximately homogeneously, if that is not the case you should
    increase the keyword-argument points_per_root.

    :param function: function handle for f(x) whose roots shall be found
    :param n_roots: number of roots to find
    :param area_end: end of interval to search in
    :param rtol: magnitude to be exceeded for the difference of two roots to be unique f(r1) - f(r2) > 10^rtol
    :param points_per_root: number of solver start-points around each root
    :param atol: absolute tolerance to zero  f(root) < atol
    :param show_plot: shows a debug plot containing the given functions behavior completed by the extracted roots
    :return: numpy.ndarray of roots

    In Detail fsolve is used to find initial candidates for roots of f(x). If a root satisfies the criteria given
    by atol and rtol it is added. If it is already in the list, a comprehension between the already present entries
    error and the current error is performed. If the newly calculated root comes with a smaller error it supersedes
    the present entry.
    """
    positive_numbers = [n_roots, points_per_root, area_end, atol]
    integers = [n_roots, points_per_root, rtol]
    if not callable(function):
        raise TypeError("callable handle is needed")
    if not all([isinstance(num, int) for num in integers]):
        raise TypeError("n_roots, points_per_root and rtol must be of type int")
    if any([num <= 0 for num in positive_numbers]):
        raise ValueError("n_roots, points_per_root, area_end and atol must be positive")
    if not isinstance(show_plot, bool):
        raise TypeError("show_plot must be of type bool")

    if cmplx:
        dim = 2
    else:
        dim = 1

    roots = np.empty((n_roots, dim))
    rounded_roots = np.empty((n_roots, dim))
    errors = np.empty((n_roots, dim))
    found_roots = 0

    values = np.arange(0, area_end, rtol*.1)
    val = iter(values)
    if cmplx:
        re_vals, im_vals = np.meshgrid(values, values)
        cvalues = np.vstack([re_vals.flatten(), im_vals.flatten()]).T
        val = iter(cvalues)

    while found_roots < n_roots:
        try:
            calculated_root, info, ier, msg = fsolve(function, val.next(), full_output=True)
        except StopIteration:
            break

        if not cmplx:
            calculated_root = [calculated_root]

        error = np.abs(info['fvec'])
        if all(error > atol):
            continue
        if any(calculated_root < 0):
            continue

        rounded_root = np.round(calculated_root, -rtol)
        cmp_arr = [a and b for a, b in rounded_root == rounded_roots]
        if any(cmp_arr):
            idx = cmp_arr.index(True)
            if all(errors[idx] > error):
                roots[idx] = calculated_root
                errors[idx] = error
            continue

        roots[found_roots] = calculated_root
        rounded_roots[found_roots] = rounded_root
        errors[found_roots] = error

        found_roots += 1

        # if found_roots == n_roots:
        #     break

    if found_roots < n_roots:
        raise ValueError("Insufficient number of roots detected. ({0} < {1}) "
                         "Try to increase the area to search in.".format(found_roots, n_roots))

    good_roots = np.sort(roots, 0)[:n_roots]

    if show_plot:
        pw = pg.plot(title="function + roots")
        if cmplx:
            res = function(cvalues)
            pw.plot(values, res[0, :])
            pw.plot(values, res[1, :])
        else:
            vec_function = np.vectorize(function, otypes=[np.float])
            pw.plot(values, vec_function(values))
            pw.plot(good_roots, vec_function(good_roots), pen=None, symbolPen=pg.mkPen("g"))

        pg.QtGui.QApplication.instance().exec_()

    return good_roots


def evaluate_placeholder_function(placeholder, input_values):
    """
    evaluate a given placeholder object, that contains functions

    :param placeholder: instance of ref:py:class: FieldVariable or ref:py:class TestFunction ref:py:class ScalarFunction
    :return: results as np.ndarray
    """
    if not isinstance(placeholder, (FieldVariable, TestFunction)):
        raise TypeError("Input Object not supported!")

    funcs = get_initial_functions(placeholder.data["func_lbl"], placeholder.order[1])
    return np.array([func(input_values) for func in funcs])


def evaluate_approximation(weights, function_label, temporal_steps, spatial_interval, spatial_step, spat_order=0):
    """
    evaluate an approximation given by weights and functions at the points given in spatial and temporal steps

    :param weights: 2d np.ndarray where axis 1 is the weight index and axis 0 the temporal index
    :param function_label: functions to use for back-projection
    :return:
    """
    funcs = get_initial_functions(function_label, spat_order)
    if weights.shape[1] != funcs.shape[0]:
        raise ValueError("weights have to fit provided functions!")

    step_cnt = int((spatial_interval[1] - spatial_interval[0])/ spatial_step)
    spatial_steps = np.linspace(spatial_interval[0], spatial_interval[1], step_cnt)

    def eval_spatially(weight_vector):
        if isinstance(function_label[0], LagrangeFirstOrder):
            # shortcut for fem approximations
            nodes = [func.top for func in funcs]
            handle = interp1d(nodes, weight_vector)
        else:
            handle = back_project_from_base(weight_vector, funcs)
        return handle(spatial_steps)

    data = np.apply_along_axis(eval_spatially, 1, weights)
    return EvalData([temporal_steps, spatial_steps], data)


def split_domain(n, a_desired, l, mode='coprime'):
    """
    Consider a domain [0,l] which is divided into two subdomains [0,a] and [a,l].
    With the dicretisation l_0 = l/n an partion a+b=l respectivly k1+k2=n
    is provided such that n is odd and a=k1*l_0 is close to a_desired.
    modes:
        - 'force_k2_as_prime_number': k2 is an prime number (k1,k2 are coprime)
        - 'coprime': k1,k2 are coprime
        - 'one_even_one_odd': just meet the specification from the doc (default)
    :param n:
    :param a_desired:
    :param l:
    :param mode:
    :return:
    """

    if not isinstance(n, (long, int, float)):
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
        :param n:
        :param num:
        :return:
        """
        k1 = (n - num)
        k2 = num
        ratio = k1 / k2
        a = (ratio * l / (1 + ratio))
        b = l - a
        diff = np.abs(a_desired - a)
        return k1, k2, a, b, ratio, diff

    cand = list()
    for num in xrange(1, 3):
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


def get_inn_domain_transformation_matrix(k1, k2, mode='n_plus_1'):
    """
    Returns the transformation matrix M. M is one part of a transformation
    x = My + Ty
    where x is the field variable of an interior point controlled parabolic system
    and y is the field variable of an boundary controlled parabolic system.
    T is a (Fredholm-) integral transformation (which can be approximated with M).
    Invertibility of M:
    - ensured if n is odd
    - ensured if k1=0 or k2=0
    - for even k1 and k2, given in some cases (further condition not known)
    - not given if k1 and k2 are odd
    - not given if k1=k2.
    modes:
        - 'n_plus_1': M.shape = (n+1,n+1), w = (w(0),...,w(n))^T, w \in {x,y}
        - '2n': M.shape = (2n,2n), w = (w(0),...,w(n),...,w(1))^T, w \in {x,y}
    :param n:
    :param k1:
    :param k2:
    :param mode:
    :return:
    """
    if not all(isinstance(i, (int, long, float)) for i in [k1, k2]):
        raise TypeError("TypeErrorMessage")
    if not all(i%1 == 0 for i in [k1, k2]):
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
    return M*0.5


def scale_equation_term_list(eqt_list, factor):
    """
    Temporary function, as long pyinduct.placeholder.EquationTerm can only be scaled individually.
    Return a scaled copy of eqt_list.
    :param eqt_list: list of EquationTerms: [scalar_term_1, integral_term_1, ....]
    :param factor: isinstance from numbers.Number
    :return:
    """
    if not isinstance(eqt_list, list):
        raise TypeError
    if not all([isinstance(term, (ph.ScalarTerm, ph.IntegralTerm)) for term in  eqt_list]):
        raise TypeError
    if not isinstance(factor, Number):
        raise TypeError

    eqt_list_copy = cp.deepcopy(eqt_list)
    for term in eqt_list_copy:
        term.scale *= factor

    return eqt_list_copy


def get_parabolic_robin_backstepping_controller(state, approx_state, d_approx_state, approx_target_state,
                                                d_approx_target_state, integral_kernel_zz, original_boundary_param,
                                                target_boundary_param, trajectory=None, scale=None):
    args = [state, approx_state, d_approx_state, approx_target_state, d_approx_target_state]
    import control as ct
    import simulation as sim
    if not all([isinstance(arg, list) for arg in args]):
        raise TypeError
    terms = state + approx_state + d_approx_state + approx_target_state + d_approx_target_state
    if not all([isinstance(term, (ph.ScalarTerm, ph.IntegralTerm)) for term in  terms]):
        raise TypeError
    if not all([isinstance(num, Number) for num in original_boundary_param+target_boundary_param+(integral_kernel_zz,)]):
        raise TypeError
    if not isinstance(scale, (Number, type(None))):
        raise TypeError
    if not isinstance(trajectory, (sim.SimulationInput, type(None))):
        raise TypeError


    alpha, beta = original_boundary_param
    alpha_t, beta_t = target_boundary_param

    unsteady_term = scale_equation_term_list(state, beta - beta_t - integral_kernel_zz)
    first_sum_1st_term = scale_equation_term_list(approx_target_state, -beta_t)
    first_sum_2nd_term = scale_equation_term_list(approx_state, beta_t)
    second_sum_1st_term = scale_equation_term_list(d_approx_target_state, -1)
    second_sum_2nd_term = scale_equation_term_list(d_approx_state, 1)
    second_sum_3rd_term = scale_equation_term_list(approx_state, integral_kernel_zz)

    control_law = unsteady_term + first_sum_1st_term + first_sum_2nd_term + \
                   second_sum_1st_term + second_sum_2nd_term + second_sum_3rd_term

    if not scale is None:
        scaled_control_law = scale_equation_term_list(control_law, scale)
        if not trajectory is None:
            trajectory.scale *= scale
    else:
        scaled_control_law = control_law

    if not trajectory is None:
        return sim.Mixer([trajectory, ct.Controller(ct.ControlLaw(scaled_control_law))])
    else:
        return sim.Mixer([ct.Controller(ct.ControlLaw(scaled_control_law))])


def _convert_to_function(coef):
    if not callable(coef):
        return lambda z: coef
    else:
        return coef


def _convert_to_scalar_function(coef, label):
    import core as cr
    if not callable(coef):
        register_functions(label, cr.Function(lambda z: coef), overwrite=True)
    elif isinstance(coef, cr.Function):
        register_functions(label, coef, overwrite=True)
    else:
        register_functions(label, cr.Function(coef), overwrite=True)
    return ph.ScalarFunction(label)


def get_parabolic_dirichlet_weak_form(init_func_label, test_func_label, input, param, spatial_domain):
    import simulation as sim
    a2, a1, a0, alpha, beta = param
    l = spatial_domain[1]
    # integral terms
    int1 = ph.IntegralTerm(ph.Product(ph.TemporalDerivedFieldVariable(init_func_label, order=1),
                                      ph.TestFunction(test_func_label, order=0)), spatial_domain)
    int2 = ph.IntegralTerm(ph.Product(ph.SpatialDerivedFieldVariable(init_func_label, order=0),
                                      ph.TestFunction(test_func_label, order=2)), spatial_domain, -a2)
    int3 = ph.IntegralTerm(ph.Product(ph.SpatialDerivedFieldVariable(init_func_label, order=0),
                                      ph.TestFunction(test_func_label, order=1)), spatial_domain, a1)
    int4 = ph.IntegralTerm(ph.Product(ph.SpatialDerivedFieldVariable(init_func_label, order=0),
                                      ph.TestFunction(test_func_label, order=0)), spatial_domain, -a0)
    # scalar terms
    s1 = ph.ScalarTerm(ph.Product(ph.Input(input),
                                  ph.TestFunction(test_func_label, order=1, location=l)), a2)
    s2 = ph.ScalarTerm(ph.Product(ph.Input(input),
                                  ph.TestFunction(test_func_label, order=0, location=l)), -a1)
    s3 = ph.ScalarTerm(ph.Product(ph.SpatialDerivedFieldVariable(init_func_label, order=1, location=l),
                                  ph.TestFunction(test_func_label, order=0, location=l)), -a2)
    s4 = ph.ScalarTerm(ph.Product(ph.SpatialDerivedFieldVariable(init_func_label, order=1, location=0),
                                  ph.TestFunction(test_func_label, order=0, location=0)), a2)

    # derive state-space system
    return sim.WeakFormulation([int1, int2, int3, int4, s1, s2, s3, s4])

def get_parabolic_robin_weak_form(init_func_label, test_func_label, input, param, spatial_domain, actuation_point=None):
    import simulation as sim
    if actuation_point == None:
        actuation_point = spatial_domain[1]
    a2, a1, a0, alpha, beta = param
    l = spatial_domain[1]
    # init ph.ScalarFunction for a1 and a0, to handle spatially varying coefficients
    # a2 = _convert_to_scalar_function(a2, "a2_z")
    a1_z = _convert_to_scalar_function(a1, "a1_z")
    a0_z = _convert_to_scalar_function(a0, "a0_z")

    # integral terms
    int1 = ph.IntegralTerm(ph.Product(ph.TemporalDerivedFieldVariable(init_func_label, order=1),
                                      ph.TestFunction(test_func_label, order=0)), spatial_domain)
    int2 = ph.IntegralTerm(ph.Product(ph.SpatialDerivedFieldVariable(init_func_label, order=1),
                                      ph.TestFunction(test_func_label, order=1)), spatial_domain, a2)
    int3 = ph.IntegralTerm(ph.Product(ph.Product(ph.SpatialDerivedFieldVariable(init_func_label, order=1), a1_z),
                                      ph.TestFunction(test_func_label, order=0)), spatial_domain, -1)
    int4 = ph.IntegralTerm(ph.Product(ph.Product(ph.SpatialDerivedFieldVariable(init_func_label, order=0), a0_z),
                                      ph.TestFunction(test_func_label, order=0)), spatial_domain, -1)

    # scalar terms
    s1 = ph.ScalarTerm(ph.Product(ph.SpatialDerivedFieldVariable(init_func_label, order=0, location=0),
                                  ph.TestFunction(test_func_label, order=0, location=0)), a2*alpha)
    s2 = ph.ScalarTerm(ph.Product(ph.SpatialDerivedFieldVariable(init_func_label, order=0, location=l),
                                  ph.TestFunction(test_func_label, order=0, location=l)), a2*beta)
    s3 = ph.ScalarTerm(ph.Product(ph.Input(input),
                                  ph.TestFunction(test_func_label, order=0, location=actuation_point)), -a2)
    # derive state-space system
    return sim.WeakFormulation([int1, int2, int3, int4, s1, s2, s3])


def find_nearest_idx(array, value):
    return (np.abs(array-value)).argmin()
