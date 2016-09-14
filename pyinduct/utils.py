"""
A few helper functions for users and developer.
"""

from subprocess import run, call
import copy as cp
import os
import warnings
from numbers import Number
import collections
import numpy as np
from scipy.optimize import root
from .registry import get_base, register_base
from . import placeholder as ph
from .placeholder import FieldVariable, TestFunction
from . import visualization as vis
import pyqtgraph as pg


class Parameters:
    """
    Empty class to pass system parameters.
    """

    def __init__(self):
        pass


def complex_wrapper(func):
    """
    Wraps complex valued function into 2 dimensional function for easier handling.

    Args:
        func (callable):

    Return:
        2dim function handle, taking x = (re(x), im(x) and returning [re(func(x), im(func(x)].
    """

    def wrapper(x):
        return np.array([np.real(func(np.complex(x[0], x[1]))), np.imag(func(np.complex(x[0], x[1])))])

    return wrapper


def find_roots(function, n_roots, grid, rtol=0, atol=1e-7, show_plot=False, complex=False, get_all=False):
    """
    Searches roots of the given function in the interval [0, area_end] and checks them with aid of rtol for uniqueness.
    It will return the exact amount of roots given by n_roots or raise ValueError.
    It is assumed that functions roots are distributed approximately homogeneously, if that is not the case you should
    increase the keyword-argument points_per_root.

    In Detail the function scipy.optimize.root is used to find initial candidates for roots of f(x). If a root
    satisfies the criteria given by atol and rtol it is added. If it is already in the list, a comprehension between
    the already present entries error and the current error is performed. If the newly calculated root comes with a
    smaller error it supersedes the present entry.

    Args:
        function: Function handle for f(x) whose roots shall be found.
        n_roots: Number of roots to find.
        grid: numpy.ndarray (first dimension should fit the input dimension of the provided func) of values where to
            start searching.
        rtol: Magnitude to be exceeded for the difference of two roots to be unique f(r1) - f(r2) > 10^rtol.
        atol: Absolute tolerance to zero  f(root) < atol.
        show_plot: Shows a debug plot containing the given functions behavior completed by the extracted roots.
        get_all: You get all N found roots (if True) even if N < n_roots.
    Return:
        numpy.ndarray of roots.
    """
    # positive_numbers = [n_roots, points_per_root, area, atol]
    # integers = [n_roots, points_per_root, rtol]
    # if not callable(function):
    #     raise TypeError("callable handle is needed")
    # if not all([isinstance(num, int) for num in integers]):
    #     raise TypeError("n_roots, points_per_root and rtol must be of type int")
    # if any([num <= 0 for num in positive_numbers]):
    #     raise ValueError("n_roots, points_per_root, area_end and atol must be positive")
    # if not isinstance(show_plot, bool):
    #     raise TypeError("show_plot must be of type bool")

    if isinstance(grid[0], Number):
        grid = [grid]

    dim = len(grid)
    if complex:
        assert dim == 2
        function = complex_wrapper(function)

    roots = np.full((n_roots, dim), np.nan)
    rounded_roots = np.full((n_roots, dim), np.nan)
    errors = np.full((n_roots,), np.nan)
    found_roots = 0

    grids = np.meshgrid(*[row for row in grid])
    values = np.vstack([arr.flatten() for arr in grids]).T

    # iterate over test_values
    val = iter(values)
    while found_roots < n_roots:
        try:
            res = root(function, next(val), tol=atol)
            # calculated_root, info, ier, msg = fsolve(function, val.next(), full_output=True)
        except StopIteration:
            break

        if not res.success:
            continue

        calculated_root = np.atleast_1d(res.x)

        error = np.linalg.norm(res.fun)

        # check for absolute tolerance
        if error > atol:
            continue

        # check if root lies in expected area
        abort = False
        for rt, ar in zip(calculated_root, grid):
            if ar.min() > rt or ar.max() < rt:
                abort = True
                break
        if abort:
            continue

        # check whether root is already present in cache
        rounded_root = np.round(calculated_root, -rtol)
        cmp_arr = [all(bools) for bools in rounded_root == rounded_roots[:found_roots]]
        if any(cmp_arr):
            idx = cmp_arr.index(True)
            if errors[idx] > error:
                roots[idx] = calculated_root
                errors[idx] = error
            # TODO check jacobian (if provided) to identify roots of higher order
            continue

        roots[found_roots] = calculated_root
        rounded_roots[found_roots] = rounded_root
        errors[found_roots] = error

        found_roots += 1

    valid_roots = roots[:found_roots]

    # sort roots
    idx = np.argsort(valid_roots[:, 0])
    good_roots = valid_roots[idx, :]

    if show_plot:
        pw = pg.plot(title="function + roots")
        if complex:
            pw.plot(good_roots[:, 0], good_roots[:, 1], pen=None, symbolPen=pg.mkPen("g"))
            # results = np.linalg.norm(function(values), axis=0)
            # pw.plot(grids.flatten, np.real(results), pen=pg.mkPen("b"))
            # pw.plot(grids.flatten, np.imag(results), pen=pg.mkPen("b", style=pg.QtCore.Qt.DashLine))
            # pw.plot(np.real(good_roots), np.real(results), pen=None, symbolPen=pg.mkPen("g"))
            # pw.plot(np.imag(good_roots), np.imag(results), pen=None, symbolPen=pg.mkPen("g"))
        else:
            if dim == 1:
                results = function(grids)
                colors = vis.create_colormap(len(grids))
                for idx, (intake, output) in enumerate(zip(grids, results)):
                    pw.plot(intake.flatten(), output.flatten(), pen=pg.mkPen(colors[idx]))
                    pw.plot(np.hstack([good_roots, function(good_roots)]), pen=None, symbolPen=pg.mkPen("g"))

        pg.QtGui.QApplication.instance().exec_()

    if not get_all and found_roots < n_roots:
        raise ValueError("Insufficient number of roots detected. ({0} < {1}) "
                         "Try to increase the area to search in.".format(found_roots, n_roots))

    if complex:
        return good_roots[:, 0] + 1j * good_roots[:, 1]

    if dim == 1:
        return good_roots.flatten()

    return good_roots


def evaluate_placeholder_function(placeholder, input_values):
    """
    Evaluate a given placeholder object, that contains functions.

    Args:
        placeholder: Instance of :py:class:`FieldVariable`, :py:class:`TestFunction` or :py:class:`ScalarFunction`.
        input_values: Values to evaluate at.

    Return:
        :py:obj:`numpy.ndarray` of results.
    """
    if not isinstance(placeholder, (FieldVariable, TestFunction)):
        raise TypeError("Input Object not supported!")

    funcs = get_base(placeholder.data["func_lbl"], placeholder.order[1])
    return np.array([func(input_values) for func in funcs])


def split_domain(n, a_desired, l, mode='coprime'):
    """
    Consider a domain [0,l] which is divided into the two sub domains [0,a] and [a,l]
    with:

    -the discretization l_0 = l/n

    -and a partition a+b=l.

    respectively k1+k2=n is calculated so that n is odd and a=k1*l_0 is close to a_desired modes:

    - 'force_k2_as_prime_number': k2 is an prime number (k1,k2 are coprime)

    - 'coprime': k1,k2 are coprime

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
            if all(not (k2_coprime_to_k1 % i == 0 and (n - k2_coprime_to_k1) % i == 0) for i in
                   range(3, min(k2_coprime_to_k1, n - k2_coprime_to_k1) + 1, 2)):
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


# TODO: rename (inn)
def get_inn_domain_transformation_matrix(k1, k2, mode='n_plus_1'):
    """
    Returns the transformation matrix M. M is one part of a transformation

    x = My + Ty

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


def scale_equation_term_list(eqt_list, factor):
    """
    Temporary function, as long :py:class:`pyinduct.placeholder.EquationTerm` can only be scaled individually.
    Return a scaled copy of eqt_list.

    Args:
        eqt_list (list):
            List  of :py:class:`pyinduct.placeholder.EquationTerm` s
        factor (numbers.Number): Scale factor.

    Return:
        Scaled copy of :py:class:`pyinduct.placeholder.EquationTerm` s (eqt_list).
    """
    if not isinstance(eqt_list, list):
        raise TypeError
    if not all([isinstance(term, (ph.ScalarTerm, ph.IntegralTerm)) for term in eqt_list]):
        raise TypeError
    if not isinstance(factor, Number):
        raise TypeError

    eqt_list_copy = cp.deepcopy(eqt_list)
    for term in eqt_list_copy:
        term.scale *= factor

    return eqt_list_copy


def get_parabolic_robin_backstepping_controller(state, approx_state, d_approx_state, approx_target_state,
                                                d_approx_target_state, integral_kernel_zz, original_beta, target_beta,
                                                trajectory=None, scale=None):
    args = [state, approx_state, d_approx_state, approx_target_state, d_approx_target_state]
    from . import control as ct
    from . import simulation as sim
    if not all([isinstance(arg, list) for arg in args]):
        raise TypeError
    terms = state + approx_state + d_approx_state + approx_target_state + d_approx_target_state
    if not all([isinstance(term, (ph.ScalarTerm, ph.IntegralTerm)) for term in terms]):
        raise TypeError
    if not all([isinstance(num, Number) for num in [original_beta, target_beta, integral_kernel_zz]]):
        raise TypeError
    if not isinstance(scale, (Number, type(None))):
        raise TypeError
    if not isinstance(trajectory, (sim.SimulationInput, type(None))):
        raise TypeError

    beta = original_beta
    beta_t = target_beta

    unsteady_term = scale_equation_term_list(state, beta - beta_t - integral_kernel_zz)
    first_sum_1st_term = scale_equation_term_list(approx_target_state, -beta_t)
    first_sum_2nd_term = scale_equation_term_list(approx_state, beta_t)
    second_sum_1st_term = scale_equation_term_list(d_approx_target_state, -1)
    second_sum_2nd_term = scale_equation_term_list(d_approx_state, 1)
    second_sum_3rd_term = scale_equation_term_list(approx_state, integral_kernel_zz)

    control_law = unsteady_term + first_sum_1st_term + first_sum_2nd_term + second_sum_1st_term + second_sum_2nd_term + second_sum_3rd_term

    if scale is not None:
        scaled_control_law = scale_equation_term_list(control_law, scale)
        if trajectory is not None:
            trajectory.scale *= scale
    else:
        scaled_control_law = control_law

    c_name = "parabolic_robin_backstepping_controller"
    if trajectory is not None:
        return sim.SimulationInputSum([trajectory, ct.Controller(ct.ControlLaw(scaled_control_law, name=c_name))])
    else:
        return sim.SimulationInputSum([ct.Controller(ct.ControlLaw(scaled_control_law, name=c_name))])


# TODO: change to factory, rename: function_wrapper
def _convert_to_function(coef):
    if not isinstance(coef, collections.Callable):
        return lambda z: coef
    else:
        return coef


def _convert_to_scalar_function(coef, label):
    from . import core as cr
    if not isinstance(coef, collections.Callable):
        register_base(label, cr.Function(lambda z: coef), overwrite=True)
    elif isinstance(coef, cr.Function):
        register_base(label, coef, overwrite=True)
    else:
        register_base(label, cr.Function(coef), overwrite=True)
    return ph.ScalarFunction(label)


def get_parabolic_dirichlet_weak_form(init_func_label, test_func_label, input, param, spatial_domain):
    from . import simulation as sim
    a2, a1, a0, alpha, beta = param
    l = spatial_domain[1]
    # integral terms
    int1 = ph.IntegralTerm(ph.Product(ph.TemporalDerivedFieldVariable(init_func_label, order=1),
                                      ph.TestFunction(test_func_label, order=0)), spatial_domain)
    int2 = ph.IntegralTerm(
        ph.Product(ph.SpatialDerivedFieldVariable(init_func_label, order=0), ph.TestFunction(test_func_label, order=2)),
        spatial_domain, -a2)
    int3 = ph.IntegralTerm(
        ph.Product(ph.SpatialDerivedFieldVariable(init_func_label, order=0), ph.TestFunction(test_func_label, order=1)),
        spatial_domain, a1)
    int4 = ph.IntegralTerm(
        ph.Product(ph.SpatialDerivedFieldVariable(init_func_label, order=0), ph.TestFunction(test_func_label, order=0)),
        spatial_domain, -a0)
    # scalar terms
    s1 = ph.ScalarTerm(ph.Product(ph.Input(input), ph.TestFunction(test_func_label, order=1, location=l)), a2)
    s2 = ph.ScalarTerm(ph.Product(ph.Input(input), ph.TestFunction(test_func_label, order=0, location=l)), -a1)
    s3 = ph.ScalarTerm(ph.Product(ph.SpatialDerivedFieldVariable(init_func_label, order=1, location=l),
                                  ph.TestFunction(test_func_label, order=0, location=l)), -a2)
    s4 = ph.ScalarTerm(ph.Product(ph.SpatialDerivedFieldVariable(init_func_label, order=1, location=0),
                                  ph.TestFunction(test_func_label, order=0, location=0)), a2)

    # derive state-space system
    return sim.WeakFormulation([int1, int2, int3, int4, s1, s2, s3, s4])


def get_parabolic_robin_weak_form(init_func_label, test_func_label, input, param, spatial_domain,
                                  actuation_type_point=None):
    from . import simulation as sim

    if actuation_type_point is None:
        actuation_type_point = spatial_domain[1]

    a2, a1, a0, alpha, beta = param
    l = spatial_domain[1]
    # init ph.ScalarFunction for a1 and a0, to handle spatially varying coefficients
    # a2 = _convert_to_scalar_function(a2, "a2_z")
    a1_z = _convert_to_scalar_function(a1, "a1_z")
    a0_z = _convert_to_scalar_function(a0, "a0_z")

    # integral terms
    int1 = ph.IntegralTerm(ph.Product(ph.TemporalDerivedFieldVariable(init_func_label, order=1),
                                      ph.TestFunction(test_func_label, order=0)), spatial_domain)
    int2 = ph.IntegralTerm(
        ph.Product(ph.SpatialDerivedFieldVariable(init_func_label, order=1), ph.TestFunction(test_func_label, order=1)),
        spatial_domain, a2)
    int3 = ph.IntegralTerm(ph.Product(ph.Product(ph.SpatialDerivedFieldVariable(init_func_label, order=1), a1_z),
                                      ph.TestFunction(test_func_label, order=0)), spatial_domain, -1)
    int4 = ph.IntegralTerm(ph.Product(ph.Product(ph.SpatialDerivedFieldVariable(init_func_label, order=0), a0_z),
                                      ph.TestFunction(test_func_label, order=0)), spatial_domain, -1)

    # scalar terms
    s1 = ph.ScalarTerm(ph.Product(ph.SpatialDerivedFieldVariable(init_func_label, order=0, location=0),
                                  ph.TestFunction(test_func_label, order=0, location=0)), a2 * alpha)
    s2 = ph.ScalarTerm(ph.Product(ph.SpatialDerivedFieldVariable(init_func_label, order=0, location=l),
                                  ph.TestFunction(test_func_label, order=0, location=l)), a2 * beta)
    s3 = ph.ScalarTerm(
        ph.Product(ph.Input(input), ph.TestFunction(test_func_label, order=0, location=actuation_type_point)), -a2)
    # derive state-space system
    return sim.WeakFormulation([int1, int2, int3, int4, s1, s2, s3])


# TODO: think about interp
def find_nearest_idx(array, value):
    return (np.abs(array - value)).argmin()


def create_dir(dir_name):
    """
    Create a directory with name :py:obj:`dir_name` relative to the current path if it doesn't already exist and
    return its full path.

    Args:
        dir_name (str): Directory name.

    Return:
        str: Full absolute path of the created directory.
    """
    path = os.sep.join([os.getcwd(), dir_name])
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.isdir(path):
        raise FileExistsError("cannot create directory, file of same name already present.")

    return path


def create_animation(input_file_mask="", input_file_names=None, target_format=".mp4"):
    """
    Create an animation from the given files.

    If no file names are given, a file selection dialog will appear.

    Args:
        input_file_mask (basestring): file name mask with c-style format string
        input_file_names (iterable): names of the files

    Return:
        animation file
    """
    # TODO process user input on frame rate file format and so on
    if input_file_mask is not "":
        output_name = "_".join(input_file_mask.split("_")[:-2]) + target_format
        args = ["-i", input_file_mask, "-c:v", "libx264", "-pix_fmt", "yuv420p", output_name]
        call(["ffmpeg"] + args)

        # ffmpeg -i Fri_Jun_24_16:14:50_2016_%04d.png transport_system.gif
        # convert Fri_Jun_24_16:14:50_2016_00*.png out.gif
