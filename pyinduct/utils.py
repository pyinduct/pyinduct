"""
A few helper functions for users and developer.
"""
import numpy as np
import collections
import copy as cp
import os
import warnings
from numbers import Number
from subprocess import call

from . import core as cr
from . import registry as rg
from . import placeholder as ph
from . import simulation as sim


class Parameters:
    """
    Handy class to collect system parameters.
    This class can be instantiated with a dict, whose keys will the become attributes of the object.
    (Bunch approach)

    Args:
        kwargs: parameters
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def evaluate_placeholder_function(placeholder, input_values):
    """
    Evaluate a given placeholder object, that contains functions.

    Args:
        placeholder: Instance of :py:class:`FieldVariable`, :py:class:`TestFunction` or :py:class:`ScalarFunction`.
        input_values: Values to evaluate at.

    Return:
        :py:obj:`numpy.ndarray` of results.
    """
    if not isinstance(placeholder, (ph.FieldVariable, ph.TestFunction)):
        raise TypeError("Input Object not supported!")

    base = rg.get_base(placeholder.data["func_lbl"]).derive(placeholder.order[1])
    return np.array([func(input_values) for func in base.fractions])


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
                                                d_approx_target_state, integral_kernel_zz, original_beta,
                                                target_beta, trajectory=None, scale=None):
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

    control_law = unsteady_term + first_sum_1st_term + first_sum_2nd_term + \
                  second_sum_1st_term + second_sum_2nd_term + second_sum_3rd_term

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


def convert_to_scalar_function(coefficient, label):
    # TODO move to placeholder.ScalarFunction and add as static method
    """
    create a :py:class:`ScalarFunction` from a coefficient.

    Args:
        coefficient (number, callable or :py:class:`core.Function`): input that is used the generate the placeholder.
            If a number is given, a constant function will be created, if it is callable it will be wrapped in a
            :py:class:`core.Function` and registered.
        label (string): label to register the created base under.

    Returns:
        :py:class:`placeholder.ScalarFunction` : Placeholder object that can be used in a weak formulation.
    """
    if isinstance(coefficient, Number):
        fraction = cr.Function(lambda z: coefficient)
    elif isinstance(coefficient, cr.Function):
        fraction = coefficient
    elif isinstance(coefficient, collections.Callable):
        fraction = cr.Function(coefficient)
    else:
        raise TypeError("Coefficient type not understood.")

    base = cr.Base(fraction)
    rg.register_base(label, base)

    return ph.ScalarFunction(label)


def get_parabolic_dirichlet_weak_form(init_func_label, test_func_label, input_handle, param, spatial_domain):
    a2, a1, a0, alpha, beta = param
    l = spatial_domain[1]

    x = ph.FieldVariable(init_func_label)
    x_dt = x.derive(temp_order=1)
    x_dz = x.derive(spat_order=1)

    psi = ph.TestFunction(test_func_label)
    psi_dz = psi.derive(1)
    psi_ddz = psi.derive(2)

    # integral terms
    int1 = ph.IntegralTerm(ph.Product(x_dt, psi), spatial_domain)
    int2 = ph.IntegralTerm(ph.Product(x, psi_ddz), spatial_domain, -a2)
    int3 = ph.IntegralTerm(ph.Product(x, psi_dz), spatial_domain, a1)
    int4 = ph.IntegralTerm(ph.Product(x, psi), spatial_domain, -a0)

    # scalar terms
    s1 = ph.ScalarTerm(ph.Product(ph.Input(input_handle), psi_dz(l)), a2)
    s2 = ph.ScalarTerm(ph.Product(ph.Input(input_handle), psi(l)), -a1)
    s3 = ph.ScalarTerm(ph.Product(x_dz(l), psi(l)), -a2)
    s4 = ph.ScalarTerm(ph.Product(x_dz(0), psi(0)), a2)

    # derive state-space system
    return sim.WeakFormulation([int1, int2, int3, int4, s1, s2, s3, s4], name="parabolic_dirichlet")


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

    # init ph.ScalarFunction for a1 and a0 to handle spatially varying coefficients
    created_base_labels = ["a0_z", "a1_z"]
    a0_z = convert_to_scalar_function(a0, created_base_labels[0])
    a1_z = convert_to_scalar_function(a1, created_base_labels[1])

    x = ph.FieldVariable(shape_base_label)
    x_dt = x.derive(temp_order=1)
    x_dz = x.derive(spat_order=1)

    psi = ph.TestFunction(test_base_label, order=0)
    psi_dz = psi.derive(1)

    # integral terms
    int1 = ph.IntegralTerm(ph.Product(x_dt, psi), spatial_domain)
    int2 = ph.IntegralTerm(ph.Product(x_dz, psi_dz), spatial_domain, a2)
    int3 = ph.IntegralTerm(ph.Product(ph.Product(x_dz, a1_z), psi), spatial_domain, -1)
    int4 = ph.IntegralTerm(ph.Product(ph.Product(x, a0_z), psi), spatial_domain, -1)

    # scalar terms
    s1 = ph.ScalarTerm(ph.Product(x(0), psi(0)), a2 * alpha)
    s2 = ph.ScalarTerm(ph.Product(x(l), psi(l)), a2 * beta)
    s3 = ph.ScalarTerm(ph.Product(ph.Input(input_handle), psi(actuation_type_point)), -a2)

    # derive state-space system
    weak_form = sim.WeakFormulation([int1, int2, int3, int4, s1, s2, s3], name="parabolic_robin_{}".format(param))
    return weak_form, created_base_labels


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
        args = ["-i", input_file_mask,
                "-c:v", "libx264",
                "-pix_fmt",  "yuv420p",
                output_name]
        call(["ffmpeg"] + args)

    # ffmpeg -i Fri_Jun_24_16:14:50_2016_%04d.png transport_system.gif
    # convert Fri_Jun_24_16:14:50_2016_00*.png out.gif
