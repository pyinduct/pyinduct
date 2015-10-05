from __future__ import division
import numpy as np
import scipy.integrate as si
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from pyinduct import get_initial_functions
from core import Function, LagrangeFirstOrder, back_project_from_initial_functions
from placeholder import FieldVariable, TestFunction
from visualization import EvalData
import warnings
import pyqtgraph as pg

warnings.simplefilter('always', UserWarning)

__author__ = 'stefan'


def cure_interval(test_function_class, interval, node_count=None, element_length=None):
    """
    Uses given test functions to cure a given interval with either node_count nodes or with
    elements of element_length
    :param interval:
    :param test_function_class:
    :return:
    """
    if not issubclass(test_function_class, Function):
        raise TypeError("test_function_class must be a SubClass of Function.")
    # TODO implement more
    if test_function_class is not LagrangeFirstOrder:
        raise TypeError("only LagrangeFirstOrder supported as test_function_class for now.")

    if not isinstance(interval, tuple):
        raise TypeError("interval must be given as tuple.")
    if len(interval) is not 2:
        raise TypeError("interval type not supported, should be (start, end)")

    if node_count and element_length:
        raise ValueError("node_count and element_length provided. Only one can be choosen.")
    if not node_count and not element_length:
        raise ValueError("neither (sensible) node_count nor element_length provided.")

    start = min(interval)
    end = max(interval)

    if node_count:
        nodes, element_length = np.linspace(start=start, stop=end, num=node_count, retstep=True)
    else:
        nodes = np.arange(start, end + element_length, element_length)
        node_count = nodes.shape[0]

    test_functions = [LagrangeFirstOrder(nodes[0], nodes[0], nodes[0] + element_length),
                      LagrangeFirstOrder(nodes[-1] - element_length, nodes[-1], nodes[-1])]
    for i in range(1, node_count - 1):
        test_functions.insert(-1, LagrangeFirstOrder(nodes[i] - element_length,
                                                     nodes[i],
                                                     nodes[i] + element_length))

    return nodes, np.asarray(test_functions)


def find_roots(function, n_roots, area_end, rtol, points_per_root=10, atol=1e-7, show_plot=False):
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
    integer = [n_roots, points_per_root, rtol]
    if not callable(function):
        raise TypeError("callable handle is needed")
    if not all([isinstance(num, int) for num in integer]):
        raise TypeError("n_roots, points_per_root and rtol must be of type int")
    if any([num <= 0 for num in positive_numbers]):
        raise ValueError("n_roots, points_per_root, area_end and atol must be positive")
    if not isinstance(show_plot, bool):
        raise TypeError("show_plot must be of type bool")

    # increase n_roots and area_end
    # TODO maybe the scaling stuff should be completely removed. basically it is undocumented behavior.
    safety_factor = 2
    own_n_roots = safety_factor * n_roots
    own_area_end = safety_factor * area_end
    values = np.linspace(0, own_area_end, own_n_roots * points_per_root)

    roots = []
    rounded_roots = []
    errors = []

    val = iter(values)
    while len(roots) < own_n_roots:
        try:
            root, info, ier, msg = fsolve(function, val.next(), full_output=True)
        except StopIteration:
            break

        if info['fvec'] > atol:
            continue
        if root < 0:
            continue

        rounded_root = np.round(root, -rtol)
        if rounded_root in rounded_roots:
            idx = rounded_roots.index(rounded_root)
            if errors[idx] > info['fvec']:
                roots[idx] = root
                errors[idx] = info['fvec']
            continue

        roots.append(root)
        rounded_roots.append(rounded_root)
        errors.append(info['fvec'])

    if len(roots) < n_roots:
        raise ValueError("Insufficient number of roots {0} detected. "
                         "Try to increase the area to search in.".format(len(roots)))

    found_roots = np.atleast_1d(sorted(roots)[:n_roots]).flatten()

    if show_plot:
        points = np.arange(0, area_end, .1)
        values = function(points)
        pw = pg.plot(title="function + roots")
        pw.plot(points, values)
        pw.plot(found_roots, function(found_roots), pen=None, symbolPen=pg.mkPen("g"))
        pg.QtGui.QApplication.instance().exec_()

    return found_roots


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


def evaluate_approximation(weights, function_label, temporal_steps, spatial_interval, spatial_step):
    """
    evaluate an approximation given by weights and functions at the points given in spatial and temporal steps

    :param weights: 2d np.ndarray where axis 1 is the weight index and axis 0 the temporal index
    :param function_label: functions to use for back-projection
    :return:
    """
    funcs = get_initial_functions(function_label, 0)
    if weights.shape[1] != funcs.shape[0]:
        raise ValueError("weights have to fit provided functions!")

    spatial_steps = np.arange(spatial_interval[0], spatial_interval[1] + spatial_step, spatial_step)

    def eval_spatially(weight_vector):
        if isinstance(function_label[0], LagrangeFirstOrder):
            # shortcut for fem approximations
            nodes = [func.top for func in function_label]
            handle = interp1d(nodes, weight_vector)
        else:
            handle = back_project_from_initial_functions(weight_vector, function_label)
        return handle(spatial_steps)

    data = np.apply_along_axis(eval_spatially, 1, weights)
    return EvalData([temporal_steps, spatial_steps], data)


def transform_eigenfunction(target_eigenvalue, init_eigenfunction, dgl_coefficients, domain):
    """
    Provide the eigenfunction y to an eigenvalue-problem of the form
    a2(z)y''(z) + a1(z)y'(z) + a0(z)y(z) = w y(z)
    where w is a predefined (potentially complex) eigenvalue and z0 <= z <= z1 is the domain.
    :param target_eigenvalue: w (float)
    :param init_eigenfunction: y(0) = [Re{y(0)}, Re{y'(0)}, Im{y(0)}, Im{y'(0)}] (list of floats)
    :param dgl_coefficients: [a0(z), a1(z), a2(z)] (list of function handles)
    :param domain: [z0, ..... , z1] (list of floats)
    :return: y(z) = [Re{y(z)}, Re{y'(z)}, Im{y(z)}, Im{y'(z)}] (list of numpy arrays)
    """

    def ff(y, z):
        a2, a1, a0 = dgl_coefficients
        wr = target_eigenvalue.real
        wi = target_eigenvalue.imag
        d_y = np.array([y[1],
                        -(a0(z) - wr) / a2(z) * y[0] - a1(z) / a2(z) * y[1] - wi / a2(z) * y[2],
                        y[3],
                        wi / a2(z) * y[0] - (a0(z) - wr) / a2(z) * y[2] - a1(z) / a2(z) * y[3]
                        ])
        return d_y

    eigenfunction = si.odeint(ff, init_eigenfunction, domain)

    return [eigenfunction[:, 0],
            eigenfunction[:, 1],
            eigenfunction[:, 2],
            eigenfunction[:, 3]]


def get_transformed_parameter_without_advection(param):
    """
    Transformation which eliminate the advection term 'a1 x(z,t)' from the
    reaction-advection-diffusion equation
    d/dt x(z,t) = a2 x''(z,t) + a1 x'(z,t) + a0 x(z,t)
    with robin boundary condition
    x'(0,t) = alpha x(0,t), x'(l,t) = -beta x(l,t)
    """
    a2, a1, a0, alpha, beta = param

    a1_n = 0.
    a0_n = a0 - a1 ** 2. / 4. / a2
    if alpha is None:
        alpha_n = None
    else:
        alpha_n = a1 / 2. / a2 + alpha
    if beta is None:
        beta_n = None
    else:
        beta_n = -a1 / 2. / a2 + beta

    return a2, a1_n, a0_n, alpha_n, beta_n


def get_adjoint_rad_robin_evp_param(param):
    """
    Return to the eigen value problem of the reaction-advection-diffusion
    equation with robin boundary conditions
    a2 y''(z) + a1 y'(z) + a0 y(z) = w y(z)
    y'(0) = alpha y(0), y'(l) = -beta y(l)
    the parameters for the adjoint Problem (with the same structure).
    """
    a2, a1, a0, alpha, beta = param

    alpha_n = a1 / a2 + alpha
    beta_n = -a1 / a2 + beta
    a1_n = -a1

    return a2, a1_n, a0, alpha_n, beta_n


def get_adjoint_rad_dirichlet_evp_param(param):
    """
    Return to the eigen value problem of the reaction-advection-diffusion
    equation with dirichlet boundary conditions
    a2 y''(z) + a1 y'(z) + a0 y(z) = w y(z)
    y(0) = 0, y(l) = 0
    the parameters for the adjoint Problem (with the same structure).
    """
    a2, a1, a0, _, _ = param

    a1_n = -a1

    return a2, a1_n, a0, None, None


def split_domain(n, a_desired, l, mode='one_even_one_odd'):
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

    if not isinstance(n, (long, int)):
        raise TypeError("Integer excepted.")
    if n % 2 == 0:
        raise ValueError("n must be odd.")
    if l <= 0:
        raise ValueError("l can not be <= 0")
    if not 0. < a_desired < l:
        raise ValueError("a_desired not in interval (0,l).")

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


def get_inn_domain_transformation_matrix(n, k1, k2, mode='n_plus_1'):
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
    if not all(isinstance(i, (int, long)) for i in [n, k1, k2]):
        raise TypeError("TypeErrorMessage")
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
    return M


class ReaAdvDifRobinEigenvalues(object):
    """ temporary """

    def __init__(self, param, l, n_roots=10):
        self.param = param
        self.l = l
        self.n_roots = n_roots

        a2, a1, a0, alpha, beta = self.param
        # real part of the roots from the characteristic equation (eigen value problem dgl)
        self.eta = -a1 / 2. / a2
        # squared imaginary part of the roots from the characteristic equation (eigen value problem dgl)
        self.om_squared = self.compute_squared_eigen_frequencies(self.param, self.eta, self.l, self.n_roots)
        self.eig_values = a0 - a2 * self.om_squared - a1 ** 2 / 4. / a2

    def compute_squared_eigen_frequencies(self, param, eta, l, n_roots):
        a2, a1, a0, alpha, beta = param

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
        om = find_roots(characteristic_equation, 2 * n_roots, om_end, rtol=-1).tolist()

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

        if len(om_squared) < n_roots:
            raise ValueError("ReaAdvDifRobinEigenvalues.compute_squared_eigen_frequencies()"
                             "can not find enough roots")

        return np.array(om_squared[:n_roots])


class ReaAdvDifDirichletEigenvalues(object):
    """ temporary """

    def __init__(self, param, l, n_roots=10):
        self._param = param
        self._l = l
        self._n_roots = n_roots

        a2, a1, a0, _, _ = self._param
        # squared imaginary part of the roots from the characteristic equation (eigen value problem dgl)
        self.om_squared = self._compute_eigen_frequencies(self._l, self._n_roots)
        self.eig_values = a0 - a2 * self.om_squared - a1 ** 2 / 4. / a2
        print 'eigen_val: '
        print self.eig_values
        print 'om: '
        print np.sqrt(self.om_squared)
        print 'eta: '
        print -a1 / 2. / a2

    def _compute_eigen_frequencies(self, l, n_roots):
        om_squared = np.array([(i * np.pi / l) ** 2 for i in xrange(1, n_roots + 1)])

        return om_squared


class ReaAdvDifRobinEigenfunction(Function):
    def __init__(self, om_squared, param, spatial_domain, norm_fak=1.):
        self._om_squared = om_squared
        self._param = param
        self.norm_fak = norm_fak
        Function.__init__(self, self._phi, nonzero=spatial_domain, derivative_handles=[self._d_phi])

    def _phi(self, z):
        a2, a1, a0, alpha, beta = self._param
        eta = -a1 / 2. / a2

        if self._om_squared >= 0.:
            om = np.sqrt(self._om_squared)
            cosX_term = np.cos(om * z)
            if round(om, 200) != 0.:
                sinX_term = (alpha - eta) / om * np.sin(om * z)
            else:
                sinX_term = (alpha - eta) * z
        else:
            om = np.sqrt(-self._om_squared)
            cosX_term = np.cosh(om * z)
            if round(om, 200) != 0.:
                sinX_term = (alpha - eta) / om * np.sinh(om * z)
            else:
                sinX_term = (alpha - eta) * z

        phi_i = np.exp(eta * z) * (cosX_term + sinX_term)

        return phi_i * self.norm_fak

    def _d_phi(self, z):
        a2, a1, a0, alpha, beta = self._param
        eta = -a1 / 2. / a2

        if self._om_squared >= 0.:
            om = np.sqrt(self._om_squared)
            cosX_term = alpha * np.cos(om * z)
            if round(om, 200) != 0.:
                sinX_term = (eta * (alpha - eta) / om - om) * np.sin(om * z)
            else:
                sinX_term = eta * (alpha - eta) * z - om * np.sin(om * z)
        else:
            om = np.sqrt(-self._om_squared)
            cosX_term = alpha * np.cosh(om * z)
            if round(om, 200) != 0.:
                sinX_term = (eta * (alpha - eta) / om + om) * np.sinh(om * z)
            else:
                sinX_term = eta * (alpha - eta) * z + om * np.sinh(om * z)

        d_phi_i = np.exp(eta * z) * (cosX_term + sinX_term)

        return d_phi_i * self.norm_fak


class ReaAdvDifDirichletEigenfunction(Function):
    def __init__(self, omega, param, spatial_domain, norm_fak=1.):
        self._omega = omega
        self._param = param
        self.norm_fak = norm_fak

        a2, a1, a0, _, _ = self._param
        self._eta = -a1 / 2. / a2
        Function.__init__(self, self._phi, nonzero=spatial_domain, derivative_handles=[self._d_phi, self._dd_phi])

    def _phi(self, z):
        eta = self._eta
        om = self._omega

        phi_i = np.exp(eta * z) * np.sin(om * z)

        return return_real_part(phi_i * self.norm_fak)

    def _d_phi(self, z):
        eta = self._eta
        om = self._omega

        d_phi_i = np.exp(eta * z) * (om * np.cos(om * z) + eta * np.sin(om * z))

        return return_real_part(d_phi_i * self.norm_fak)

    def _dd_phi(self, z):
        eta = self._eta
        om = self._omega

        d_phi_i = np.exp(eta * z) * (om * (eta + 1) * np.cos(om * z) + (eta - om ** 2) * np.sin(om * z))

        return return_real_part(d_phi_i * self.norm_fak)


def return_real_part(to_return):
    """
    Check if the imaginary part of to_return vanishes
    and return the real part
    :param to_return:
    :return:
    """
    if isinstance(to_return, (list, np.ndarray)):
        if not all([c.imag == 0. for c in to_return]):
            raise ValueError("Something goes wrong, imaginary part does not vanish")
        return np.asarray([c.real for c in to_return])
    elif isinstance(to_return, (float, int, long)):
        if not to_return.imag == 0.:
            raise ValueError("Something goes wrong, imaginary part does not vanish")
        return to_return.real


def normalize(phi, psi, l):
    """ temporary """
    z_normalize = np.linspace(0., l, 1e5)

    integrand = phi(z_normalize) * psi(z_normalize)
    integral = si.simps(integrand, z_normalize)
    scale = 1. / np.sqrt(integral)

    return scale


if __name__ == '__main__':
    def ax(z):
        return np.sin(np.pi * z) + 1


    Phi = transform_eigenfunction(1.5,
                                  [1., 1., 1., 1.],
                                  [ax, ax, ax],
                                  np.linspace(0, 1, 1e1),
                                  )

    print Phi
