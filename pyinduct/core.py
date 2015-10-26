# -*- coding: utf-8 -*-
from __future__ import division
from abc import ABCMeta, abstractmethod
from numbers import Number
import numpy as np
from scipy import integrate
from numpy.polynomial import polynomial as npoly

from pyinduct import get_initial_functions


def sanitize_input(input_object, allowed_type):
    """
    sanitizes input data

    :param input_object:
    :param allowed_type:
    :return:
    """
    # input_object = np.atleast_2d(input_object)
    if isinstance(input_object, allowed_type):
        input_object = np.asarray([input_object])
    elif isinstance(input_object, np.ndarray):
        # test if input is an array of type allowed_type.
        for obj in np.nditer(input_object, flags=["refs_ok"]):
            if not isinstance(np.asscalar(obj), allowed_type):
                raise TypeError("Only objects of type: {0} accepted.".format(allowed_type))
    else:
        raise TypeError("input must be (an numpy.ndarray) of type: {0}".format(allowed_type))

    return input_object


class Function:
    """
    To ensure the accurateness of numerical handling, areas where it is nonzero have to be provided
    The user can choose between providing a (piecewise) analytical or pure numerical representation of the function
    """

    def __init__(self, eval_handle, domain=(-np.inf, np.inf), nonzero=(-np.inf, np.inf), derivative_handles=[]):
        if not callable(eval_handle):
            raise TypeError("callable has to be provided as function_handle")
        self._function_handle = np.vectorize(eval_handle, otypes=[np.float])

        for der_handle in derivative_handles:
            if not callable(der_handle):
                raise TypeError("callable has to be provided as member of derivative_handles")
        self._derivative_handles = derivative_handles

        for kw, val in zip(["domain", "nonzero"], [domain, nonzero]):
            if not isinstance(val, list):
                if isinstance(val, tuple):
                    val = [val]
                else:
                    raise TypeError("List of tuples has to be provided for {0}".format(kw))
            setattr(self, kw, sorted([(min(interval), max(interval)) for interval in val], key=lambda x: x[0]))

    def _check_domain(self, value):
        """
        checks if value fits into domain

        :param value: point where function shall be evaluated
        :raises: ValueError if value not in domain
        """
        in_domain = False
        for interval in self.domain:
            # TODO support multi dimensional access
            if interval[0] <= value <= interval[1]:
                in_domain = True
                break

        if not in_domain:
            raise ValueError("Function evaluated outside its domain!")

    def __call__(self, args):
        """
        handle that is used to evaluate the function on a given point

        :param args: function parameter
        :return: function value
        """
        args = np.atleast_1d(args)
        self._check_domain(args[0])
        ret_val = self._function_handle(args)
        if ret_val.size == 1:
            return ret_val[0]
        else:
            return ret_val

    def derive(self, order=1):
        """
        factory method that is used to evaluate the spatial derivative of this function
        """
        if not isinstance(order, int):
            raise TypeError("only integer allowed as derivation order")
        if order == 0:
            return self
        if order < 0 or order > len(self._derivative_handles):
            raise ValueError("function cannot be differentiated that often.")

        derivative = Function(self._derivative_handles[order - 1], domain=self.domain, nonzero=self.nonzero,
                              derivative_handles=self._derivative_handles[order:])
        return derivative

    def scale(self, factor):
        """
        factory method to scale this function.
        factor can be a number or a function
        """
        # TODO if factor is a function, deny scaling of derivatives and delete them!
        if factor == 1:
            return self

        def scale_factory(func):
            def _scaled_func(z):
                if callable(factor):
                    return factor(z) * func(z)
                else:
                    return factor * func(z)

            return _scaled_func

        scaled = Function(scale_factory(self._function_handle), domain=self.domain, nonzero=self.nonzero,
                          derivative_handles=[scale_factory(der_handle) for der_handle in self._derivative_handles])
        return scaled


class LagrangeFirstOrder(Function):
    """
    Implementation of an lagrangian initial function of order 1
      ^
    1-|         ^
      |        /|\
      |       / | \
      |      /  |  \
    0-|-----/   |   \-------------------------> z
            |   |   |
          start,top,end

    :param start: start node
    :param top: top node, where :math:`f(x) = 1`
    :param start: end node
    """

    def __init__(self, start, top, end):
        if not start <= top <= end or start == end:
            raise ValueError("Input data is nonsense, see Definition.")

        if start == top:
            Function.__init__(self, self._lagrange1st_border_left,
                              nonzero=(start, end), derivative_handles=[self._der_lagrange1st_border_left])
        elif top == end:
            Function.__init__(self, self._lagrange1st_border_right,
                              nonzero=(start, end), derivative_handles=[self._der_lagrange1st_border_right])
        else:
            Function.__init__(self, self._lagrange1st_interior,
                              nonzero=(start, end), derivative_handles=[self._der_lagrange1st_interior])

        self._start = start
        self.top = top
        self._end = end

        # speed
        self._a = self.top - self._start
        self._b = self._end - self.top

    def _lagrange1st_border_left(self, z):
        """
        left border equation for lagrange 1st order
        """
        if z < self.top or z >= self._end:
            return 0
        else:
            return (self.top - z) / self._b + 1

    def _lagrange1st_border_right(self, z):
        """
        right border equation for lagrange 1st order
        """
        if z <= self._start or z > self._end:
            return 0
        else:
            return (z - self._start) / self._a

    def _lagrange1st_interior(self, z):
        """
        interior equations for lagrange 1st order
        """
        if z < self._start or z > self._end:
            return 0
        elif self._start <= z <= self.top:
            return (z - self._start) / self._a
        else:
            return (self.top - z) / self._b + 1

    def _der_lagrange1st_border_left(self, z):
        """
        left border equation for lagrange 1st order
        """
        if z < self.top or z >= self._end:
            return 0
        else:
            return -1 / self._b

    def _der_lagrange1st_border_right(self, z):
        """
        right border equation for lagrange 1st order
        """
        if z <= self._start or z > self._end:
            return 0
        else:
            return 1 / self._a

    def _der_lagrange1st_interior(self, z):
        """
        interior equations for lagrange 1st order
        """
        if z < self._start or z > self._end or z == self.top:
            return 0
        elif self._start <= z < self.top:
            return 1 / self._a
        else:
            return -1 / self._b

            # @staticmethod
            # TODO implement correct one
            # def quad_int():
            #     return 2/3


class LagrangeSecondOrder(Function):
    """
    Implementation of an lagrangian initial function of order 2
      ^                                    _
    1-|           ^                      / | \
      |          /|\                   /   |   \
      |         / | \                 /    |    \
      |        /  |  \               /     |     \
    0-|--\----/   |   \----/--------/------|----- \---> z
          \_/     |    \_/
       start    top       end     start   top    end
         |<----- d ------>|        |<---- d/2 --->|
    :param start: start node
    :param top: top node, where :math:`f(x) = 1`
    :param end: end node
    :param max_element_length: value of the length d (see sketch)
    """

    def __init__(self, start, top, end, max_element_length):
        self._element_length = end-start
        if not start <= top <= end or start == end or (not np.isclose(self._element_length, max_element_length) and not np.isclose(self._element_length, max_element_length/2)):
            raise ValueError("Input data is nonsense, see Definition.")

        self._start = start
        self.top = top
        self._end = end
        self._e_2 = max_element_length/4

        if start == top:
            self._gen_left_top_poly()
            Function.__init__(self, self._lagrange2nd_border_left, nonzero=(start, end),
                              derivative_handles=[self._der_lagrange2nd_border_left,
                                                  self._dder_lagrange2nd_border_left])
        elif top == end:
            self._gen_right_top_poly()
            Function.__init__(self, self._lagrange2nd_border_right, nonzero=(start, end),
                              derivative_handles=[self._der_lagrange2nd_border_right,
                                                  self._dder_lagrange2nd_border_right])
        elif np.isclose(end-start, max_element_length):
            self._gen_left_top_poly()
            self._gen_right_top_poly()
            Function.__init__(self, self._lagrange2nd_interior, nonzero=(start, end),
                              derivative_handles=[self._der_lagrange2nd_interior,
                                                  self._dder_lagrange2nd_interior])
        elif np.isclose(end-start, max_element_length/2):
            self._gen_mid_top_poly()
            Function.__init__(self, self._lagrange2nd_interior_half, nonzero=(start, end),
                              derivative_handles=[self._der_lagrange2nd_interior_half,
                                                  self._dder_lagrange2nd_interior_half])
        else:
            raise ValueError("Following arguments do not meet the specs from LagrangeSecondOrder: start, end")

    def _gen_left_top_poly(self):
        left_top_poly = npoly.Polynomial(npoly.polyfromroots((self._e_2, 2*self._e_2)))
        self._left_top_poly = npoly.Polynomial(left_top_poly.coef/left_top_poly(0))

    def _gen_right_top_poly(self):
        right_top_poly = npoly.Polynomial(npoly.polyfromroots((0, self._e_2)))
        self._right_top_poly = npoly.Polynomial(right_top_poly.coef/right_top_poly(2*self._e_2))

    def _gen_mid_top_poly(self):
        mid_top_poly = npoly.Polynomial(npoly.polyfromroots((0, 2*self._e_2)))
        self._mid_top_poly = npoly.Polynomial(mid_top_poly.coef/mid_top_poly(self._e_2))

    def _lagrange2nd_border_left(self, z, der_order=0):
        """
        left border equation for lagrange 2nd order
        """
        if z < self.top or z > self._end:
            return 0
        else:
            return self._left_top_poly.deriv(der_order)(z)

    def _lagrange2nd_border_right(self, z, der_order=0):
        """
        right border equation for lagrange 2nd order
        """
        if z < self._start or z > self._end:
            return 0
        else:
            return self._right_top_poly.deriv(der_order)(z-self._start)

    def _lagrange2nd_interior(self, z, der_order=0):
        """
        wide (d) interior equations for lagrange 2nd order
        """
        if z < self._start or z > self._end:
            return 0
        elif z == self.top and der_order > 0:
            return 0
        elif self._start <= z <= self.top:
            return self._right_top_poly.deriv(der_order)(z-self._start)
        else:
            return self._left_top_poly.deriv(der_order)(z-self.top)

    def _lagrange2nd_interior_half(self, z, der_order=0):
        """
        small (d/2) interior equations for lagrange 2nd order
        """
        if z < self._start or z > self._end:
            return 0
        else:
            return self._mid_top_poly.deriv(der_order)(z-self._start)

    def _der_lagrange2nd_border_left(self, z):
        return self._lagrange2nd_border_left(z, der_order=1)

    def _der_lagrange2nd_border_right(self, z):
        return self._lagrange2nd_border_right(z, der_order=1)

    def _der_lagrange2nd_interior(self, z):
        return self._lagrange2nd_interior(z, der_order=1)

    def _der_lagrange2nd_interior_half(self, z):
        return self._lagrange2nd_interior_half(z, der_order=1)

    def _dder_lagrange2nd_border_left(self, z):
        return self._lagrange2nd_border_left(z, der_order=2)

    def _dder_lagrange2nd_border_right(self, z):
        return self._lagrange2nd_border_right(z, der_order=2)

    def _dder_lagrange2nd_interior(self, z):
        return self._lagrange2nd_interior(z, der_order=2)

    def _dder_lagrange2nd_interior_half(self, z):
        return self._lagrange2nd_interior_half(z, der_order=2)


class FunctionVector:
    """
    class that implements vectors of function and scalars to cope with situations where distributed as well as
    concentrated elements have to be provided
    """
    __metaclass__ = ABCMeta

    def __init__(self, members):
        self.members = members

    @abstractmethod
    def scalar_product(first, second):
        """
        define how the scalar product is defined between certain FunctionVectors.
        Implementations must be static
        """
        pass

    @abstractmethod
    def scale(self, factor):
        """
        factory method to obtain instances of this vector scaled by the given factor.

        :param factor: factor to scale the vector
        """
        pass


class SimpleFunctionVector(FunctionVector):
    """
    implementation of the "simple" distributed case, only one member which is a Function
    """

    def __init__(self, function):
        if not isinstance(function, Function):
            raise TypeError("Only Function objects accepted as function")
        FunctionVector.__init__(self, function)

    @staticmethod
    def scalar_product(first, second):
        if not isinstance(first, SimpleFunctionVector) or not isinstance(second, SimpleFunctionVector):
            raise TypeError("only SimpleFunctionVectors supported")
        return dot_product_l2(first.members, second.members)

    def scale(self, factor):
        return SimpleFunctionVector(self.members.scale(factor))


class ComposedFunctionVector(FunctionVector):
    """
    implementation of composite function vector :math:`\\boldsymbol{x}`.
    It contains one Function member :math:`x(t)` and one scalar member :math:`\\xi.`

    .. math::
        \\boldsymbol{x} = \\begin{pmatrix}
            x(z) \\\\
            \\xi
        \\end{pmatrix}
    """

    def __init__(self, function, scalar):
        if not isinstance(function, Function):
            raise TypeError("Only Function objects accepted as function")
        if not isinstance(scalar, (int, long, float)):
            raise TypeError("Only int or float objects accepted as scalar")

        FunctionVector.__init__(self, [function, scalar])

    @staticmethod
    def scalar_product(first, second):
        """
        special way the scalar product of this composite vector is calculated
        """
        if not isinstance(first, ComposedFunctionVector) or not isinstance(second, ComposedFunctionVector):
            raise TypeError("only ComposedFunctionVector supported")
        return dot_product_l2(first.members[0], second.members[0]) + first.members[1] * second.members[1]

    def scale(self, factor):
        return ComposedFunctionVector(self.members[0].scale(factor),
                                      self.members[1] * factor)


def domain_intersection(first, second):
    """
    calculate intersection two domains

    :param first: domain
    :param second: domain
    :return: intersection
    """
    if isinstance(first, tuple):
        first = [first]
    if isinstance(second, tuple):
        second = [second]

    intersection = []
    first_idx = 0
    second_idx = 0
    last_first_idx = 0
    last_second_idx = 0
    last_first_upper = None
    last_second_upper = None

    while first_idx < len(first) and second_idx < len(second):
        # TODO remove interval and boundary checking? should be done before
        if last_first_upper is not None and first_idx is not last_first_idx:
            if last_first_upper >= first[first_idx][0]:
                raise ValueError("Intervals not ordered!")
        if last_second_upper is not None and second_idx is not last_second_idx:
            if last_second_upper >= second[second_idx][0]:
                raise ValueError("Intervals not ordered!")

        if first[first_idx][0] > first[first_idx][1]:
            raise ValueError("Interval boundaries given in wrong order")
        if second[second_idx][0] > second[second_idx][1]:
            raise ValueError("Interval boundaries given in wrong order")

        # backup for interval order check
        last_first_idx = first_idx
        last_second_idx = second_idx
        last_first_upper = first[first_idx][1]
        last_second_upper = second[second_idx][1]

        # no common domain -> search
        if second[second_idx][0] <= first[first_idx][0] <= second[second_idx][1]:
            # common start found in 1st domain
            start = first[first_idx][0]
        elif first[first_idx][0] <= second[second_idx][0] <= first[first_idx][1]:
            # common start found in 2nd domain
            start = second[second_idx][0]
        else:
            # intervals have no intersection
            first_idx += 1
            continue

        # add end
        if first[first_idx][1] <= second[second_idx][1]:
            end = first[first_idx][1]
            first_idx += 1
        else:
            end = second[second_idx][1]
            second_idx += 1

        # complete domain found
        if not np.isclose(start, end):
            intersection.append((start, end))

    return intersection


def _dot_product_l2(first, second):
    """
    calculates the inner product of two functions

    :param first: function
    :param second: function
    :return: inner product
    """
    if not isinstance(first, Function) or not isinstance(second, Function):
        raise TypeError("Wrong type(s) supplied both must be a {0}".format(Function))

    # TODO remember not only 1d possible here!
    limits = domain_intersection(first.domain, second.domain)
    nonzero = domain_intersection(first.nonzero, second.nonzero)
    areas = domain_intersection(limits, nonzero)

    # try some shortcuts
    if first == second:
        if hasattr(first, "quad_int"):
            return first.quad_int()

    if 0:
        # TODO let Function Class handle product
        if type(first) is type(second):
            pass

    # standard case
    func = lambda z: first(z) * second(z)
    result, error = integrate_function(func, areas)

    return result


def integrate_function(function, interval):
    """
    integrates the given function over given interval

    :param function:
    :param interval:
    :return:
    """
    result = 0
    err = 0
    for area in interval:
        res = integrate.quad(function, area[0], area[1])
        result += res[0]
        err += res[1]

    return result, err


def dot_product(first, second):
    """
    calculates the inner product of the scalars

    :param first:
    :param second:
    :return:
    """
    return np.inner(first, second)


def dot_product_l2(first, second):
    """
    vectorized version of dot_product

    :param first: numpy.ndarray of function
    :param second: numpy.ndarray of function
    :return: numpy.nadarray of inner product
    """
    if "handle" not in dot_product_l2.__dict__:
        dot_product_l2.handle = np.vectorize(_dot_product_l2)
    return dot_product_l2.handle(first, second)


def calculate_function_matrix_differential(functions_a, functions_b,
                                           derivative_order_a, derivative_order_b, locations=None):
    """
    see calculate function matrix, except for the circumstance that derivatives of given order will be used and the
    derivatives can be evaluated at location before calculation. (saves integral computation)

    :param functions_a:
    :param functions_b:
    :param derivative_order_a:
    :param derivative_order_b:
    :param locations: points to evaluate
    :return:
    """
    der_a = np.asarray([func.derive(derivative_order_a) for func in functions_a])
    der_b = np.asarray([func.derive(derivative_order_b) for func in functions_b])
    if locations is None:
        return calculate_function_matrix(der_a, der_b)
    else:
        if not isinstance(locations, tuple) or len(locations) != 2:
            raise TypeError("only tuples of len 2 allowed for locations.")

        vals_a = np.atleast_1d([der(locations[0]) for der in der_a])
        vals_b = np.asarray([der(locations[1]) for der in der_b])
        return calculate_scalar_matrix(vals_a, vals_b.T)


def calculate_scalar_matrix(values_a, values_b):
    """
    helper function to calculate a matrix of scalars

    :param values_a:
    :param values_b:
    :return:
    """
    i, j = np.mgrid[0:values_a.shape[0], 0:values_b.shape[0]]
    vals_i = values_a[i]
    vals_j = values_b[j]
    return np.multiply(vals_i, vals_j)


def calculate_function_matrix(functions_a, functions_b):
    """
    calculates a matrix :math:`A` whose elements are the scalar products of each element from functions_a and
    functions_b, so that :math:`a_{ij} = \\langle \\mathrm{functions_a}_i\\,,\\: \\mathrm{functions_b}_j\\rangle`.

    :param functions_a: (np.ndarray of) :py:class:`Function`
    :param functions_b: (np.ndarray of) :py:class:`Function`
    :return: matrix :math:`A` as np.ndarray
    """
    funcs_a = sanitize_input(functions_a, Function)
    funcs_b = sanitize_input(functions_b, Function)

    i, j = np.mgrid[0:funcs_a.shape[0], 0:funcs_b.shape[0]]
    funcs_i = funcs_a[i]
    funcs_j = funcs_b[j]
    return dot_product_l2(funcs_i, funcs_j)


def project_on_initial_functions(function, initial_functions):
    """
    projects given function on a new basis

    :param function: function the approximate
    :param initial_functions: initial functions
    :return: weights
    """
    if not isinstance(function, Function):
        raise TypeError("Only pyinduct.Function accepted as 'func'")

    if isinstance(initial_functions, Function):  # convenience case
        initial_functions = np.asarray([initial_functions])

    if not isinstance(initial_functions, np.ndarray):
        raise TypeError("Only numpy.ndarray accepted as 'initial_functions'")

    # compute <x(z, t), phi_i(z)>
    projections = dot_product_l2(function, initial_functions)

    # compute <phi_i(z), phi_j(z)> for 0 < i, j < n
    scale_mat = calculate_function_matrix(initial_functions, initial_functions)

    return np.dot(np.linalg.inv(scale_mat), projections)


def back_project_from_initial_functions(weights, initial_functions):
    """
    build handle for function that was expressed in test functions with weights

    :param weights:
    :param initial_functions:
    :return: evaluation handle
    """
    if isinstance(weights, Number):
        weights = np.asarray([weights])
    if isinstance(initial_functions, Function):
        initial_functions = np.asarray([initial_functions])
    if not isinstance(weights, np.ndarray) or not isinstance(initial_functions, np.ndarray):
        raise TypeError("Only numpy ndarrays accepted as input")

    if weights.shape[0] != initial_functions.shape[0]:
        raise ValueError("Lengths of weights and initial_initial_functions do not match!")

    def eval_handle(z):
        return sum([weights[i] * initial_functions[i](z) for i in range(weights.shape[0])])

    # TODO test if bottom one is faster
    return np.vectorize(eval_handle)
    return eval_handle


def change_projection_base(src_weights, src_initial_funcs, dst_initial_funcs):
    """
    converts given weights that form an approximation using src_test_functions to the best possible fit using
    dst_test_functions.

    :param src_weights: current weights
    :param src_initial_funcs: original test functions (np.ndarray)
    :param dst_initial_funcs: target test functions (np.ndarray)
    :return: target weights
    """
    pro_mat = calculate_base_projection(src_initial_funcs, dst_initial_funcs)
    return project_weights(pro_mat, src_weights)


def project_weights(projection_matrix, src_weights):
    """
    project weights on new basis using the provided projection

    :param src_weights: src_weights
    :param projection_matrix: projection
    :return:
    """
    if isinstance(src_weights, Number):
        src_weights = np.asarray([src_weights])

    return np.dot(projection_matrix, src_weights)


def calculate_base_projection(src_initial_funcs, dst_initial_funcs):
    """
    calculates the base transformation :math:`V` so that the vector of src_weights can be transformed in a vector of
    dst_weights, making the smallest error possible. Quadratic error is used as the error-norm for this case.

    :param dst_initial_funcs: new projection base
    :param src_initial_funcs: current projection base
    :return:
    """
    if isinstance(src_initial_funcs, Function):
        src_initial_funcs = np.asarray([src_initial_funcs])
    if isinstance(dst_initial_funcs, Function):
        dst_initial_funcs = np.asarray([dst_initial_funcs])

    # compute T matrix: <phi_tilde_i(z), phi_dash_j(z)> for 0 < i < n, 0 < j < m
    t_mat = calculate_function_matrix(dst_initial_funcs, src_initial_funcs)
    # compute R matrix: <phi_dash_i(z), phi_dash_j(z)> for 0 < i, j < m
    r_mat = calculate_function_matrix(dst_initial_funcs, dst_initial_funcs)
    # compute V matrix: T*inv(R)
    v_mat = np.dot(np.linalg.inv(r_mat), t_mat)

    return v_mat


def normalize_function(x1, x2=None):
    """
    takes two (vectors of) functions :math:`\\boldsymbol{x}_1` and  :math:`\\boldsymbol{x}_2` and normalizes them so
    that :math:`\\langle\\boldsymbol{x}_1\\,,\:\\boldsymbol{x}_2\\rangle = 1`.
    If only one function is given, :math:`\\boldsymbol{x}_2` is set to :math:`\\boldsymbol{x}_1`.

    :param x1: core.Function or core.FunctionVector :math:`\\boldsymbol{x}_1`
    :param x2: core.Function or core.FunctionVector :math:`\\boldsymbol{x}_2`
    :return:
    """
    if not isinstance(x1, FunctionVector) and not isinstance(x1, Function):
        raise TypeError("only core.Function and core.FunctionVector supported.")

    if x2 is None:
        x2 = x1
    if type(x1) != type(x2):
        raise TypeError("only arguments of same type allowed.")

    if isinstance(x1, Function):
        product = dot_product_l2(x1, x2)
    else:
        product = x1.scalar_product(x1, x2)

    if abs(product) < np.finfo(float).eps:
        raise ValueError("given function are orthogonal. no normalization possible.")

    scale_factor = np.sqrt(1 / product)
    if x1 == x2:
        return x1.scale(scale_factor)
    else:
        return x1.scale(scale_factor), x2.scale(scale_factor)
