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
    sanitizes input data by testing if input is an array of type allowed_type.

    :param input_object:
    :param allowed_type:
    :return:
    """
    input_object = np.atleast_1d(input_object)
    for obj in np.nditer(input_object, flags=["refs_ok"]):
        if not isinstance(np.asscalar(obj), allowed_type):
            raise TypeError("Only objects of type: {0} accepted.".format(allowed_type))

    return input_object


class BaseFraction(object):
    """
    abstract base class representing a basis that can be used to describe functions of several variables
    """
    __metaclass__ = ABCMeta

    def __init__(self, members):
        self.members = members

    def derive(self, order):
        """
        basic implementation of derive function.
        overwrite when subclassing to add more functionality
        :param order: derivative order
        :return: derived object
        """
        if order == 0:
            return self
        else:
            raise ValueError("No derivatives implemented in BaseFraction. Overwrite derive method to implement your "
                             "own!")

    def transformation_hint(self, info):
        """
        method that provides a information about how to transform weights from one BaseFraction into another.

        In Detail this function has to return a callable, which will take the weights of the source and will return the
        weights of the target system. It can have keyword arguments for other data which is required to perform the
        transformation.
        Information about theses extra keyword arguments should be provided in form of a dictionary which is returned
        is keyword arguments are need by the transformation handle.

        The input object will contain the following information:
            - source basis in form of an array of the source Fractions
            - destination basis in form of an array of the destination Fractions
            - available temporal derivative order of source weights
            - needed temporal derivative order for destination weights

        Overwrite this Method in your implementation.
        """
        msg = "This is {0} speaking, \n" \
              "You requested information about how to transform from {1} to {2} \n" \
              "furthermore the source derivative order is {3} and the target one is {4}" \
              "but this is just a dummy method so implement your own hint to make this work!".format(
            self.__class__.__name__, info.src_base[0].__class__.__name__, info.dst_base[0].__class__.__name__,
            info.src_der_order, info.dst_der_order)

        raise NotImplementedError(msg)

    @abstractmethod
    def scalar_product_hint(self):
        """
        hint that returns steps for scalar product calculation.
        In detail this means a list object containing function calls to fill with (first, second) parameters
        that will calculate the scalar product when summed up
        :return:
        """
        pass

    @abstractmethod
    def scale(self, factor):
        """
        factory method to obtain instances of this base fraction, scaled by the given factor.

        :param factor: factor to scale the vector
        """
        pass

    @abstractmethod
    def get_member(self, idx):
        """
        getter function to access members of BaseFraction
        """
        pass


# TODO remove
# class SimpleFunctionVector(BaseFraction):
#     """
#     implementation of the "simple" distributed case, only one member which is a Function
#     """
#
#     def __init__(self, function):
#         if not isinstance(function, Function):
#             raise TypeError("Only Function objects accepted as function")
#         BaseFraction.__init__(self, function)
#
#     def scalar_product_hint(self):
#         return [dot_product_l2]
#
#     @staticmethod
#     def scalar_product(first, second):
#         if not isinstance(first, SimpleFunctionVector) or not isinstance(second, SimpleFunctionVector):
#             raise TypeError("only SimpleFunctionVectors supported")
#         return dot_product_l2(first.members, second.members)
#
#     def get_member(self, idx):
#         if idx != 0:
#             raise ValueError("only one member available!")
#         return self.members
#
#     def scale(self, factor):
#         """
#         easy one, let cr.Function handle the scaling
#         """
#         return SimpleFunctionVector(self.members.scale(factor))


class Function(BaseFraction):
    """
    To ensure the accurateness of numerical handling, areas where it is nonzero have to be provided
    The user can choose between providing a (piecewise) analytical or pure numerical representation of the function
    """

    def __init__(self, eval_handle, domain=(-np.inf, np.inf), nonzero=(-np.inf, np.inf), derivative_handles=[],
                 vectorial=False):
        BaseFraction.__init__(self,  self)
        if not callable(eval_handle):
            raise TypeError("callable has to be provided as function_handle")
        self._function_handle = eval_handle

        if isinstance(eval_handle, Function):
            raise TypeError("Function cannot be initialized with Function!")

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

        self.vectorial = vectorial

    def scalar_product_hint(self):
        """
        this one is easy
        """
        return [dot_product_l2]

    def get_member(self, idx):
        return self

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

        if callable(factor):
            scaled = Function(scale_factory(self._function_handle), domain=self.domain, nonzero=self.nonzero)
        else:
            scaled = Function(scale_factory(self._function_handle), domain=self.domain, nonzero=self.nonzero,
                              derivative_handles=[scale_factory(der_handle) for der_handle in self._derivative_handles])
        return scaled

    def _check_domain(self, value):
        """
        checks if value fits into domain

        :param value: point where function shall be evaluated
        :raises: ValueError if value not in domain
        """
        in_domain = False
        value = np.atleast_1d(value)
        for interval in self.domain:
            if all(value >= interval[0]) and all(value <= interval[1]):
                in_domain = True
                break

        if not in_domain:
            raise ValueError("Function evaluated outside its domain!")

    def __call__(self, argument):
        """
        handle that is used to evaluate the function on a given point

        :param argument: function parameter
        :return: function value
        """
        self._check_domain(argument)
        if self.vectorial:
            return self._function_handle(argument)
        else:
            try:
                ret_val = []
                for arg in argument:
                    ret_val.append(self._function_handle(arg))

                return np.array(ret_val)
            except TypeError, e:
                return self._function_handle(argument)

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


class ComposedFunctionVector(BaseFraction):
    """
    implementation of composite function vector :math:`\\boldsymbol{x}`.


    .. math::
        \\boldsymbol{x} = \\begin{pmatrix}
            x_1(z) \\\\
            \\vdots \\\\
            x_n(z) \\\\
            \\xi_1 \\\\
            \\vdots \\\\
            \\xi_m \\\\
        \\end{pmatrix}

    """

    def __init__(self, functions, scalars):
        funcs = sanitize_input(functions, Function)
        scals = sanitize_input(scalars, Number)

        BaseFraction.__init__(self, {"funcs": funcs, "scalars": scals})

    def scalar_product_hint(self):
        return [dot_product_l2 for funcs in self.members["funcs"]]\
               + [np.multiply for scals in self.members["scalars"]]

    def get_member(self, idx):
        if idx < len(self.members["funcs"]):
            return self.members["funcs"][idx]
        elif idx - len(self.members["funcs"]) < len(self.members["scalars"]):
            return self.members["scalars"][idx - len(self.members["funcs"])]
        else:
            raise ValueError("wrong index")

    def scale(self, factor):
        return self.__class__(np.array([func.scale(factor) for func in self.members["funcs"]]),
                              np.array([scal * factor for scal in self.members["scalars"]]))


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
        raise TypeError("Wrong type(s) supplied. both must be a {0}".format(Function))

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
    # TODO seems like for now vectorize is the better alternative here
    # frst = sanitize_input(first, Function)
    # scnd = sanitize_input(second, Function)
    #
    # res = np.zeros_like(frst)
    #
    # first_iter = frst.flat
    # second_iter = scnd.flat
    # res_iter = res.flat
    #
    # while True:
    #     try:
    #         f = first_iter.next()
    #         s = second_iter.next()
    #         r = res_iter.next()
    #         r[...] = _dot_product_l2(f, s)
    #     except StopIteration:
    #         break
    #
    # return res
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
        return calculate_product_matrix(dot_product_l2, der_a, der_b)
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


def calculate_product_matrix(product_handle, first_member, second_member):
    """
    calculates a matrix :math:`A` whose elements are the scalar products of each element from functions_a and
    functions_b, so that :math:`a_{ij} = \\langle \\mathrm{functions_a}_i\\,,\\: \\mathrm{functions_b}_j\\rangle`.

    :param second_member: (np.ndarray of) :py:class:`Function`
    :param first_member: (np.ndarray of) :py:class:`Function`
    :return: matrix :math:`A` as np.ndarray
    """
    # funcs_a = sanitize_input(second_member, Function)
    # funcs_b = sanitize_input(second_member, Function)

    i, j = np.mgrid[0:first_member.shape[0], 0:second_member.shape[0]]
    funcs_i = first_member[i]
    funcs_j = second_member[j]
    return product_handle(funcs_i, funcs_j)


def project_on_base(function, base):
    """
    projects given function on a basis given by base

    :param function: function the approximate
    :param base: functions or function vectors that generate the base
    :return: weights
    """
    if not isinstance(function, Function):
        raise TypeError("Only pyinduct.Function accepted as 'func'")

    if isinstance(base, Function):  # convenience case
        base = np.asarray([base])

    if not isinstance(base, np.ndarray):
        raise TypeError("Only numpy.ndarray accepted as 'initial_functions'")

    # compute <x(z, t), phi_i(z)>
    projections = np.array([dot_product_l2(function, frac) for frac in base])

    # compute <phi_i(z), phi_j(z)> for 0 < i, j < n
    scale_mat = calculate_product_matrix(dot_product_l2, base, base)

    return np.dot(np.linalg.inv(scale_mat), projections)


def back_project_from_base(weights, base):
    """
    build handle for function that was expressed in a certain basis with weights

    :param weights:
    :param base:
    :return: evaluation handle
    """
    if isinstance(weights, Number):
        weights = np.asarray([weights])
    if isinstance(base, Function):
        base = np.asarray([base])
    if not isinstance(weights, np.ndarray) or not isinstance(base, np.ndarray):
        raise TypeError("Only numpy ndarrays accepted as input")

    if weights.shape[0] != base.shape[0]:
        raise ValueError("Lengths of weights and initial_initial_functions do not match!")

    def eval_handle(z):
        return sum([weights[i] * base[i](z) for i in range(weights.shape[0])])

    # TODO test if bottom one is faster
    return np.vectorize(eval_handle)
    return eval_handle


def change_projection_base(src_weights, src_base, dst_base):
    """
    converts given weights that form an approximation using src_base to the best possible fit using
    dst_base. bases can be given as BaseFraction Array or Function Array.

    :param src_weights: current weights
    :param src_base: original basis (np.ndarray)
    :param dst_base: target basis (np.ndarray)
    :return: target weights
    """
    pro_mat = calculate_base_projection(src_base, dst_base)
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


def calculate_base_projection(src_base, dst_base):
    """
    calculates the base transformation :math:`V` so that the vector of src_weights can be transformed in a vector of
    dst_weights, making the smallest error possible. Quadratic error is used as the error-norm for this case.

    :param dst_base: new projection base
    :param src_base: current projection base
    :return:
    """
    src_base = np.atleast_1d(src_base)
    dst_base = np.atleast_1d(dst_base)

    if not hasattr(src_base[0], "scalar_product_hint"):
        raise TypeError("Input type not supported.")

    # compute P and Q matrices, where P = Sum(P_n) and Q = Sum(Q_n)
    hints = src_base[0].scalar_product_hint()  # assume that all are the same (if not an error will occur anyway)
    p_matrices = []
    q_matrices = []
    for idx, hint in enumerate(hints):
        dst_members = np.array([dst_frac.get_member(idx) for dst_frac in dst_base])
        src_members = np.array([src_frac.get_member(idx) for src_frac in src_base])

        # compute P_n matrix: <phi_tilde_ni(z), phi_dash_nj(z)> for 0 < i < N, 0 < j < M
        p_matrices.append(calculate_product_matrix(hint, dst_members, src_members))

        # compute Q_n matrix: <phi_dash_ni(z), phi_dash_nj(z)> for 0 < i < M, 0 < j < M
        q_matrices.append(calculate_product_matrix(hint, dst_members, dst_members))

    p_mat = np.sum(p_matrices, axis=0)
    q_mat = np.sum(q_matrices, axis=0)

    # compute V matrix: inv(Q)*P
    v_mat = np.dot(np.linalg.inv(q_mat), p_mat)
    return v_mat


def normalize_function(x1, x2=None):
    """
    takes two (vectors of) functions :math:`\\boldsymbol{x}_1` and  :math:`\\boldsymbol{x}_2` and normalizes them so
    that :math:`\\langle\\boldsymbol{x}_1\\,,\:\\boldsymbol{x}_2\\rangle = 1`.
    If only one function is given, :math:`\\boldsymbol{x}_2` is set to :math:`\\boldsymbol{x}_1`.

    :param x1: core.BaseFraction :math:`\\boldsymbol{x}_1`
    :param x2: core.BaseFraction :math:`\\boldsymbol{x}_2`
    :return:
    """
    if x2 is None:
        x2 = x1
    if type(x1) != type(x2):
        raise TypeError("only arguments of same type allowed.")

    if not hasattr(x1, "scalar_product_hint"):
        raise TypeError("Input type not supported.")

    hints = x1.scalar_product_hint()
    res = 0
    for idx, hint in enumerate(hints):
        res += hint(x1.get_member(idx), x2.get_member(idx))

    if abs(res) < np.finfo(float).eps:
        raise ValueError("given base fractions are orthogonal. no normalization possible.")

    scale_factor = np.sqrt(1 / res)
    if x1 == x2:
        return x1.scale(scale_factor)
    else:
        return x1.scale(scale_factor), x2.scale(scale_factor)
