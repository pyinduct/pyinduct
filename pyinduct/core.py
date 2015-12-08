# -*- coding: utf-8 -*-
from __future__ import division
from abc import ABCMeta, abstractmethod
from copy import copy
from numbers import Number
import numpy as np
from scipy import integrate
from scipy.linalg import block_diag
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
    # TODO implement more verbose content in all abstarct functions and then kick out the abstract stuff
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

    def transformation_hint(self, info, target):
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

        Overwrite this Method in your implementation to support conversion between bases that differ from yours.

        This implementation will cover the most basic case, where to two baseFractions are of same type. For any other
        case it will raise an exception.
        """
        # TODO handle target option!
        if target is False:
            raise NotImplementedError

        cls = info.src_base[0].__class__ if target else info.dst_base[0].__class__
        if cls == self.__class__:
            return self._transformation_factory(info), None
        else:
            # No Idea what to do.
            msg = "This is {1} speaking, \n" \
                  "You requested information about how to transform to '{0}'({1}) from '{2}'({3}), \n" \
                  "furthermore the source derivative order is {4} and the target one is {4}. \n" \
                  "But this is a dumb method so implement your own hint to make things work!".format(
                info.dst_lbl, self.__class__.__name__, info.src_lbl, info.src_base[0].__class__.__name__,
                info.dst_base[0].__class__.__name__, info.src_order, info.dst_order)
            raise NotImplementedError(msg)

    @staticmethod
    def _transformation_factory(info):
        mat = calculate_expanded_base_transformation_matrix(info.src_base, info.dst_base, info.src_order,
                                                            info.dst_order)

        def handle(weights):
            return np.dot(mat, weights)
        return handle

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


class Function(BaseFraction):
    """
    To ensure the accurateness of numerical handling, areas where it is nonzero have to be provided
    The user can choose between providing a (piecewise) analytical or pure numerical representation of the function
    """
    # TODO: overload add and mul operators

    def __init__(self, eval_handle, domain=(-np.inf, np.inf), nonzero=(-np.inf, np.inf), derivative_handles=[],
                 vectorial=False):
        BaseFraction.__init__(self,  self)

        # domain and nonzero area
        for kw, val in zip(["domain", "nonzero"], [domain, nonzero]):
            if not isinstance(val, list):
                if isinstance(val, tuple):
                    val = [val]
                else:
                    raise TypeError("List of tuples has to be provided for {0}".format(kw))
            setattr(self, kw, sorted([(min(interval), max(interval)) for interval in val], key=lambda x: x[0]))

        # handle must be callable
        if not callable(eval_handle):
            raise TypeError("callable has to be provided as function_handle")
        if isinstance(eval_handle, Function):
            raise TypeError("Function cannot be initialized with Function!")

        # handle must return scalar when called with scalar
        testval = self.domain[0][1]
        if testval is np.inf:
            testval = 1
        if not isinstance(eval_handle(testval), Number):
            raise TypeError("callable must return number when called with scalar")
        if vectorial:
            if not isinstance(eval_handle(np.array([testval]*10)), np.ndarray):
                raise TypeError("callable must return np.ndarray when called with vector")
        self._function_handle = eval_handle
        self.vectorial = vectorial

        # derivatives
        for der_handle in derivative_handles:
            if not callable(der_handle):
                raise TypeError("callable has to be provided as member of derivative_handles")
        self._derivative_handles = derivative_handles

    def transformation_hint(self, info, target):
        """
        default method for Functions. If src is a subclass of Function, use default strategy.
        If a different behaviour is desired, overwrite this method.
        :param info:
        :param target:
        :return:
        """
        # TODO handle target option!
        if target is False:
            raise NotImplementedError

        if isinstance(info.src_base[0], Function) and isinstance(info.dst_base[0], Function):
            return self._transformation_factory(info), None
        else:
            raise NotImplementedError

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

        :param value: point(s) where function shall be evaluated
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

    limits = domain_intersection(first.domain, second.domain)
    nonzero = domain_intersection(first.nonzero, second.nonzero)
    areas = domain_intersection(limits, nonzero)

    # try some shortcuts
    if first == second:
        if hasattr(first, "quad_int"):
            return first.quad_int()

    if 0:
        # TODO let Function Class handle product to gain more speed
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
        # res = integrate.quad(function, area[0], area[1])
        res = complex_quadrature(function, area[0], area[1])
        result += res[0]
        err += res[1]

    return np.real_if_close(result), err


def complex_quadrature(func, a, b, **kwargs):
    """
    wraps the scipy qaudpack routines to handle complex valued functions

    :param func: callable
    :param a: lower limit
    :param b: upper limit
    :param kwargs: kwargs for func
    :return:
    """
    def real_func(x):
        return np.real(func(x))

    def imag_func(x):
        return np.imag(func(x))

    real_integral = integrate.quad(real_func, a, b, **kwargs)
    imag_integral = integrate.quad(imag_func, a, b, **kwargs)

    return real_integral[0] +1j*imag_integral[0], real_integral[1] + imag_integral[1]


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
        dot_product_l2.handle = np.vectorize(_dot_product_l2, otypes=[np.complex])
    return np.real_if_close(dot_product_l2.handle(first, second))


def calculate_scalar_matrix(values_a, values_b):
    """
    convenience function wrapper of :py:function:`calculate_scalar_product_matrix` for the case of scalar elements.

    :param values_a:
    :param values_b:
    :return:
    """
    return calculate_scalar_product_matrix(np.multiply,
                                           sanitize_input(values_a, Number),
                                           sanitize_input(values_b, Number))

    # i, j = np.mgrid[0:values_a.shape[0], 0:values_b.shape[0]]
    # vals_i = values_a[i]
    # vals_j = values_b[j]
    # return np.multiply(vals_i, vals_j)


def calculate_scalar_product_matrix(scalar_product_handle, base_a, base_b):
    """
    calculates a matrix :math:`A` whose elements are the scalar products of each element from Bases and b,
    so that :math:`a_{ij} = \\langle \\mathrm{a}_i\\,,\\: \\mathrm{b}_j\\rangle`.

    :param base_a: (array of) BaseFraction
    :param base_b: (array of) BaseFraction
    :return: matrix :math:`A` as np.ndarray
    """
    # TODO make use of symmetry to save some operations
    i, j = np.mgrid[0:base_a.shape[0], 0:base_b.shape[0]]
    funcs_i = base_a[i]
    funcs_j = base_b[j]

    return scalar_product_handle(funcs_i, funcs_j)


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
    scale_mat = calculate_scalar_product_matrix(dot_product_l2, base, base)

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
        res = np.real_if_close(sum([weights[i] * base[i](z) for i in range(weights.shape[0])]), tol=1e6)
        if not all(np.imag(res) == 0):
            print("warning: complex values encountered! {0}".format(np.max(np.imag(res))))
            # return np.real(res)
            return np.zeros_like(z)

        return res

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
    pro_mat = calculate_base_transformation_matrix(src_base, dst_base)
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


class TransformationInfo(object):
    """
    wrapper that holds information about transformations
    """
    def __init__(self):
        self.src_lbl = None
        self.dst_lbl = None
        self.src_base = None
        self.dst_base = None
        self.src_order = None
        self.dst_order = None

    def __hash__(self):
        return hash((self.src_lbl, self.dst_lbl, self.src_order, self.dst_order))

    def __eq__(self, other):
        return (self.src_lbl, self.dst_lbl, self.src_order, self.dst_order) == \
               (other.src_lbl, other.dst_lbl, other.src_order, other.dst_order)


def get_weight_transformation(info):
    """
    somehow calculates a handle that will transform weights from src into weights for dst with the given derivative
    orders.

    :param info: transformation info
    :return: handle
    """
    # trivial case
    if info.src_lbl == info.dst_lbl:
        mat = calculate_expanded_base_transformation_matrix(info.src_base, None, info.src_order, info.dst_order, True)

        def identity(weights):
            return np.dot(mat, weights)

        return identity

    # try to get help from the destination base
    handle, hint = info.dst_base[0].transformation_hint(info, True)
    # if handle is None:
    #     # try source instead
    #     handle, hint = info.src_base[0].transformation_hint(info, False)

    if handle is None:
        raise TypeError("no transformation between given bases possible!")

    # check termination criterion
    if hint is None:
        # direct transformation possible
        return handle

    kwargs = {}
    new_handle = None
    if hasattr(hint, "extras"):
        # try to gain transformations that will satisfy the extra terms
        for dep_lbl, dep_order in hint.extras.iteritems():
            new_info = copy(info)
            new_info.dst_lbl = dep_lbl
            new_info.dst_base = get_initial_functions(dep_lbl, 0)
            new_info.dst_order = dep_order
            dep_handle = get_weight_transformation(new_info)
            kwargs[dep_lbl] = dep_handle

    if hint.src_lbl is not None:
        # transformation to assistant system required
        new_handle = get_weight_transformation(hint)

    def last_handle(weights):
        if new_handle:
            return handle(new_handle(weights), **kwargs)
        else:
            return handle(weights, **kwargs)

    return last_handle


def calculate_expanded_base_transformation_matrix(src_base, dst_base, src_order, dst_order, use_eye=False):
    """
    constructs a transformation matrix from basis given by 'src_base' to basis given by 'dst_base' that also
    transforms all temporal derivatives of the given weights.

    :param src_base: the source basis, given by an array of BaseFractions
    :param dst_base: the destination basis, given by an array of BaseFractions
    :param src_order: temporal derivative order available in src
    :param dst_order: temporal derivative order needed in dst
    :param use_eye: use identity as base transformation matrix
    :return: transformation matrix as 2d np.ndarray
    """
    if src_order < dst_order:
        raise ValueError("higher derivative order needed than provided!")

    # build core transformation
    if use_eye:
        core_transformation = np.eye(src_base.size)
    else:
        core_transformation = calculate_base_transformation_matrix(src_base, dst_base)

    # build block matrix
    part_transformation = block_diag(*[core_transformation for i in range(dst_order + 1)])
    complete_transformation = np.hstack([part_transformation] + [np.zeros((part_transformation.shape[0], src_base.size))
                                                                 for i in range(src_order - dst_order)])
    return complete_transformation


def calculate_base_transformation_matrix(src_base, dst_base):
    """
    calculates the transformation matrix :math:`V` so that the vector of src_weights can be transformed in a vector of
    dst_weights, making the smallest error possible. Quadratic error is used as the error-norm for this case.

    This method assumes that all are members of the given bases are of same type and that the BaseFractions,
    that define them define compatible scalar products. An error will occur otherwise.
    For BaseFractions, that are not compatible in a way that a simple matrix vector multiplication will do the job,
    XXXX should be used.

    :param dst_base: new projection base
    :param src_base: current projection base
    :return:
    """
    src_base = sanitize_input(src_base, BaseFraction)
    dst_base = sanitize_input(dst_base, BaseFraction)

    if not hasattr(src_base[0], "scalar_product_hint"):
        raise TypeError("Input type not supported.")

    # compute P and Q matrices, where P = Sum(P_n) and Q = Sum(Q_n)
    s_hints = src_base[0].scalar_product_hint()
    d_hints = dst_base[0].scalar_product_hint()

    p_matrices = []
    q_matrices = []
    for idx, (s_hint, d_hint) in enumerate(zip(s_hints, d_hints)):
        dst_members = np.array([dst_frac.get_member(idx) for dst_frac in dst_base])
        src_members = np.array([src_frac.get_member(idx) for src_frac in src_base])

        # compute P_n matrix: <phi_tilde_ni(z), phi_dash_nj(z)> for 0 < i < N, 0 < j < M
        p_matrices.append(calculate_scalar_product_matrix(s_hint, dst_members, src_members))

        # compute Q_n matrix: <phi_dash_ni(z), phi_dash_nj(z)> for 0 < i < M, 0 < j < M
        q_matrices.append(calculate_scalar_product_matrix(d_hint, dst_members, dst_members))

    p_mat = np.sum(p_matrices, axis=0)
    q_mat = np.sum(q_matrices, axis=0)

    # compute V matrix: inv(Q)*P
    v_mat = np.dot(np.linalg.inv(q_mat), p_mat)
    return v_mat


# TODO rename to something that emphasizes the general application a little more e.g. normalize_base_fraction
def normalize_function(x1, x2=None):
    """
    takes two the two BaseFractions :math:`\\boldsymbol{x}_1` and  :math:`\\boldsymbol{x}_2` and normalizes them so
    that :math:`\\langle\\boldsymbol{x}_1\\,,\:\\boldsymbol{x}_2\\rangle = 1`.
    If only one function is given, :math:`\\boldsymbol{x}_2` is set to :math:`\\boldsymbol{x}_1`.

    :param x1: core.BaseFraction :math:`\\boldsymbol{x}_1`
    :param x2: core.BaseFraction :math:`\\boldsymbol{x}_2`
    :raises : ValueError if given BaseFraction are orthogonal
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
