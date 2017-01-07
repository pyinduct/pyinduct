"""
In the Core module you can find all basic classes and functions which form the backbone of the toolbox.
"""

from abc import ABCMeta, abstractmethod
from copy import copy, deepcopy
from numbers import Number
import numpy as np
from scipy import integrate
from scipy.linalg import block_diag
from .registry import get_base
import collections


def sanitize_input(input_object, allowed_type):
    """
    Sanitizes input data by testing if *input_object* is an array of type *allowed_type*.

    Args:
        input_object: Object which is to be checked.
        allowed_type: desired type

    Return:
        input_object
    """
    input_object = np.atleast_1d(input_object)
    for obj in np.nditer(input_object, flags=["refs_ok"]):
        if not isinstance(np.asscalar(obj), allowed_type):
            raise TypeError("Only objects of type: {0} accepted.".format(allowed_type))

    return input_object


class BaseFraction:
    """
    Abstract base class representing a basis that can be used to describe functions of several variables.
    """

    def __init__(self, members):
        self.members = members

    @staticmethod
    def _transformation_factory(info):
        mat = calculate_expanded_base_transformation_matrix(info.src_base, info.dst_base, info.src_order,
                                                            info.dst_order)

        def handle(weights):
            return np.dot(mat, weights)

        return handle

    def transformation_hint(self, info, target):
        """
        Method that provides a information about how to transform weights from one :py:class:`BaseFraction` into
        another.

        In Detail this function has to return a callable, which will take the weights of the source- and return the
        weights of the target system. It may have keyword arguments for other data which is required to perform the
        transformation.
        Information about these extra keyword arguments should be provided in form of a dictionary whose keys are
        keyword arguments of the returned transformation handle.

        Note:
            This implementation covers the most basic case, where the two :py:class:`BaseFraction` s are of same type.
            For any other case it will raise an exception.
            Overwrite this Method in your implementation to support conversion between bases that differ from yours.

        Args:
            info: :py:class:`TransformationInfo`
            target: :py:class:`TransformationInfo`

        Raises:
            NotImplementedError:

        Returns:
            Transformation handle
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
                  "But this is a dumb method so implement your own hint to make things work!".format(info.dst_lbl,
                self.__class__.__name__, info.src_lbl, info.src_base[0].__class__.__name__,
                info.dst_base[0].__class__.__name__, info.src_order, info.dst_order)
            raise NotImplementedError(msg)

    def scalar_product_hint(self):
        """
        Empty Hint that can return steps for scalar product calculation.

        In detail this means a list object containing function calls to fill with (first, second) parameters
        that will calculate the scalar product when summed up.

        Note:
            Overwrite to implement custom functionality.
            For an example implementation see :py:class:`Function`
        """
        pass

    def derive(self, order):
        """
        Basic implementation of derive function.
        Empty implementation, overwrite to use this functionality.
        For an example implementation see :py:class:`Function`

        Args:
            order (:class:`numbers.Number`): derivative order
        Return:
            :py:class:`BaseFraction`: derived object
        """
        if order == 0:
            return self
        else:
            raise NotImplementedError("This is an empty function."
                                      " Overwrite it in your implementation to use this functionality.")

    def scale(self, factor):
        """
        Factory method to obtain instances of this base fraction, scaled by the given factor.
        Empty function, overwrite to implement custom functionality.
        For an example implementation see :py:class:`Function`

        Args:
            factor: Factor to scale the vector.
        """
        raise NotImplementedError("This is an empty function."
                                  " Overwrite it in your implementation to use this functionality.")

    def get_member(self, idx):
        """
        Getter function to access members.
        Empty function, overwrite to implement custom functionality.
        For an example implementation see :py:class:`Function`

        Note:
            Empty function, overwrite to implement custom functionality.

        Args:
            idx: member index
        """
        raise NotImplementedError("This is an empty function."
                                  " Overwrite it in your implementation to use this functionality.")


class Function(BaseFraction):
    """
    Most common instance of a :py:class:`BaseFraction`.
    This class handles all tasks concerning derivation and evaluation of functions. It is used broad across the toolbox
    and therefore incorporates some very specific attributes.
    For example, to ensure the accurateness of numerical handling functions may only evaluated in areas where they
    provide nonzero return values. Also their domain has to be taken into account. Therefore the attributes *domain*
    and *nonzero* are provided.

    To save implementation time, ready to go version like :py:class:`pyinduct.shapefunctions.LagrangeFirstOrder`
    are provided in the :py:mod:`pyinduct.simulation` module.

    For the implementation of new shape functions subclass this implementation or directly provide a callable
    *eval_handle* and callable *derivative_handles* if spatial derivatives are required for the application.
    """

    # TODO: overload add and mul operators

    def __init__(self, eval_handle, domain=(-np.inf, np.inf), nonzero=(-np.inf, np.inf), derivative_handles=None):
        """
        Args:
            eval_handle: Callable object that can be evaluated.
            domain: Domain on which the eval_handle is defined.
            nonzero: Region in which the eval_handle will give nonzero output.
            derivative_handles (list): List of callable(s) that contain derivatives of eval_handle
        """
        super().__init__(self)
        self._vectorial = False
        self._function_handle = None
        self._derivative_handles = None

        # domain and nonzero area
        for kw, val in zip(["domain", "nonzero"], [domain, nonzero]):
            if not isinstance(val, list):
                if isinstance(val, tuple):
                    val = [val]
                else:
                    raise TypeError("List of tuples has to be provided for {0}".format(kw))
            setattr(self, kw, sorted([(min(interval), max(interval)) for interval in val], key=lambda x: x[0]))

        self.function_handle = eval_handle
        self.derivative_handles = derivative_handles

    @property
    def derivative_handles(self):
        return self._derivative_handles

    @derivative_handles.setter
    def derivative_handles(self, eval_handle_derivatives):
        if eval_handle_derivatives is None:
            eval_handle_derivatives = []
        if not isinstance(eval_handle_derivatives, collections.Iterable):
            eval_handle_derivatives = [eval_handle_derivatives]
        for der_handle in eval_handle_derivatives:
            if not isinstance(der_handle, collections.Callable):
                raise TypeError("derivative_handles must be callable")
        self._derivative_handles = eval_handle_derivatives

    @property
    def function_handle(self):
        return self._function_handle

    @function_handle.setter
    def function_handle(self, eval_handle):
        # handle must be callable
        if not isinstance(eval_handle, collections.Callable):
            raise TypeError("callable has to be provided as function_handle")

        # handle must return scalar when called with scalar
        test_value = self.domain[0][1]
        if test_value is np.inf:
            test_value = 1
        if not isinstance(eval_handle(test_value), Number):
            raise TypeError("callable must return number when called with scalar")

        self._function_handle = eval_handle

        # test vectorial input
        test_data = np.array([test_value] * 10)
        try:
            res = eval_handle(test_data)
        except BaseException as e:
            # looks like the function does _not_ handle vectorial input
            self._vectorial = False
            return

        if not isinstance(res, np.ndarray):
            # raise TypeError("callable must return np.ndarray when called with vector")
            self._vectorial = False
            return
        if res.shape != test_data.shape:
            # raise TypeError("result of call with vector must be of same shape")
            self._vectorial = False
            return

        self._vectorial = True

    def _check_domain(self, values):
        """
        Checks if values fit into domain.

        Args:
            values (array_like): Point(s) where function shall be evaluated.

        Raises:
            ValueError: If values exceed the domain.
        """
        # in_domain = False
        values = np.atleast_1d(values)
        for interval in self.domain:
            if any(values < interval[0]) or any(values > interval[1]):
                raise ValueError("Function evaluated outside it's domain with {}".format(values))

                # if all(value >= interval[0]) and all(value <= interval[1]):
                #     in_domain = True
                #     break

                # if not in_domain:
                #     raise ValueError("Function evaluated outside its domain!")

    def __call__(self, argument):
        """
        Handle that is used to evaluate the function on a given point.

        Args:
            argument: Function parameter

        Return:
            function value
        """
        self._check_domain(argument)
        if self._vectorial:
            if not isinstance(argument, np.ndarray):
                # a little convenience helper here
                argument = np.array(argument)
            return self._function_handle(argument)
        else:
            try:
                ret_val = []
                for arg in argument:
                    ret_val.append(self._function_handle(arg))

                return np.array(ret_val)
            except TypeError as e:
                return self._function_handle(argument)

    def get_member(self, idx):
        """
        Implementation of the abstract parent method.

        Since the :py:class:`Function` has only one member (itself) the parameter *idx* is ignored and *self* is
        returned.

        Args:
            idx: ignored.

        Return:
            self
        """
        return self

    def raise_to(self, power):
        """
        Raises the function to the given *power*.

        Warning:
            Derivatives are lost after this action is performed.

        Args:
            power (:obj:`numbers.Number`): power to raise the function to

        Return:
            raised function
        """
        if power == 1:
            return self

        def raise_factory(func):
            def _raised_func(z):
                return np.power(func(z), power)

            return _raised_func

        new_obj = deepcopy(self)
        new_obj.derivative_handles = None
        new_obj.function_handle = raise_factory(self.function_handle)
        return new_obj

    def scale(self, factor):
        """
        Factory method to scale a :py:class:`pyinduct.core.Function`.

        Args:
            factor : :obj:`numbers.Number` or a callable.
        """
        if factor == 1:
            return self

        def scale_factory(func):
            def _scaled_func(z):
                if isinstance(factor, collections.Callable):
                    return factor(z) * func(z)
                else:
                    return factor * func(z)

            return _scaled_func

        new_obj = deepcopy(self)
        if isinstance(factor, collections.Callable):
            # derivatives are lost
            new_obj.derivative_handles = None
            new_obj.function_handle = scale_factory(self._function_handle)
        else:
            # derivatives can be scaled
            new_obj.derivative_handles = [scale_factory(der_handle) for der_handle in self.derivative_handles]
            new_obj.function_handle = scale_factory(self._function_handle)

        return new_obj

    def derive(self, order=1):
        """
        Spatially derive this :py:class:`Function`.

        This is done by neglecting *order* derivative handles and to select handle :math:`\\text{order} - 1` as the new
        evaluation_handle.

        Args:
            order (int): the amount of derivations to perform

        Raises:
            TypeError: If *order* is not of type int.
            ValueError: If the requested derivative order is higher than the provided one.

        Returns:
            :py:class:`Function` the derived function.

        """
        if not isinstance(order, int):
            raise TypeError("only integer allowed as derivation order")
        if order == 0:
            return self
        if order < 0 or order > len(self.derivative_handles):
            raise ValueError("function cannot be differentiated that often.")

        new_obj = deepcopy(self)
        new_obj.derivative_handles = self.derivative_handles[order - 1:]
        new_obj.function_handle = new_obj.derivative_handles.pop(0)
        return new_obj

    def evaluation_hint(self, values):
        """
        If evaluation can be accelerated by using special properties of a function, this function can be
        overwritten to performs that computation. It gets passed an array of places where the caller
        wants to evaluate the function and should return an array of the same length, containing the results.

        Note:
            This implementation just calls the normal evaluation hook.

        Args:
            values: places to be evaluated at

        Returns:
            numpy.ndarray: Evaluation results.
        """
        return self(values)

    def transformation_hint(self, info, target):
        """
        If *info.src_base* is a subclass of Function, use default strategy.

        Note:
            If a different behaviour is desired, overwrite this method.

        Args:
            info (:py:class:`TransformationInfo`): Information about the requested transformation.
            target (bool): Is the called object the target of the transformation?
                If False, source and target in *info* will be swapped.

        Return:
            transformation handle

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
        Return the hint that the :py:func:`pyinduct.core.dot_product_l2` has to calculated to gain the scalar product.
        """
        return [dot_product_l2]


class ComposedFunctionVector(BaseFraction):
    """
    Implementation of composite function vector :math:`\\boldsymbol{x}`.

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
        return [dot_product_l2 for funcs in self.members["funcs"]] + [np.multiply for scals in self.members["scalars"]]

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
    Calculate intersection(s) of two domains.

    Args:
        first (:py:class:`pyinduct.simulation.Domain`): first domain
        second (:py:class:`pyinduct.simulation.Domain`): second domain

    Return:
        list: intersection(s) given by (start, end) tuples.
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
    Calculates the inner product of two functions.

    Args:
        first (:py:class:`pyinduct.core.Function`): first function
        second (:py:class:`pyinduct.core.Function`): second function

    Todo:
        rename to scalar_dot_product and make therefore non private

    Return:
        inner product
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
    def function(z):
        return first(z) * second(z)

    result, error = integrate_function(function, areas)

    return result


def integrate_function(function, interval):
    # TODO: Wrong documentation of argument interval.
    # TODO: Unexepted types of (Return) tuple elements (one numpy.array and one float). This should be changed or documented if it is desired.
    """
    Integrates the given *function* over the *interval* using :func:`complex_quadrature`.

    Args:
        function(callable): Function to integrate.
        interval(Tuple): (start, end) values of the Interval to integrate on.

    Return:
        tuple: (Result of the Integration, errors that occurred during the integration).

    """
    result = 0
    err = 0
    for area in interval:
        res = complex_quadrature(function, area[0], area[1])
        result += res[0]
        err += res[1]

    return np.real_if_close(result), err


def complex_quadrature(func, a, b, **kwargs):
    """
    Wraps the scipy.qaudpack routines to handle complex valued functions.

    Args:
        func (callable): function
        a (:obj:`numbers.Number`): lower limit
        b (:obj:`numbers.Number`): upper limit
        **kwargs: Arbitrary keyword arguments for desired scipy.qaudpack routine.

    Return:
        tuple: (real part, imaginary part)
    """

    def real_func(x):
        return np.real(func(x))

    def imag_func(x):
        return np.imag(func(x))

    real_integral = integrate.quad(real_func, a, b, **kwargs)
    imag_integral = integrate.quad(imag_func, a, b, **kwargs)

    return real_integral[0] + 1j * imag_integral[0], real_integral[1] + imag_integral[1]


def dot_product(first, second):
    """
    Calculates the inner product of two vectors.

    Args:
        first (:obj:`numpy.ndarray`): first vector
        second (:obj:`numpy.ndarray`): second vector

    Return:
        inner product
    """
    return np.inner(first, second)


def dot_product_l2(first, second):
    """
    Vectorized version of dot_product.

    Args:
        first (callable or :obj:`numpy.ndarray`):  (1d array of) callable(s)
        second (callable or :obj:`numpy.ndarray`):  (1d array of) callable(s)

    Return:
        numpy.ndarray:  array of inner products
    """
    # sanitize input
    first = np.atleast_1d(first)
    second = np.atleast_1d(second)

    # if len(first.shape) == 2 or len(second.shape) == 2:
    #     if first.shape[1] > 1 or second.shape[1] > 1:
    #         raise NotImplementedError("input dimension not understood.")

    # calculate output size and allocate output
    out = np.ones(first.shape) * np.nan

    # TODO propagate vectorization into _dot_product_l2 to save this loop
    # loop over entries
    for idx, (f, s) in enumerate(zip(first, second)):
        out[idx] = _dot_product_l2(f, s)

    return out


def calculate_scalar_matrix(values_a, values_b):
    """
    Convenience version of py:function:`calculate_scalar_product_matrix` with :py:func:`numpy.multiply` hardcoded as
    *scalar_product_handle*.

    Args:
        values_a (numbers.Number or numpy.ndarray): (array of) value(s) for rows
        values_b (numbers.Number or numpy.ndarray): (array of) value(s) for columns

    Return:
        numpy.ndarray: Matrix containing the pairwise products of the elements from *values_a* and *values_b*.
    """
    return calculate_scalar_product_matrix(np.multiply, sanitize_input(values_a, Number),
                                           sanitize_input(values_b, Number))


def calculate_scalar_product_matrix(scalar_product_handle, base_a, base_b, optimize=False):
    """
    Calculates a matrix :math:`A` , whose elements are the scalar products of each element from *base_a* and *base_b*,
    so that :math:`a_{ij} = \\langle \\mathrm{a}_i\\,,\\: \\mathrm{b}_j\\rangle`.

    Args:
        scalar_product_handle (callable): function handle that is called to calculate the scalar product.
            This function has to be able to cope with (1d) vectorial input.
        base_a (numpy.ndarray): array of :py:class:`BaseFraction`
        base_b (numpy.ndarray): array of :py:class:`BaseFraction`
        optimize (bool): switch to turn on the symmetry based speed up. For development purposes only.

    TODO:
        making use of the commutable scalar product could save time, run some test on this

    Return:
        numpy.ndarray: matrix :math:`A`
    """
    if optimize:
        raise NotImplementedError("this approach leads to wrong results atm.")
        # There are certain conditions that hvae to be satisfied to make use of a symmetrical procut matrix
        # not all of these conditions are checked here and the implementation itself is not yet free from errros.

        # if scalar_product handle commutes whe can save some operations
        if base_a.size > base_b.size:
            base_1 = base_a
            base_2 = base_b
            transposed = False
        else:
            base_1 = base_b
            base_2 = base_a
            transposed = True

        # if 0:
        #     # one way
        #     idx_1 = []
        #     idx_2 = []
        #     for n in range(base_1.shape[0]):
        #         for m in range(min(n + 1, base_2.shape[0])):
        #             idx_1.append(n)
        #             idx_2.append(m)
        #
        #     fractions_1 = base_1[np.array(idx_1)]
        #     fractions_2 = base_2[np.array(idx_2)]
        # else:
        #     # another way not really working yet

        i, j = np.mgrid[0:base_1.shape[0], 0:base_2.shape[0]]

        end_shape = (base_1.shape[0], base_2.shape[0])
        lower_mask = np.tril(np.ones(end_shape))
        lower_idx = np.flatnonzero(lower_mask)

        i_lower = i.flatten()[lower_idx]
        j_lower = j.flatten()[lower_idx]
        fractions_1 = base_1[i_lower]
        fractions_2 = base_2[j_lower]

        # compute
        res = scalar_product_handle(fractions_1, fractions_2)

        # reconstruct
        end_shape = (base_1.shape[0], base_2.shape[0])
        lower_part = np.zeros(end_shape).flatten()

        # reconstruct lower half
        lower_part[lower_idx] = res
        lower_part = lower_part.reshape(end_shape)
        # print(lower_part)

        # mirror symmetrical half
        upper_part = np.zeros(end_shape).flatten()
        temp_out = copy(lower_part[:base_2.shape[0], :].T)
        upper_mask = np.triu(np.ones_like(temp_out), k=1)
        upper_idx = np.flatnonzero(upper_mask)
        # print(upper_mask)
        # print(upper_idx)

        upper_part[upper_idx] = temp_out.flatten()[upper_idx]
        upper_part = upper_part.reshape(end_shape)
        # print(upper_part)

        out = lower_part + upper_part
        return out if not transposed else out.T

    else:
        i, j = np.mgrid[0:base_a.shape[0], 0:base_b.shape[0]]
        fractions_i = base_a[i]
        fractions_j = base_b[j]

        res = scalar_product_handle(fractions_i.flatten(), fractions_j.flatten())
        return res.reshape(fractions_i.shape)


def project_on_base(function, base):
    """
    Projects a *function* on a basis given by *base*.

    Args:
        function (:py:class:`Function`): Function to approximate.
        base: Single :py:class:`Function` or :obj:`numpy.ndarray` that generates a basis.

    Return:
        numpy.ndarray: Weight vector in the given *base*
    """
    if isinstance(base, Function):  # convenience case
        base = np.asarray([base])

    if not isinstance(base, np.ndarray):
        raise TypeError("Only numpy.ndarray accepted as 'initial_functions'")

    # compute <x(z, t), phi_i(z)> (vector)
    projections = calculate_scalar_product_matrix(dot_product_l2, np.array([function]), base).flatten()

    # compute <phi_i(z), phi_j(z)> for 0 < i, j < n (matrix)
    scale_mat = calculate_scalar_product_matrix(dot_product_l2, base, base)

    return np.dot(np.linalg.inv(scale_mat), projections)


def back_project_from_base(weights, base):
    """
    Build evaluation handle for a distributed variable that was approximated as a set of *weights* om a certain *base*.

    Args:
        weights (numpy.ndarray): Weight vector.
        base (numpy.ndarray): Vector that generates the base.

    Return:
        evaluation handle
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
        # TODO call uniform complex converter instead
        res = np.real_if_close(sum([weights[i] * base[i](z) for i in range(weights.shape[0])]), tol=1e6)
        if not all(np.imag(res) == 0):
            print(("warning: complex values encountered! {0}".format(np.max(np.imag(res)))))
            # return np.real(res)
            return np.zeros_like(z)

        return res

    return eval_handle


def change_projection_base(src_weights, src_base, dst_base):
    """
    Converts given weights that form an approximation using *src_base* to the best possible fit using
    *dst_base*. Bases can be given as :py:class:`BaseFraction` array.

    Args:
        src_weights (numpy.ndarray): Vector of numbers.
        src_base (numpy.ndarray): Vector of :py:class:`BaseFraction` s that generate the source basis
        dst_base (numpy.ndarray): Vector of :py:class:`BaseFraction` s that generate the target basis

    Return:
        :obj:`numpy.ndarray`: target weights
    """
    pro_mat = calculate_base_transformation_matrix(src_base, dst_base)
    return project_weights(pro_mat, src_weights)


def project_weights(projection_matrix, src_weights):
    """
    Project *src_weights* on new basis using the provided *projection_matrix*.

    Args:
        projection_matrix (:py:class:`numpy.ndarray`): projection between the source and the target basis;
            dimension (m, n)
        src_weights (:py:class:`numpy.ndarray`): weights in the source basis; dimension (1, m)

    Return:
        :py:class:`numpy.ndarray`: weights in the target basis; dimension (1, n)
    """
    if isinstance(src_weights, Number):
        src_weights = np.asarray([src_weights])

    return np.dot(projection_matrix, src_weights)


class TransformationInfo:
    """
    Structure that holds information about transformations between different bases.

    This class serves as an easy to use structure to aggregate information, describing transformations between different
    :py:class:`BaseFraction` s. It can be tested for equality to check the equity of transformations and is hashable
    which makes it usable as dictionary key to cache different transformations.

    Attributes:
        src_lbl(str): label of source basis
        dst_lbl(str): label destination basis
        src_base(:obj:`numpy.ndarray`): source basis in form of an array of the source Fractions
        dst_base(:obj:`numpy.ndarray`): destination basis in form of an array of the destination Fractions
        src_order: available temporal derivative order of source weights
        dst_order: needed temporal derivative order for destination weights
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
        return (self.src_lbl, self.dst_lbl, self.src_order, self.dst_order) == (
        other.src_lbl, other.dst_lbl, other.src_order, other.dst_order)


def get_weight_transformation(info):
    """
    Create a handle that will transform weights from *info.src_base* into weights for *info-dst_base*
    while paying respect to the given derivative orders.

    This is accomplished by recursively iterating through source and destination bases and evaluating their
    :attr:`transformation_hints`.

    Args:
        info(py:class:`TransformationInfo`): information about the requested transformation.

    Return:
        callable: transformation function handle
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
        for dep_lbl, dep_order in hint.extras.items():
            new_info = copy(info)
            new_info.dst_lbl = dep_lbl
            new_info.dst_base = get_base(dep_lbl, 0)
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
    Constructs a transformation matrix :math:`\\bar V` from basis given by *src_base* to basis given
    by *dst_base* that also transforms all temporal derivatives of the given weights.

    See:
        :py:func:`calculate_base_transformation_matrix` for further details.

    Args:
        dst_base (:py:class:`BaseFraction`): New projection base.
        src_base (:py:class:`BaseFraction`): Current projection base.
        src_order: Temporal derivative order available in *src_base*.
        dst_order: Temporal derivative order needed in *dst_base*.
        use_eye (bool): Use identity as base transformation matrix. (For selection of derivatives in the same base)

    Raises:
        ValueError: If destination needs a higher derivative order than source can provide.

    Return:
        :obj:`numpy.ndarray`: Transformation matrix
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
    complete_transformation = np.hstack(
        [part_transformation] + [np.zeros((part_transformation.shape[0], src_base.size)) for i in
                                 range(src_order - dst_order)])
    return complete_transformation


def calculate_base_transformation_matrix(src_base, dst_base):
    """
    Calculates the transformation matrix :math:`V` , so that the a set of weights, describing a function in the
    *src_base* will express the same function in the *dst_base*, while minimizing the reprojection error.
    An quadratic error is used as the error-norm for this case.

    Warning:
        This method assumes that all members of the given bases have the same type and that their
        :py:class:`BaseFraction` s, define compatible scalar products.

    Raises:
        TypeError: If given bases do not provide an :py:func:`scalar_product_hint` method.

    Args:
        dst_base (:py:class:`BaseFraction`): New projection base.
        src_base (:py:class:`BaseFraction`): Current projection base.

    Return:
        :py:class:`numpy.ndarray`: Transformation matrix :math:`V` .
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


def normalize_base(b1, b2=None):
    """
    Takes two arrays of :py:class:`BaseFraction` s :math:`\\boldsymbol{b}_1` and  :math:`\\boldsymbol{b}_1` and normalizes them so
    that :math:`\\langle\\boldsymbol{b}_{1i}\\,,\:\\boldsymbol{b}_{2i}\\rangle = 1`.
    If only one base is given, :math:`\\boldsymbol{b}_2` is set to :math:`\\boldsymbol{b}_1`.

    Args:
        b1 (np.array of :py:class:`BaseFraction`): :math:`\\boldsymbol{b}_1`
        b2 (np.array of :py:class:`BaseFraction`): :math:`\\boldsymbol{b}_2`

    Raises:
        ValueError: If :math:`\\boldsymbol{b}_1` and :math:`\\boldsymbol{b}_2` are orthogonal.

    Return:
        np.array of :py:class:`BaseFraction` : if *b2* is None,
           otherwise: Tuple of 2 :py:class:`BaseFraction` arrays.
    """
    auto_normalization = False
    if b2 is None:
        b2 = b1
        auto_normalization = True

    if type(b1) != type(b2):
        raise TypeError("only arguments of same type allowed.")

    if not hasattr(b1[0], "scalar_product_hint"):
        raise TypeError("Input type not supported.")

    hints = b1[0].scalar_product_hint()
    res = np.zeros(b1.shape)
    for idx, hint in enumerate(hints):
        members_1 = np.array([fraction.get_member(idx) for fraction in b1])
        members_2 = np.array([fraction.get_member(idx) for fraction in b2])
        res += hint(members_1, members_2)

    if any(res < np.finfo(float).eps):
        if any(np.isclose(res, 0)):
            raise ValueError("given base fractions are orthogonal. no normalization possible.")
        else:
            raise ValueError("imaginary scale required. no normalization possible.")

    scale_factors = np.sqrt(1 / res)
    b1_scaled = np.array([frac.scale(factor) for frac, factor in zip(b1, scale_factors)])

    if auto_normalization:
        return b1_scaled
    else:
        b2_scaled = np.array([frac.scale(factor) for frac, factor in zip(b2, scale_factors)])
        return b1_scaled, b2_scaled
