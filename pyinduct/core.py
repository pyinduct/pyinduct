"""
In the Core module you can find all basic classes and functions which form the backbone of the toolbox.
"""
import warnings

import numpy as np
import collections
from copy import copy, deepcopy
from numbers import Number

from scipy import integrate
from scipy.linalg import block_diag
from scipy.optimize import root
from scipy.interpolate import interp1d

from .registry import get_base

__all__ = ["Domain", "EvalData", "Parameters", "find_roots", "sanitize_input",
           "real", "Base", "BaseFraction", "StackedBase", "Function",
           "ComposedFunctionVector", "normalize_base", "project_on_base",
           "change_projection_base", "project_on_bases",
           "back_project_from_base", "calculate_scalar_product_matrix",
           "calculate_base_transformation_matrix",
           "calculate_expanded_base_transformation_matrix", "dot_product_l2",
           "generic_scalar_product"]


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

    def scalar_product_hint(self):
        """
        Empty Hint that can return steps for scalar product calculation.

        In detail this means a list object containing function calls to fill
        with (first, second) parameters that will calculate the scalar product
        when summed up.

        Note:
            Overwrite to implement custom functionality.
            For an example implementation see :py:class:`.Function`
        """
        pass

    def derive(self, order):
        """
        Basic implementation of derive function.
        Empty implementation, overwrite to use this functionality.
        For an example implementation see :py:class:`.Function`

        Args:
            order (:class:`numbers.Number`): derivative order
        Return:
            :py:class:`.BaseFraction`: derived object
        """
        if order == 0:
            return self
        else:
            raise NotImplementedError("This is an empty function."
                                      " Overwrite it in your implementation to use this functionality.")

    def scale(self, factor):
        """
        Factory method to obtain instances of this base fraction, scaled by the
        given factor. Empty function, overwrite to implement custom
        functionality. For an example implementation see :py:class:`.Function`.

        Args:
            factor: Factor to scale the vector.
        """
        raise NotImplementedError("This is an empty function."
                                  " Overwrite it in your implementation to use this functionality.")

    def raise_to(self, power):
        """
        Raises this fraction to the given *power*.

        Args:
            power (:obj:`numbers.Number`): power to raise the fraction onto

        Return:
            raised fraction
        """
        if power == 1:
            return self
        else:
            raise NotImplementedError("Implement this functionality to make use of it.")

    def get_member(self, idx):
        """
        Getter function to access members.
        Empty function, overwrite to implement custom functionality.
        For an example implementation see :py:class:`.Function`

        Note:
            Empty function, overwrite to implement custom functionality.

        Args:
            idx: member index
        """
        raise NotImplementedError("This is an empty function."
                                  " Overwrite it in your implementation to use this functionality.")


class Function(BaseFraction):
    """
    Most common instance of a :py:class:`.BaseFraction`.
    This class handles all tasks concerning derivation and evaluation of
    functions. It is used broad across the toolbox and therefore incorporates
    some very specific attributes. For example, to ensure the accurateness of
    numerical handling functions may only evaluated in areas where they provide
    nonzero return values. Also their domain has to be taken into account.
    Therefore the attributes *domain* and *nonzero* are provided.

    To save implementation time, ready to go version like
    :py:class:`.LagrangeFirstOrder` are provided in the
    :py:mod:`pyinduct.simulation` module.

    For the implementation of new shape functions subclass this implementation
    or directly provide a callable *eval_handle* and callable
    *derivative_handles* if spatial derivatives are required for the
    application.
    """

    # TODO: overload add and mul operators

    def __init__(self, eval_handle, domain=(-np.inf, np.inf), nonzero=(-np.inf, np.inf), derivative_handles=None):
        """
        Args:
            eval_handle (callable): Callable object that can be evaluated.
            domain((list of) tuples: Domain on which the eval_handle is defined.
            nonzero(tuple): Region in which the eval_handle will return
                nonzero output. Must be a subset of *domain*
            derivative_handles (list): List of callable(s) that contain
                derivatives of eval_handle
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
                    print(domain)
                    raise TypeError("List or tuple has to be provided for {0}".format(kw))
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
            print(test_value)
            print(type(eval_handle(test_value)))
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
                raise ValueError("Function evaluated outside it's "
                                 "domain {} with {}".format(self.domain,
                                                            values))

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

        Since the :py:class:`.Function` has only one member (itself) the
        parameter *idx* is ignored and *self* is returned.

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
        Factory method to scale a :py:class:`.Function`.

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
        r"""
        Spatially derive this :py:class:`.Function`.

        This is done by neglecting *order* derivative handles and to select
        handle :math:`\text{order} - 1` as the new evaluation_handle.

        Args:
            order (int): the amount of derivations to perform

        Raises:
            TypeError: If *order* is not of type int.
            ValueError: If the requested derivative order is higher than the
                provided one.

        Returns:
            :py:class:`.Function` the derived function.

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

    def scalar_product_hint(self):
        """
        Return the hint that the :py:func:`.dot_product_l2` has to calculated to
        gain the scalar product.
        """
        return [dot_product_l2]

    @staticmethod
    def from_constant(constant, **kwargs):
        """
        Create a :py:class:`.Function` that returns a constant value.

        Args:
            constant (number): value to return
            **kwargs: all kwargs get passed to :py:class:`.Function`

        Returns:
            :py:class:`.Function`: A constant function
        """
        def f(z):
            return constant

        def f_dz(z):
            return 0

        func = Function(eval_handle=f, derivative_handles=[f_dz], **kwargs)
        return func

    @staticmethod
    def from_data(x, y, **kwargs):
        """
        Create a :py:class:`.Function` based on discrete data by
        interpolating.

        The interpolation is done by using :py:class:`interp1d` from scipy,
        the *kwargs* will be passed.

        Args:
            x (array-like): Places where the function has been evaluated .
            y (array-like): Function values at *x*.
            **kwargs: all kwargs get passed to :py:class:`.Function` .

        Returns:
            :py:class:`.Function`: An interpolating function.
        """
        dom = kwargs.pop("domain", (min(x), max(x)))
        nonzero = kwargs.pop("nonzero", dom)
        der_handles = kwargs.pop("derivative_handles", None)

        interp = interp1d(x, y, **kwargs)

        # TODO fix this behaviour
        def wrapper(z):
            res = interp(z)
            if res.size == 1:
                return np.float(res)
            return res

        func = Function(eval_handle=wrapper,
                        domain=dom,
                        nonzero=nonzero,
                        derivative_handles=der_handles)

        return func


class ComposedFunctionVector(BaseFraction):
    r"""
    Implementation of composite function vector :math:`\boldsymbol{x}`.

    .. math::
        \boldsymbol{x} = \begin{pmatrix}
            x_1(z) \\
            \vdots \\
            x_n(z) \\
            \xi_1 \\
            \vdots \\
            \xi_m \\
        \end{pmatrix}
    """

    def __init__(self, functions, scalars):
        funcs = sanitize_input(functions, Function)
        scals = sanitize_input(scalars, Number)

        BaseFraction.__init__(self, {"funcs": funcs, "scalars": scals})

    def scalar_product_hint(self):
        return [dot_product_l2 for funcs in self.members["funcs"]] \
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


class Base:
    """
    Base class for approximation bases.
    In general, a :py:class:`.Base` is formed by a certain amount of
    :py:class:`.BaseFractions` and therefore forms finite-dimensional subspace
    of the distributed problem's domain. Most of the time, the user does not
    need to interact with this class.

    Args:
        fractions (iterable of :py:class:`.BaseFraction`): List, array or
            dict of :py:class:`.BaseFraction`'s
    """
    def __init__(self, fractions):
        # TODO check if Fractions are consistent in Type and provided hints
        self.fractions = sanitize_input(fractions, BaseFraction)

    def __iter__(self):
        return iter(self.fractions)

    def __len__(self):
        return len(self.fractions)

    def __getitem__(self, item):
        return self.fractions[item]

    @staticmethod
    def _transformation_factory(info):
        mat = calculate_expanded_base_transformation_matrix(info.src_base, info.dst_base, info.src_order,
                                                            info.dst_order)

        def handle(weights):
            return np.dot(mat, weights)

        return handle

    def transformation_hint(self, info):
        """
        Method that provides a information about how to transform weights from
        one :py:class:`.BaseFraction` into another.

        In Detail this function has to return a callable, which will take the
        weights of the source- and return the weights of the target system. It
        may have keyword arguments for other data which is required to perform
        the transformation. Information about these extra keyword arguments
        should be provided in form of a dictionary whose keys are keyword
        arguments of the returned transformation handle.

        Note:
            This implementation covers the most basic case, where the two
            :py:class:`.BaseFraction`'s are of same type. For any other case it
            will raise an exception. Overwrite this Method in your
            implementation to support conversion between bases that differ from
            yours.

        Args:
            info: :py:class:`.TransformationInfo`

        Raises:
            NotImplementedError:

        Returns:
            Transformation handle
        """
        if info.src_base.__class__ == info.dst_base.__class__:
            return self._transformation_factory(info), None
        else:
            # No Idea what to do.
            return None, None

    def scalar_product_hint(self):
        """
        Hint that returns steps for scalar product calculation with elements of
        this base.

        Note:
            Overwrite to implement custom functionality.
            For an example implementation see :py:class:`.Function`
        """
        return self.fractions[0].scalar_product_hint()

    def derive(self, order):
        """
        Basic implementation of derive function.
        Empty implementation, overwrite to use this functionality.
        For an example implementation see :py:class:`.Function`

        Args:
            order (:class:`numbers.Number`): derivative order
        Return:
            :py:class:`.Base`: derived object
        """
        if order == 0:
            return self
        else:
            return self.__class__([f.derive(order) for f in self.fractions])

    def scale(self, factor):
        """
        Factory method to obtain instances of this base, scaled by the given factor.

        Args:
            factor: factor or function to scale this base with.
        """
        if factor == 1:
            return self
        else:
            return self.__class__([f.scale(factor) for f in self.fractions])

    def raise_to(self, power):
        """
        Factory method to obtain instances of this base, raised by the given power.

        Args:
            power: power to raise the basis onto.
        """
        if power == 1:
            return self
        else:
            return self.__class__([f.raise_to(power) for f in self.fractions])


class StackedBase(Base):
    """
    Implementation of a basis vector that is obtained by stacking different bases onto each other.
        This typically occurs when the bases of coupled systems are joined to create a unified system.

    Args:
        fractions (dict): Dictionary with base_label and corresponding function
    """

    def __init__(self, fractions, base_info):
        super().__init__(fractions)
        self._info = base_info

    def scalar_product_hint(self):
        return [dot_product_l2 for k in self.members.keys()]

    def get_member(self, idx):
        return list(self.members.values())[idx]

    def scale(self, factor):
        return self.__class__({lbl: func.scale(factor) for lbl, func in self.members})

    def transformation_hint(self, info):
        """
        If *info.src_lbl* is a member, just return it, using to correct derivative transformation, otherwise
        return `None`

        Args:
            info (:py:class:`.TransformationInfo`): Information about the
                requested transformation.
        Return:
            transformation handle

        """
        # we only know how to get from a stacked base to one of our parts
        if info.src_base.__class__ != self.__class__ or info.dst_lbl not in self._info.keys():
            return None, None

        # we can help
        start_idx = self._info[info.dst_lbl]["start"]
        sel_len = self._info[info.dst_lbl]["size"]
        src_ord = self._info[info.dst_lbl]["order"]
        trans_mat = calculate_expanded_base_transformation_matrix(info.dst_base, info.dst_base,
                                                                  src_ord, info.dst_order,
                                                                  use_eye=True)

        def selection_func(weights):
            return trans_mat @ weights[start_idx: start_idx + sel_len]

        return selection_func, None


def domain_intersection(first, second):
    """
    Calculate intersection(s) of two domains.

    Args:
        first (:py:class:`.Domain`): first domain
        second (:py:class:`.Domain`): second domain

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
        first (:py:class:`.Function`): first function
        second (:py:class:`.Function`): second function

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
        """
        Take the complex conjugate of the first element and multiply it
        by the second.
        """
        return np.conj(first(z)) * second(z)

    result, error = integrate_function(function, areas)
    return np.real_if_close(result)


def integrate_function(function, interval):
    """
    Integrates the given *function* over the *interval* using
    :func:`.complex_quadrature`.

    Args:
        function(callable): Function to integrate.
        interval(list of tuples): List of (start, end) values of the intervals
            to integrate on.

    Return:
        tuple: (Result of the Integration, errors that occurred during the
            integration).

    """
    result = 0
    err = 0
    for area in interval:
        res = complex_quadrature(function, area[0], area[1])
        result += res[0]
        err += res[1]

    return result, err


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

    return (real_integral[0] + 1j * imag_integral[0],
            real_integral[1] + imag_integral[1])


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

    # calculate output size and allocate output
    out = np.ones(first.shape, dtype=complex) * np.nan

    # TODO propagate vectorization into _dot_product_l2 to save this loop
    # loop over entries
    for idx, (f, s) in enumerate(zip(first, second)):
        out[idx] = _dot_product_l2(f, s)

    return np.real_if_close(out)


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
    return calculate_scalar_product_matrix(np.multiply,
                                           sanitize_input(values_a, Number),
                                           sanitize_input(values_b, Number))


def calculate_scalar_product_matrix(scalar_product_handle, base_a, base_b,
                                    optimize=False):
    r"""
    Calculates a matrix :math:`A` , whose elements are the scalar products of
    each element from *base_a* and *base_b*, so that
    :math:`a_{ij} = \langle \mathrm{a}_i\,,\: \mathrm{b}_j\rangle`.

    Args:
        scalar_product_handle (callable): function handle that is called to
            calculate the scalar product. This function has to be able to cope
            with (1d) vectorial input.
        base_a (:py:class:`.Base`): Basis a
        base_b (:py:class:`.Base`): Basis b
        optimize (bool): Switch to turn on the symmetry based speed up.
            For development purposes only.

    TODO:
        making use of the commutable scalar product could save time,
        run some test on this

    Return:
        numpy.ndarray: matrix :math:`A`
    """
    fractions_a = base_a.fractions
    fractions_b = base_b.fractions

    if optimize:
        raise NotImplementedError("this approach leads to wrong results atm.")
        # There are certain conditions that have to be satisfied to make use of a symmetrical product matrix
        # not all of these conditions are checked here and the implementation itself is not yet free from errors.

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
        i, j = np.mgrid[0:fractions_a.shape[0],
                        0:fractions_b.shape[0]]
        fractions_i = fractions_a[i]
        fractions_j = fractions_b[j]

        res = scalar_product_handle(fractions_i.flatten(),
                                    fractions_j.flatten())
        return res.reshape(fractions_i.shape)


def project_on_base(state, base):
    """
    Projects a *state* on a basis given by *base*.

    Args:
        state (array_like): List of functions to approximate.
        base (:py:class:`.Base`): Basis to project onto.

    Return:
        numpy.ndarray: Weight vector in the given *base*
    """
    if not isinstance(base, Base):
        raise TypeError("Only pyinduct.core.Base accepted as base")

    # compute <x(z, t), phi_i(z)> (vector)
    projections = calculate_scalar_product_matrix(dot_product_l2,
                                                  Base(state),
                                                  base).squeeze()

    # compute <phi_i(z), phi_j(z)> for 0 < i, j < n (matrix)
    scale_mat = calculate_scalar_product_matrix(dot_product_l2, base, base)

    return np.reshape(np.dot(np.linalg.inv(scale_mat), projections), (scale_mat.shape[0], ))


def project_on_bases(states, canonical_equations):
    """
    Convenience wrapper for :py:func:`.project_on_base`.
    Calculate the state, assuming it will be constituted by the dominant
    base of the respective system. The keys from the dictionaries
    *canonical_equations* and *states* must be the same.

    Args:
        states: Dictionary with a list of functions as values.
        canonical_equations: List of :py:class:`.CanonicalEquation` instances.

    Returns:
        numpy.array: Finit dimensional state as 1d-array corresponding to the
            concatenated dominant bases from *canonical_equations*.
    """
    q0 = np.array([])
    for ce in canonical_equations:
        lbl = ce.dominant_lbl
        q0 = np.hstack(tuple([q0] + [project_on_base(state, get_base(lbl))
                                     for state in states[ce.name]]))

    return q0


def back_project_from_base(weights, base):
    """
    Build evaluation handle for a distributed variable that was approximated
    as a set of *weights* om a certain *base*.

    Args:
        weights (numpy.ndarray): Weight vector.
        base (:py:class:`.Base`): Base to be used for the projection.

    Return:
        evaluation handle
    """
    if isinstance(weights, Number):
        weights = np.asarray([weights])
    if weights.shape[0] != base.fractions.shape[0]:
        raise ValueError("Lengths of weights and initial_initial_functions "
                         "do not match!")

    def eval_handle(z):
        res = sum([weights[i] * base.fractions[i](z)
                   for i in range(weights.shape[0])])
        return real(res)

    return eval_handle


def change_projection_base(src_weights, src_base, dst_base):
    """
    Converts given weights that form an approximation using *src_base*
    to the best possible fit using *dst_base*.

    Args:
        src_weights (numpy.ndarray): Vector of numbers.
        src_base (:py:class:`.Base`): The source Basis.
        dst_base (:py:class:`.Base`): The destination Basis.

    Return:
        :obj:`numpy.ndarray`: target weights
    """
    pro_mat = calculate_base_transformation_matrix(src_base, dst_base)
    return project_weights(pro_mat, src_weights)


def project_weights(projection_matrix, src_weights):
    """
    Project *src_weights* on new basis using the provided *projection_matrix*.

    Args:
        projection_matrix (:py:class:`numpy.ndarray`): projection between
            the source and the target basis;
            dimension (m, n)
        src_weights (:py:class:`numpy.ndarray`): weights in the source basis;
            dimension (1, m)

    Return:
        :py:class:`numpy.ndarray`: weights in the target basis;
            dimension (1, n)
    """
    src_weights = sanitize_input(src_weights, Number)
    return np.dot(projection_matrix, src_weights)


class TransformationInfo:
    """
    Structure that holds information about transformations between different
    bases.

    This class serves as an easy to use structure to aggregate information,
    describing transformations between different
    :py:class:`.BaseFraction` s. It can be tested for equality to check the
    equity of transformations and is hashable
    which makes it usable as dictionary key to cache different transformations.

    Attributes:
        src_lbl(str): label of source basis
        dst_lbl(str): label destination basis
        src_base(:obj:`numpy.ndarray`): source basis in form of an array of
            the source Fractions
        dst_base(:obj:`numpy.ndarray`): destination basis in form of an
            array of the destination Fractions
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
        return (self.src_lbl, self.dst_lbl, self.src_order, self.dst_order) == \
               (other.src_lbl, other.dst_lbl, other.src_order, other.dst_order)

    def mirror(self):
        """
        Factory method, that creates a new TransformationInfo object by
        mirroring *src* and *dst* terms.
        This helps handling requests to different bases.
        """
        new_info = TransformationInfo()
        new_info.src_lbl = self.dst_lbl
        new_info.src_base = self.dst_base
        new_info.src_order = self.src_order
        new_info.dst_lbl = self.dst_lbl
        new_info.dst_base = self.src_base
        new_info.dst_order = self.src_order
        return new_info


def get_weight_transformation(info):
    """
    Create a handle that will transform weights from *info.src_base* into
    weights for *info-dst_base* while paying respect to the given derivative
    orders.

    This is accomplished by recursively iterating through source and
    destination bases and evaluating their :attr:`transformation_hints`.

    Args:
        info(:py:class:`.TransformationInfo`): information about the requested
            transformation.

    Return:
        callable: transformation function handle
    """
    # TODO since this lives in core now, get rid of base labels
    # trivial case
    if info.src_lbl == info.dst_lbl:
        mat = calculate_expanded_base_transformation_matrix(
            info.src_base, info.dst_base,
            info.src_order, info.dst_order,
            True)

        def identity(weights):
            return np.dot(mat, weights)

        return identity

    # try to get help from the destination base
    handle, hint = info.dst_base.transformation_hint(info)
    if handle is None:
        # try source instead
        handle, hint = info.src_base.transformation_hint(info)

    if handle is None:
        raise TypeError(
            ("get_weight_transformation(): \n"
             "You requested information about how to transform to '{0}'({1}) \n"
             "from '{3}'({4}), furthermore the source derivative order is \n"
             "{2} and the target one is {5}. No transformation could be \n"
             "found, remember to implement your own 'transformation_hint' \n"
             "method for non-standard bases.").format(
                info.dst_lbl,
                info.dst_base.__class__.__name__,
                info.dst_order,
                info.src_lbl,
                info.src_base.__class__.__name__,
                info.src_order,
            ))

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
            new_info.dst_base = get_base(dep_lbl)
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


def get_transformation_info(source_label, destination_label,
                            source_order, destination_order):
    """
    Provide the weights transformation from one/source base to
    another/destination base.

    Args:
        source_label (str): Label from the source base.
        destination_label (str): Label from the destination base.
        source_order: Order from the available time derivative
            of the source weights.
        destination_order: Order from the desired time derivative
            of the destination weights.

    Returns:
        :py:class:`.TransformationInfo`: Transformation info object.

    """
    info = TransformationInfo()
    info.src_lbl = source_label
    info.src_base = get_base(info.src_lbl)
    info.src_order = source_order
    info.dst_lbl = destination_label
    info.dst_base = get_base(info.dst_lbl)
    info.dst_order = destination_order

    return info


def calculate_expanded_base_transformation_matrix(src_base, dst_base, src_order, dst_order, use_eye=False):
    r"""
    Constructs a transformation matrix :math:`\bar V` from basis given by
    *src_base* to basis given by *dst_base* that also transforms all temporal
    derivatives of the given weights.

    See:
        :py:func:`.calculate_base_transformation_matrix` for further details.

    Args:
        dst_base (:py:class:`.Base`): New projection base.
        src_base (:py:class:`.Base`): Current projection base.
        src_order: Temporal derivative order available in *src_base*.
        dst_order: Temporal derivative order needed in *dst_base*.
        use_eye (bool): Use identity as base transformation matrix.
            (For easy selection of derivatives in the same base)

    Raises:
        ValueError: If destination needs a higher derivative order than source
            can provide.

    Return:
        :obj:`numpy.ndarray`: Transformation matrix
    """
    if src_order < dst_order:
        raise ValueError(("higher 'dst_order'({0}) demanded than "
                          + "'src_order'({1}) can provide for this strategy.").format(dst_order, src_order))

    # build core transformation
    if use_eye:
        core_transformation = np.eye(src_base.fractions.size)
    else:
        core_transformation = calculate_base_transformation_matrix(src_base, dst_base)

    # build block matrix
    part_transformation = block_diag(*[core_transformation for i in range(dst_order + 1)])
    complete_transformation = np.hstack([part_transformation]
                                        + [np.zeros((part_transformation.shape[0], src_base.fractions.size))
                                           for i in range(src_order - dst_order)])
    return complete_transformation


def calculate_base_transformation_matrix(src_base, dst_base):
    """
    Calculates the transformation matrix :math:`V` , so that the a
    set of weights, describing a function in the
    *src_base* will express the same function in the *dst_base*, while
    minimizing the reprojection error.
    An quadratic error is used as the error-norm for this case.

    Warning:
        This method assumes that all members of the given bases have
        the same type and that their
        :py:class:`.BaseFraction` s, define compatible scalar products.

    Raises:
        TypeError: If given bases do not provide an
            :py:func:`.scalar_product_hint` method.

    Args:
        src_base (:py:class:`.Base`): Current projection base.
        dst_base (:py:class:`.Base`): New projection base.

    Return:
        :py:class:`numpy.ndarray`: Transformation matrix :math:`V` .
    """
    # compute P and Q matrices, where P = Sum(P_n) and Q = Sum(Q_n)
    s_hints = src_base.scalar_product_hint()
    d_hints = dst_base.scalar_product_hint()

    p_matrices = []
    q_matrices = []
    for idx, (s_hint, d_hint) in enumerate(zip(s_hints, d_hints)):
        dst_members = Base([dst_frac.get_member(idx)
                            for dst_frac in dst_base.fractions])
        src_members = Base([src_frac.get_member(idx)
                            for src_frac in src_base.fractions])

        # compute P_n matrix:
        # <phi_tilde_ni(z), phi_dash_nj(z)> for 0 < i < N, 0 < j < M
        p_matrices.append(calculate_scalar_product_matrix(s_hint,
                                                          dst_members,
                                                          src_members))

        # compute Q_n matrix:
        # <phi_dash_ni(z), phi_dash_nj(z)> for 0 < i < M, 0 < j < M
        q_matrices.append(calculate_scalar_product_matrix(d_hint,
                                                          dst_members,
                                                          dst_members))

    p_mat = np.sum(p_matrices, axis=0)
    q_mat = np.sum(q_matrices, axis=0)

    # compute V matrix, where V = inv(Q)*P
    v_mat = np.dot(np.linalg.inv(q_mat), p_mat)
    return v_mat


def normalize_base(b1, b2=None):
    r"""
    Takes two :py:class:`.Base`'s :math:`\boldsymbol{b}_1` ,
    :math:`\boldsymbol{b}_1` and normalizes them so that
    :math:`\langle\boldsymbol{b}_{1i}\,
    ,\:\boldsymbol{b}_{2i}\rangle = 1`.
    If only one base is given, :math:`\boldsymbol{b}_2`
    defaults to :math:`\boldsymbol{b}_1`.

    Args:
        b1 (:py:class:`.Base`): :math:`\boldsymbol{b}_1`
        b2 (:py:class:`.Base`): :math:`\boldsymbol{b}_2`

    Raises:
        ValueError: If :math:`\boldsymbol{b}_1`
            and :math:`\boldsymbol{b}_2` are orthogonal.

    Return:
        :py:class:`.Base` : if *b2* is None,
           otherwise: Tuple of 2 :py:class:`.Base`'s.
    """
    auto_normalization = False
    if b2 is None:
        auto_normalization = True

    res = generic_scalar_product(b1, b2)

    if any(res < np.finfo(float).eps):
        if any(np.isclose(res, 0)):
            raise ValueError("given base fractions are orthogonal. "
                             "no normalization possible.")
        else:
            raise ValueError("imaginary scale required. "
                             "no normalization possible.")

    scale_factors = np.sqrt(1 / res)
    b1_scaled = b1.__class__(
        [frac.scale(factor)
         for frac, factor in zip(b1.fractions, scale_factors)])

    if auto_normalization:
        return b1_scaled
    else:
        b2_scaled = b2.__class__(
            [frac.scale(factor)
             for frac, factor in zip(b2.fractions, scale_factors)])
        return b1_scaled, b2_scaled


def generic_scalar_product(b1, b2=None):
    """
    Calculates the pairwise scalar product between the elements
    of the :py:class:`.Base` *b1* and *b2*.

    Args:
        b1 (:py:class:`.Base`): first basis
        b2 (:py:class:`.Base`): second basis, if omitted
            defaults to *b1*

    Note:
        If *b2* is omitted, the result can be used to normalize
        *b1* in terms of its scalar product.
    """
    if b2 is None:
        b2 = b1

    if type(b1) != type(b2):
        raise TypeError("only arguments of same type allowed.")

    hints = b1.scalar_product_hint()
    res = np.zeros(b1.fractions.shape, dtype=complex)
    for idx, hint in enumerate(hints):
        members_1 = np.array([fraction.get_member(idx)
                              for fraction in b1.fractions])
        members_2 = np.array([fraction.get_member(idx)
                              for fraction in b2.fractions])
        res += hint(members_1, members_2)
    return np.real_if_close(res)


def find_roots(function, grid, n_roots=None, rtol=1.e-5, atol=1.e-8,
               cmplx=False, sort_mode="norm"):
    r"""
    Searches *n_roots* roots of the *function* :math:`f(\boldsymbol{x})`
    on the given *grid* and checks them for uniqueness with aid of *rtol*.

    In Detail :py:func:`scipy.optimize.root` is used to find initial candidates
    for roots of :math:`f(\boldsymbol{x})` . If a root satisfies the criteria
    given by atol and rtol it is added. If it is already in the list,
    a comprehension between the already present entries' error and the
    current error is performed. If the newly calculated root comes
    with a smaller error it supersedes the present entry.

    Raises:
        ValueError: If the demanded amount of roots can't be found.

    Args:
        function (callable): Function handle for math:`f(\boldsymbol{x})`
            whose roots shall be found.
        grid (list): Grid to use as starting point for root detection.
            The :math:`i` th element of this list provides sample points
            for the :math:`i` th parameter of :math:`\boldsymbol{x}` .
        n_roots (int): Number of roots to find. If none is given, return
            all roots that could be found in the given area.
        rtol: Tolerance to be exceeded for the difference of two roots
            to be unique: :math:`f(r1) - f(r2) > \textrm{rtol}` .
        atol: Absolute tolerance to zero: :math:`f(x^0) < \textrm{atol}` .
        cmplx(bool): Set to True if the given *function* is complex valued.
        sort_mode(str): Specify tho order in which the extracted roots shall be
            sorted. Default "norm" sorts entries by their :math:`l_2` norm,
            while "component" will sort them in increasing order by every
            component.

    Return:
        numpy.ndarray of roots; sorted in the order they are returned by
        :math:`f(\boldsymbol{x})` .
    """
    if isinstance(grid[0], Number):
        grid = [grid]

    dim = len(grid)
    if cmplx:
        assert dim == 2
        function = complex_wrapper(function)

    roots = []
    errors = []

    grids = np.meshgrid(*[row for row in grid])
    values = np.vstack([arr.flatten() for arr in grids]).T

    # iterate over test_values
    val = iter(values)
    while True:
        try:
            res = root(function, next(val), tol=atol)
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
            if ar.min() - atol > rt or ar.max() + atol < rt:
                abort = True
                break
        if abort:
            continue

        if roots:
            # check whether root is already present in cache
            cmp_arr = np.isclose(calculated_root, roots, atol=rtol)
            cmp_vec = [all(elements) for elements in cmp_arr]
            if any(cmp_vec):
                idx = cmp_vec.index(True)
                if errors[idx] > error:
                    roots[idx] = calculated_root
                    errors[idx] = error
                    # TODO check jacobian (if provided)
                    # to identify roots of higher order
                continue

        roots.append(calculated_root)
        errors.append(error)

    if n_roots is None:
        n_roots = len(roots)

    if len(roots) < n_roots:
        raise ValueError("Insufficient number of roots detected. ({0} < {1}) "
                         "Try to increase the area to search in.".format(
                            len(roots), n_roots))

    valid_roots = np.array(roots)

    if len(valid_roots) == 0:
        return list()

    # sort roots
    if sort_mode == "norm":
        # sort entries by their norm
        idx = np.argsort(np.linalg.norm(valid_roots, axis=1))
        sorted_roots = valid_roots[idx, :]

    elif sort_mode == "component":
        # completely sort first column before we start
        idx = np.argsort(valid_roots[:, 0])
        sorted_roots = valid_roots[idx, :]

        for layer in range(valid_roots.shape[1] - 1):
            for rt in sorted_roots[:, layer]:
                eq_mask = np.isclose(sorted_roots[:, layer], rt, rtol=rtol)
                idx = np.argsort(sorted_roots[eq_mask, layer + 1])
                sorted_roots[eq_mask] = sorted_roots[eq_mask][idx, :]
    else:
        raise ValueError("Sort mode: {} not supported.".format(sort_mode))

    good_roots = sorted_roots[:n_roots]

    if cmplx:
        return good_roots[:, 0] + 1j * good_roots[:, 1]

    if dim == 1:
        return good_roots.flatten()

    return good_roots


def complex_wrapper(func):
    """
    Wraps complex valued functions into two-dimensional functions.
    This enables the root-finding routine to handle it as a
    vectorial function.

    Args:
        func (callable): Callable that returns a complex result.

    Return:
        two-dimensional, callable: function handle,
            taking x = (re(x), im(x) and returning [re(func(x), im(func(x)].
    """

    def wrapper(x):
        return np.array([np.real(func(np.complex(x[0], x[1]))),
                         np.imag(func(np.complex(x[0], x[1])))])

    return wrapper


class Parameters:
    """
    Handy class to collect system parameters.
    This class can be instantiated with a dict, whose keys will the
    become attributes of the object.
    (Bunch approach)

    Args:
        kwargs: parameters
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Domain(object):
    """
    Helper class that manages ranges for data evaluation, containing
    parameters.

    Args:
        bounds (tuple): Interval bounds.
        num (int): Number of points in interval.
        step (numbers.Number): Distance between points (if homogeneous).
        points (array_like): Points themselves.

    Note:
        If num and step are given, num will take precedence.
    """

    def __init__(self, bounds=None, num=None, step=None, points=None):
        if points is not None:
            # points are given, easy one
            self._values = np.atleast_1d(points)
            self._limits = (points.min(), points.max())
            self._num = points.size
            # TODO check for evenly spaced entries
            # for now just use provided information
            self._step = step
        elif bounds and num:
            self._limits = bounds
            self._num = num
            self._values, self._step = np.linspace(bounds[0],
                                                   bounds[1],
                                                   num,
                                                   retstep=True)
            if step is not None and not np.isclose(self._step, step):
                raise ValueError("could not satisfy both redundant "
                                 "requirements for num and step!")
        elif bounds and step:
            self._limits = bounds
            # calculate number of needed points but save correct step size
            self._num = int((bounds[1] - bounds[0]) / step + 1.5)
            self._values, self._step = np.linspace(bounds[0],
                                                   bounds[1],
                                                   self._num,
                                                   retstep=True)
            if np.abs(step - self._step) > 1e-1:
                warnings.warn("desired step-size {} doesn't fit to given "
                              "interval, changing to {}".format(step,
                                                                self._step))
        else:
            raise ValueError("not enough arguments provided!")

    def __len__(self):
        return len(self._values)

    def __getitem__(self, item):
        return self._values[item]

    @property
    def step(self):
        return self._step

    @property
    def bounds(self):
        return self._limits

    @property
    def points(self):
        return self._values


def real(data):
    """
    Check if the imaginary part of :code:`data` vanishes
    and return its real part if it does.

    Args:
        data (numbers.Number or array_like): Possibly complex data to check.

    Raises:
        ValueError: If provided data can't be converted within
         the given tolerance limit.

    Return:
        numbers.Number or array_like: Real part of :code:`data`.
    """
    candidates = np.real_if_close(data, tol=100)

    if candidates.dtype == 'complex':
        raise ValueError("Imaginary part does not vanish, "
                         + "check for implementation errors.")

    # TODO make numpy array to common data type (even for scalar values)
    if candidates.size == 1:
        return float(candidates)

    return candidates


class EvalData:
    """
    Convenience wrapper for function evaluation.
    Contains the input data that was used for evaluation and the results.
    """

    def __init__(self, input_data, output_data, name=""):
        # check type and dimensions
        assert isinstance(input_data, list)
        assert isinstance(output_data, np.ndarray)

        # output_data has to contain at least len(input_data) dimensions
        assert len(input_data) <= len(output_data.shape)

        for dim in range(len(input_data)):
            assert len(input_data[dim]) == output_data.shape[dim]

        self.input_data = input_data
        if output_data.size == 0:
            raise ValueError("No initialisation possible with an empty array!")
        self.output_data = output_data
        self.min = output_data.min()
        self.max = output_data.max()
        self.name = name
