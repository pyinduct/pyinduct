"""
In the Core module you can find all basic classes and functions which form the backbone of the toolbox.
"""
import warnings
import numbers

import numpy as np
import numpy.ma as ma
import collections
from copy import copy, deepcopy
from numbers import Number

from scipy import integrate
from scipy.linalg import block_diag
from scipy.optimize import root
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline, RegularGridInterpolator

from .registry import get_base

__all__ = ["Domain", "EvalData", "Parameters",
           "Base", "BaseFraction", "StackedBase",
           "Function", "ConstantFunction", "ComposedFunctionVector",
           "find_roots", "sanitize_input", "real", "dot_product_l2",
           "normalize_base", "project_on_base", "change_projection_base",
           "back_project_from_base",
           "calculate_scalar_product_matrix",
           "calculate_base_transformation_matrix",
           "calculate_expanded_base_transformation_matrix",
           ]


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
        if not isinstance(obj.item(), allowed_type):
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

        Note:
            Overwrite to implement custom functionality.
            For an example implementation see :py:class:`.Function`
        """
        pass

    def function_space_hint(self):
        """
        Empty Hint that can return properties which uniquely define
        the function space of the :py:class:`.BaseFraction`.

        Note:
            Overwrite to implement custom functionality.
            For an example implementation see :py:class:`.Function`.
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

    def __call__(self, *args, **kwargs):
        """
        Spatial evaluation of the base fraction.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:

        """
        raise NotImplementedError("This is an empty function."
                                  " Overwrite it in your implementation to use this functionality.")

    def add_neutral_element(self):
        """
        Return the neutral element of addition for this object.

        In other words: `self + ret_val == self`.
        """
        raise NotImplementedError()

    def mul_neutral_element(self):
        """
        Return the neutral element of multiplication for this object.

        In other words: `self * ret_val == self`.
        """
        raise NotImplementedError()

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

    def _apply_operator(self, operator, additive=False):
        """
        Return a new base fraction with the given operator applied.

        Args:
            operator: Object that can be applied to the base fraction.
            additive: Define if the given operator is additive. Default: False.
                For an additive operator G and two base fractions f, h the
                relation G(f + h) = G(f) + G(h) holds. If the operator is
                not additive the derivatives will be discarded.
        """
        raise NotImplementedError()

    def real(self):
        """
        Return the real part of the base fraction.
        """
        return self._apply_operator(np.real, additive=True)

    def imag(self):
        """
        Return the imaginary port of the base fraction.
        """
        return self._apply_operator(np.imag, additive=True)

    def conj(self):
        """
        Return the complex conjugated base fraction.
        """
        return self._apply_operator(np.conj, additive=True)


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

    Args:
        eval_handle (callable): Callable object that can be evaluated.
        domain((list of) tuples): Domain on which the eval_handle is defined.
        nonzero(tuple): Region in which the eval_handle will return
        nonzero output. Must be a subset of *domain*
        derivative_handles (list): List of callable(s) that contain
        derivatives of eval_handle
    """

    # TODO: overload add and mul operators

    def __init__(self, eval_handle, domain=(-np.inf, np.inf), nonzero=(-np.inf, np.inf), derivative_handles=None):
        super().__init__(None)
        self._vectorial = False
        self._function_handle = None
        self._derivative_handles = None

        self.domain = set()
        self.nonzero = set()
        for kw, val in zip(["domain", "nonzero"], [domain, nonzero]):
            if not isinstance(val, set):
                if isinstance(val, tuple):
                    val = {val}
                else:
                    raise TypeError("(Set of) or tuple(s) has to be provided "
                                    "for {0}".format(kw))

            setattr(self, kw, domain_simplification(val))

        self.function_handle = eval_handle
        self.derivative_handles = derivative_handles

    @property
    def derivative_handles(self):
        return self._derivative_handles

    @derivative_handles.setter
    def derivative_handles(self, eval_handle_derivatives):
        if eval_handle_derivatives is None:
            eval_handle_derivatives = []
        if not isinstance(eval_handle_derivatives, collections.abc.Iterable):
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
        test_value = next(iter(self.domain))[1]
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
        values = np.atleast_1d(values)
        if values.dtype == complex:
            raise TypeError("Only real valued arguments considered for "
                            "pyinduct function. \nProvide value: {}\n"
                            "Data type: {}".format(values, values.dtype))

        mask = np.full(len(values), False)
        for interval in self.domain:
            d_mask = np.logical_and(values >= interval[0],
                                    values <= interval[1])
            np.logical_or(mask, d_mask, out=mask)

        if not all(mask):
            raise ValueError("Function evaluated outside it's domain {} with {}"
                             "".format(self.domain,
                                       values[np.logical_not(mask)]))

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

    def scalar_product_hint(self):
        """
        Return the hint that the :py:func:`._dot_product_l2` has to
        calculated to gain the scalar product.
        """
        return dot_product_l2

    def function_space_hint(self):
        """
        Return the hint that this function is an element of the
        an scalar product space which is uniquely defined by
        the scalar product :py:meth:`.scalar_product_hint`.

        Note:
            If you are working on different function spaces, you have
            to overwrite this hint in order to provide more properties
            which characterize your specific function space. For
            example the domain of the functions.
        """
        return self.scalar_product_hint(), self.domain

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

    def add_neutral_element(self):
        return ConstantFunction(0, domain=self.domain)

    def mul_neutral_element(self):
        return ConstantFunction(1, domain=self.domain)

    def _apply_operator(self, operator, additive=False):
        """
        Return a new function with the given operator applied.
        See docstring of :py:meth:`.BaseFraction._apply_operator`.
        """
        def apply(func):
            def handle(z):
                return operator(func(z))
            return handle

        new_obj = deepcopy(self)
        new_obj.function_handle = apply(self.function_handle)
        if additive:
            new_obj.derivative_handles = [
                apply(f) for f in self.derivative_handles]
        else:
            new_obj.derivative_handles = None

        return new_obj


class ConstantFunction(Function):
    """
    A :py:class:`.Function` that returns a constant value.

    This function can be differentiated without limits.

    Args:
        constant (number): value to return

    Keyword Args:
        **kwargs: All other kwargs get passed to :py:class:`.Function`.

    """

    def __init__(self, constant, **kwargs):
        self._constant = constant

        func_kwargs = dict(eval_handle=self._constant_function_handle)
        if "nonzero" in kwargs:
            if constant == 0:
                if kwargs["nonzero"] != set():
                    raise ValueError("Constant Function with constant 0 must have an"
                                     " empty set nonzero area.")
            if "domain" in kwargs:
                if kwargs["nonzero"] != kwargs["domain"]:
                    raise ValueError(
                        "Constant Function is expected to be constant on the complete "
                        "domain. Nonzero argument is prohibited")
            else:
                func_kwargs["domain"] = kwargs["nonzero"]
            func_kwargs["nonzero"] = kwargs["nonzero"]
        else:
            if "domain" in kwargs:
                func_kwargs["domain"] = kwargs["domain"]
                func_kwargs["nonzero"] = kwargs["domain"]
            if constant == 0:
                func_kwargs["nonzero"] = set()

        if "derivative_handles" in kwargs:
            warnings.warn(
                "Derivative handles passed to ConstantFunction are discarded")

        super().__init__( **func_kwargs)

    def _constant_function_handle(self, z):
        return self._constant * np.ones_like(z)

    def derive(self, order=1):
        if not isinstance(order, int):
            raise TypeError("only integer allowed as derivation order")

        if order == 0:
            return self

        if order < 0:
            raise ValueError("only derivative order >= 0 supported")

        zero_func = deepcopy(self)
        zero_func._constant = 0
        zero_func.nonzero = set()

        return zero_func


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

    def __call__(self, arguments):
        f_res = np.array([f(arguments) for f in self.members["funcs"]])
        s_res = self.members["scalars"]
        if f_res.ndim > 1:
            s_res = s_res[:, None] * np.ones_like(f_res)
        res = np.concatenate((f_res, s_res))
        return res

    def scalar_product_hint(self):
        func_hints = [f.scalar_product_hint() for f in self.members["funcs"]]
        scalar_hints = [dot_product for s in self.members["scalars"]]
        return func_hints + scalar_hints

    def function_space_hint(self):
        """
        Return the hint that this function is an element of the
        an scalar product space which is uniquely defined by

            * the scalar product
              :py:meth:`.ComposedFunctionVector.scalar_product`
            * :code:`len(self.members["funcs"])` functions
            * and :code:`len(self.members["scalars"])` scalars.
        """
        func_hints = [f.function_space_hint() for f in self.members["funcs"]]
        scalar_hints = [dot_product for s in self.members["scalars"]]
        return func_hints + scalar_hints

    def get_member(self, idx):
        if idx < len(self.members["funcs"]):
            return self.members["funcs"][idx]
        elif idx - len(self.members["funcs"]) < len(self.members["scalars"]):
            return self.members["scalars"][idx - len(self.members["funcs"])]
        else:
            raise ValueError("wrong index")

    def scale(self, factor):
        if isinstance(factor, ComposedFunctionVector):
            if not len(self.members["funcs"]) == len(factor.members["funcs"]):
                raise ValueError

            if not len(self.members["scalars"]) == len(factor.members["scalars"]):
                raise ValueError

            return self.__class__(np.array(
                [func.scale(scale) for func, scale in
                 zip(self.members["funcs"], factor.members["funcs"])]),
                [scalar * scale for scalar, scale in
                 zip(self.members["scalars"], factor.members["scalars"])],
            )

        elif isinstance(factor, Number):
            return self.__class__(
                np.array([func.scale(factor) for func in self.members["funcs"]]),
                np.array([scal * factor for scal in self.members["scalars"]])
            )

        else:
            raise TypeError("ComposedFunctionVector can only be scaled with "
                            "compatible ComposedFunctionVector of with a"
                            "constant scalar")

    def mul_neutral_element(self):
        """
        Create neutral element of multiplication that is compatible to this
        object.

        Returns: Comp. Function Vector with constant functions returning 1 and
            scalars of value 1.

        """
        funcs = [f.mul_neutral_element() for f in self.members["funcs"]]
        scalar_constants = [1 for f in self.members["scalars"]]
        neut = ComposedFunctionVector(funcs, scalar_constants)
        return neut

    def add_neutral_element(self):
        """
        Create neutral element of addition that is compatible to this
        object.

        Returns: Comp. Function Vector with constant functions returning 0 and
            scalars of value 0.

        """
        funcs = [f.add_neutral_element() for f in self.members["funcs"]]
        scalar_constants = [0 for f in self.members["scalars"]]
        neut = ComposedFunctionVector(funcs, scalar_constants)
        return neut

    def _apply_operator(self, operator, additive=False):
        """
        Return a new composed function vector with the given operator applied.
        See docstring of :py:meth:`.BaseFraction._apply_operator`.
        """
        funcs = [f._apply_operator(operator, additive=additive)
                 for f in self.members["funcs"]]
        scalar_constants = [operator(s) for s in self.members["scalars"]]
        new_obj = ComposedFunctionVector(funcs, scalar_constants)
        return new_obj


class ConstantComposedFunctionVector(ComposedFunctionVector):
    r"""
    Constant composite function vector :math:`\boldsymbol{x}`.

    .. math::
        \boldsymbol{x} = \begin{pmatrix}
            z \mapsto x_1(z) = c_1 \\
            \vdots \\
            z \mapsto x_n(z) = c_n \\
            d_1 \\
            \vdots \\
            c_n \\
        \end{pmatrix}


    Args:
        func_constants (array-like): Constants for the functions.
        scalar_constants (array-like): The scalar constants.
        **func_kwargs: Keyword args that are passed to the ConstantFunction.
    """

    def __init__(self, func_constants, scalar_constants, **func_kwargs):
        func_constants = sanitize_input(func_constants, Number)
        scalars = sanitize_input(scalar_constants, Number)

        funcs = [ConstantFunction(c, **func_kwargs) for c in func_constants]
        super().__init__(funcs, scalars)


class ApproximationBasis:
    """
    Base class for an approximation basis.

    An approximation basis is formed by some objects on which given distributed
    variables may be projected.
    """
    def scalar_product_hint(self):
        """
        Hint that returns steps for scalar product calculation with elements of
        this base.

        Note:
            Overwrite to implement custom functionality.
        """
        raise NotImplementedError()

    def function_space_hint(self):
        """
        Hint that returns properties that characterize the functional
        space of the fractions.
        It can be used to determine if function spaces match.

        Note:
            Overwrite to implement custom functionality.
        """
        raise NotImplementedError()

    def is_compatible_to(self, other):
        """
        Helper functions that checks compatibility between two approximation
        bases.

        In this case compatibility is given if the two bases live in the same
        function space.

        Args:
             other (:py:class:`.Approximation Base`): Approximation basis to
                compare with.

        Returns: True if bases match, False if they do not.

        """
        return self.function_space_hint() == other.function_space_hint()


class Base(ApproximationBasis):
    """
    Base class for approximation bases.

    In general, a :py:class:`.Base` is formed by a certain amount of
    :py:class:`.BaseFractions` and therefore forms finite-dimensional subspace
    of the distributed problem's domain. Most of the time, the user does not
    need to interact with this class.

    Args:
        fractions (iterable of :py:class:`.BaseFraction`): List, array or
            dict of :py:class:`.BaseFraction`'s
        matching_base_lbls (list of str): List of labels from exactly matching
            bases, for which no transformation is necessary.
            Useful for transformations from bases that 'live' in
            different function spaces but evolve with the same time
            dynamic/coefficients (e.g. modal bases).
        intermediate_base_lbls (list of str): If it is certain that this base
            instance will be asked (as destination base) to return a
            transformation to a source base, whose implementation is
            cumbersome, its label can be provided here. This will trigger the
            generation of the transformation using build-in features.
            The algorithm, implemented in :py:class:`.get_weights_transformation`
            is then called again with the intermediate base as destination base
            and the 'old' source base. With this technique arbitrary long
            transformation chains are possible, if the provided intermediate
            bases again define intermediate bases.
    """
    def __init__(self, fractions,
                 matching_base_lbls=None, intermediate_base_lbls=None):
        fractions = sanitize_input(fractions, BaseFraction)

        # check type
        base_space = fractions[0].function_space_hint()
        if not all([frac.function_space_hint() == base_space
                    for frac in fractions]):
            raise ValueError("Provided fractions must be compatible!")

        self.fractions = fractions
        self.matching_base_lbls = matching_base_lbls
        if self.matching_base_lbls is None:
            self.matching_base_lbls = []
        if isinstance(self.matching_base_lbls, str):
            self.matching_base_lbls = [self.matching_base_lbls]

        self.intermediate_base_lbls = intermediate_base_lbls
        if self.intermediate_base_lbls is None:
            self.intermediate_base_lbls = []
        if isinstance(self.intermediate_base_lbls, str):
            self.intermediate_base_lbls = [self.intermediate_base_lbls]

    def __iter__(self):
        return iter(self.fractions)

    def __len__(self):
        return len(self.fractions)

    def __getitem__(self, item):
        return self.fractions[item]

    @staticmethod
    def _transformation_factory(info, equivalent=False):
        mat = calculate_expanded_base_transformation_matrix(info.src_base,
                                                            info.dst_base,
                                                            info.src_order,
                                                            info.dst_order,
                                                            use_eye=equivalent)

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
        if info.src_lbl == info.dst_lbl:
            # trivial case
            return self._transformation_factory(info, equivalent=True), None

        # check for matching bases
        match_cond_src = (self is info.src_base
                          and info.dst_lbl in self.matching_base_lbls)
        match_cond_dst = (self is info.dst_base
                          and info.src_lbl in self.matching_base_lbls)
        if match_cond_src or match_cond_dst:
            # bases are a match
            if len(info.dst_base) != len(info.src_base):
                msg = "Base length mismatch: len({})={} != len({})={}"
                raise ValueError(msg.format(info.src_lbl, len(info.src_base),
                                            info.dst_lbl, len(info.dst_base)))

            if info.src_order >= info.dst_order:
                # forward weights
                return self._transformation_factory(info, True), None

        # check for compatible base
        compat_cond_src = (self is info.src_base
                           and self.is_compatible_to(info.dst_base))
        compat_cond_dst = (self is info.dst_base
                           and self.is_compatible_to(info.src_base))
        if compat_cond_src or compat_cond_dst:
            # bases are compatible, use standard approach
            return self._transformation_factory(info), None

        if self.intermediate_base_lbls is not None:
            # try intermediate bases
            for inter_lbl in self.intermediate_base_lbls:
                trafo, hint = self._get_intermediate_transform(info, inter_lbl)
                if trafo is not None:
                    return trafo, hint

        # No Idea what to do.
        return None, None

    def _get_intermediate_transform(self, info, inter_lbl):
        if self is info.src_base:
            # build trafo from us to middleman
            intermediate_info = get_transformation_info(
                info.src_lbl, inter_lbl,
                info.src_order, info.src_order
            )
            handle = get_weight_transformation(intermediate_info)
            if info.dst_lbl == inter_lbl:
                # middleman is the source -> we are finished
                return handle, None

            # create hint from middleman to dst
            hint = get_transformation_info(
                inter_lbl, info.dst_lbl,
                info.src_order, info.dst_order
            )
            return handle, hint
        if self is info.dst_base:
            # build trafo from middleman to us
            intermediate_info = get_transformation_info(
                inter_lbl, info.dst_lbl,
                info.src_order, info.dst_order
            )
            handle = get_weight_transformation(intermediate_info)
            if info.src_lbl == inter_lbl:
                # middleman is the source -> we are finished
                return handle, None

            # create hint from src to middleman
            hint = get_transformation_info(
                info.src_lbl, inter_lbl,
                info.src_order, info.src_order
            )
            return handle, hint

        # No Idea what to do.
        return None, None

    def scalar_product_hint(self):
        """
        Hint that returns steps for scalar product calculation with elements of
        this base.

        Note:
            Overwrite to implement custom functionality.
        """
        return self.fractions[0].scalar_product_hint()

    def function_space_hint(self):
        """
        Hint that returns properties that characterize the functional
        space of the fractions.
        It can be used to determine if function spaces match.

        Note:
            Overwrite to implement custom functionality.
        """
        return self.fractions[0].function_space_hint()

    def derive(self, order):
        """
        Basic implementation of derive function.
        Empty implementation, overwrite to use this functionality.

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
            raise ValueError("This funcionality is deprecated.")

    def get_attribute(self, attr):
        """
        Retrieve an attribute from the fractions of the base.

        Args:
            attr(str): Attribute to query the fractions for.

        Returns:
            :py:class:`np.ndarray`: Array of ``len(fractions)`` holding the
            attributes. With `None` entries if the attribute is missing.

        """
        return np.array([getattr(frac, attr, None) for frac in self.fractions])


class StackedBase(ApproximationBasis):
    """
    Implementation of a basis vector that is obtained by stacking different
    bases onto each other. This typically occurs when the bases of coupled
    systems are joined to create a unified system.

    Args:
        base_info (OrderedDict): Dictionary with `base_label` as keys and
            dictionaries holding information about the bases as values.
            In detail, these Information must contain:

                - sys_name (str): Name of the system the base is associated with.
                - order (int): Highest temporal derivative order with which the
                    base shall be represented in the stacked base.
                - base (:py:class:`.ApproximationBase`): The actual basis.
    """

    def __init__(self, base_info):
        self.base_lbls = []
        self.system_names = []
        self.orders = []

        self._bases = []
        self._cum_frac_idxs = [0]
        self._cum_weight_idxs = [0]
        for lbl, info in base_info.items():
            # public properties
            self.base_lbls.append(lbl)
            self.system_names.append(info["sys_name"])
            order = info["order"]
            self.orders.append(order)
            base = info["base"]

            # internal properties
            self._bases.append(base)
            self._cum_frac_idxs.append(self._cum_frac_idxs[-1] + len(base))
            self._cum_weight_idxs.append(self._cum_weight_idxs[-1]
                                         + (order + 1) * len(base))

        self.fractions = np.concatenate([b.fractions for b in self._bases])
        self._size = self._cum_frac_idxs.pop(-1)
        self._weight_size = self._cum_weight_idxs.pop(-1)

    def scalar_product_hint(self):
        return NotImplemented

    def function_space_hint(self):
        return hash(self)

    def is_compatible_to(self, other):
        return False

    def scale(self, factor):
        raise NotImplementedError("Stacked base should not be scaled.")

    def transformation_hint(self, info):
        """
        If *info.src_lbl* is a member, just return it, using to correct
        derivative transformation, otherwise return `None`

        Args:
            info (:py:class:`.TransformationInfo`): Information about the
                requested transformation.
        Return:
            transformation handle
        """
        if info.src_order != 0:
            # this can be implemented but is not really meaningful
            return None, None

        # we only know how to get from a stacked base to one of our parts
        if info.src_base != self:
            return None, None
        if info.dst_lbl not in self.base_lbls:
            return None, None

        # check maximum available derivative order
        dst_idx = self.base_lbls.index(info.dst_lbl)
        init_src_ord = self.orders[dst_idx]
        if info.dst_order > init_src_ord:
            return None, None

        # get transform
        trans_mat = calculate_expanded_base_transformation_matrix(
            info.dst_base,
            info.dst_base,
            init_src_ord,
            info.dst_order,
            use_eye=True)

        start_idx = self._cum_weight_idxs[dst_idx]
        length = (init_src_ord + 1) * len(self._bases[dst_idx])

        def selection_func(weights):
            assert len(weights) == self._weight_size
            return trans_mat @ weights[start_idx: start_idx + length]

        return selection_func, None


def domain_simplification(domain):
    """
    Simplify a domain, given by possibly overlapping subdomains.

    Args:
        domain (set): Set of tuples, defining the (start, end) points of the
            subdomains.

    Returns:
        list: Simplified domain.
    """
    new_dom = set()
    temp_dom = list()

    # sort sub domains
    for idx, sub_dom in enumerate(domain):
        if sub_dom[0] > sub_dom[1]:
            temp_dom.append(sub_dom[::-1])
        else:
            temp_dom.append(sub_dom)

    # look for overlapping sub domains
    for s_idx, start_dom in enumerate(temp_dom):
        candidates = []
        for e_idx, end_dom in enumerate(temp_dom):
            if s_idx == e_idx:
                continue

            if start_dom[0] > end_dom[0]:
                # second one starts earlier, postpone
                continue

            if start_dom[1] > end_dom[0]:
                # two domains overlap
                candidates.append(e_idx)

        if not candidates:
            continue

        greatest_idx = candidates[np.argmax([temp_dom[idx][1]
                                             for idx in candidates])]
        if start_dom[1] >= temp_dom[greatest_idx][1]:
            # the second domain is a real sub set of the first one
            # save only the first
            new_dom.add(start_dom)
        else:
            # second one goes further -> join them
            new_dom.add((start_dom[0], temp_dom[greatest_idx][1]))

    if new_dom and new_dom != domain:
        return domain_simplification(new_dom)
    else:
        return set(temp_dom)


def domain_intersection(first, second):
    """
    Calculate intersection(s) of two domains.

    Args:
        first (set): (Set of) tuples defining the first domain.
        second (set): (Set of) tuples defining the second domain.

    Return:
        set: Intersection given by (start, end) tuples.
    """
    if isinstance(first, tuple):
        first = [first]
    if isinstance(first, set):
        first = list(first)
    if isinstance(second, tuple):
        second = [second]
    if isinstance(second, set):
        second = list(second)

    intersection = set()
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
            intersection.add((start, end))

    return intersection


def integrate_function(func, interval):
    """
    Numerically integrate a function on a given interval using
    :func:`.complex_quadrature`.

    Args:
        func(callable): Function to integrate.
        interval(list of tuples): List of (start, end) values of the intervals
            to integrate on.

    Return:
        tuple: (Result of the Integration, errors that occurred during the
        integration).
    """
    result = 0
    err = 0
    for area in interval:
        res = complex_quadrature(func, area[0], area[1])
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

    return (real_integral[0] + 1j * imag_integral[0],
            real_integral[1] + imag_integral[1])


def dot_product(first, second):
    """
    Calculate the inner product of two vectors.

    Args:
        first (:obj:`numpy.ndarray`): first vector
        second (:obj:`numpy.ndarray`): second vector

    Return:
        inner product
    """
    return np.inner(first, second)


def dot_product_l2(first, second):
    r"""
    Calculate the inner product on L2.

    Given two functions :math:`\varphi(z)` and :math:`\psi(z)` this functions
    calculates

    .. math::
        \left< \varphi(z) | \psi(z) \right> =
            \int\limits_{\Gamma_0}^{\Gamma_1}
            \bar\varphi(\zeta) \psi(\zeta) \,\textup{d}\zeta \:.

    Args:
        first (:py:class:`.Function`): first function
        second (:py:class:`.Function`): second function

    Return:
        inner product
    """
    if not isinstance(first, Function) or not isinstance(second, Function):
        raise TypeError("Wrong type(s) supplied. both must be a {0}".format(Function))

    if not first.domain == second.domain:
        raise ValueError("Domains of arguments must be identical, "
                         "but {} and {} were given".format(first.domain,
                                                           second.domain))
    nonzero = domain_intersection(first.nonzero, second.nonzero)
    areas = domain_intersection(first.domain, nonzero)

    # try some shortcuts
    if first == second:
        if hasattr(first, "quad_int"):
            return first.quad_int()

    if 0:
        # TODO let Function Class handle product to gain more speed
        if type(first) is type(second):
            pass

    # standard case
    def func(z):
        """
        Take the complex conjugate of the first element and multiply it
        by the second.
        """
        return np.conj(first(z)) * second(z)

    result, error = integrate_function(func, areas)
    return result


def vectorize_scalar_product(first, second, scalar_product):
    r"""
    Call the given :code:`scalar_product` in a loop for the arguments
    in :code:`left` and :code:`right`.

    Given two vectors of functions

    .. math::
        \boldsymbol{\varphi}(z)
        = \left(\varphi_0(z), \dotsc, \varphi_N(z)\right)^T

    and

    .. math::
        \boldsymbol{\psi}(z) = \left(\psi_0(z), \dotsc, \psi_N(z)\right)^T` ,

    this function computes
    :math:`\left< \boldsymbol{\varphi}(z) | \boldsymbol{\psi}(z) \right>_{L2}`
    where

    .. math::
        \left< \varphi_i(z) | \psi_j(z) \right>_{L2} =
        \int\limits_{\Gamma_0}^{\Gamma_1}
        \bar\varphi_i(\zeta) \psi_j(\zeta) \,\textup{d}\zeta \:.

    Herein, :math:`\bar\varphi_i(\zeta)` denotes the complex conjugate and
    :math:`\Gamma_0` as well as :math:`\Gamma_1` are derived by computing the
    intersection of the nonzero areas of the involved functions.

    Args:
        first (callable or :obj:`numpy.ndarray`):  (1d array of n) callable(s)
        second (callable or :obj:`numpy.ndarray`):  (1d array of n) callable(s)

    Raises:
        ValueError, if the provided arrays are not equally long.

    Return:
        numpy.ndarray:  Array of inner products
    """
    # sanitize input
    first = np.atleast_1d(first)
    second = np.atleast_1d(second)

    try:
        iter(scalar_product)
    except TypeError:
        scalar_product = (scalar_product, )

    if len(first) != len(second):
        raise ValueError("Provided function vectors must be of same length.")

    # calculate output size and allocate output
    out = np.zeros(first.shape, dtype=complex)

    # TODO propagate vectorization into _dot_product_l2 to save this loop
    # loop over entries
    for idx, (f, s) in enumerate(zip(first, second)):
        for m_idx, scal_prod in enumerate(scalar_product):
            out[idx] += scal_prod(f.get_member(m_idx), s.get_member(m_idx))

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
    return calculate_scalar_product_matrix(sanitize_input(values_a, Number),
                                           sanitize_input(values_b, Number),
                                           np.multiply)


def calculate_scalar_product_matrix(base_a, base_b, scalar_product=None,
                                    optimize=True):
    r"""
    Calculates a matrix :math:`A` , whose elements are the scalar products of
    each element from *base_a* and *base_b*, so that
    :math:`a_{ij} = \langle \mathrm{a}_i\,,\: \mathrm{b}_j\rangle`.

    Args:
        base_a (:py:class:`.ApproximationBase`): Basis a
        base_b (:py:class:`.ApproximationBase`): Basis b
        scalar_product: (List of) function objects that are passed the members
            of the given bases as pairs. Defaults to the scalar product given by
            `base_a`
        optimize (bool): Switch to turn on the symmetry based speed up.
            For development purposes only.

    Return:
        numpy.ndarray: matrix :math:`A`
    """
    if not base_a.is_compatible_to(base_b):
        raise TypeError("Bases must be from the same type.")

    if scalar_product is None:
        scalar_product = base_a.scalar_product_hint()

    fractions_a = base_a.fractions
    fractions_b = base_b.fractions

    if optimize and base_a == base_b:
        # since the scalar_product commutes whe can save some operations
        dim = fractions_a.shape[0]
        output = np.zeros((dim, dim), dtype=np.complex)
        i, j = np.mgrid[0:dim, 0:dim]

        # compute only upper half
        upper_idxs = np.triu_indices(dim)
        i_upper = i[upper_idxs]
        j_upper = j[upper_idxs]
        output[upper_idxs] = vectorize_scalar_product(fractions_a[i_upper],
                                                      fractions_a[j_upper],
                                                      scalar_product)

        # reconstruct using symmetry
        output += np.conjugate(np.triu(output, 1)).T
        return np.real_if_close(output)
    else:
        i, j = np.mgrid[0:fractions_a.shape[0],
                        0:fractions_b.shape[0]]
        fractions_i = fractions_a[i]
        fractions_j = fractions_b[j]

        res = vectorize_scalar_product(fractions_i.flatten(),
                                       fractions_j.flatten(),
                                       scalar_product)

        return res.reshape(fractions_i.shape)


def project_on_base(state, base):
    """
    Projects a *state* on a basis given by *base*.

    Args:
        state (array_like): List of functions to approximate.
        base (:py:class:`.ApproximationBase`): Basis to project onto.

    Return:
        numpy.ndarray: Weight vector in the given *base*
    """
    if not isinstance(base, ApproximationBasis):
        raise TypeError("Projection only possible on approximation bases.")

    # compute <x(z, t), phi_i(z)> (vector)
    projections = calculate_scalar_product_matrix(base.__class__(state),
                                                  base)

    # compute <phi_i(z), phi_j(z)> for 0 < i, j < n (matrix)
    scale_mat = calculate_scalar_product_matrix(base, base)

    res = np.linalg.inv(scale_mat) @ projections.T
    return np.reshape(res, (scale_mat.shape[0],))


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
        numpy.array: Finite dimensional state as 1d-array corresponding to the
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
        base (:py:class:`.ApproximationBase`): Base to be used for the projection.

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
        src_base (:py:class:`.ApproximationBase`): The source Basis.
        dst_base (:py:class:`.ApproximationBase`): The destination Basis.

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

    def as_tuple(self):
        return self.src_lbl, self.dst_lbl, self.src_order, self.dst_order

    def __hash__(self):
        return hash(self.as_tuple())

    def __eq__(self, other):
        if not isinstance(other, TransformationInfo):
            raise TypeError("Unknown type to compare with")

        return self.as_tuple() == other.as_tuple()

    def mirror(self):
        """
        Factory method, that creates a new TransformationInfo object by
        mirroring *src* and *dst* terms.
        This helps handling requests to different bases.
        """
        new_info = TransformationInfo()
        new_info.src_lbl = self.dst_lbl
        new_info.src_base = self.dst_base
        new_info.src_order = self.dst_order
        new_info.dst_lbl = self.src_lbl
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
                            source_order=0, destination_order=0):
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


def calculate_expanded_base_transformation_matrix(src_base, dst_base,
                                                  src_order, dst_order,
                                                  use_eye=False):
    r"""
    Constructs a transformation matrix :math:`\bar V` from basis given by
    *src_base* to basis given by *dst_base* that also transforms all temporal
    derivatives of the given weights.

    See:
        :py:func:`.calculate_base_transformation_matrix` for further details.

    Args:
        dst_base (:py:class:`.ApproximationBase`): New projection base.
        src_base (:py:class:`.ApproximationBase`): Current projection base.
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
                          + "'src_order'({1}) can provide for this strategy."
                            "").format(dst_order, src_order))

    # build core transformation
    if use_eye:
        core_transformation = np.eye(src_base.fractions.size)
    else:
        core_transformation = calculate_base_transformation_matrix(src_base,
                                                                   dst_base)

    # build block matrix
    part_transformation = block_diag(*[core_transformation
                                       for i in range(dst_order + 1)])
    complete_transformation = np.hstack([part_transformation]
                                        + [np.zeros((part_transformation.shape[0],
                                                     src_base.fractions.size))
                                           for i in range(src_order - dst_order)])
    return complete_transformation


def calculate_base_transformation_matrix(src_base, dst_base, scalar_product=None):
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
        src_base (:py:class:`.ApproximationBase`): Current projection base.
        dst_base (:py:class:`.ApproximationBase`): New projection base.
        scalar_product (list of callable): Callbacks for product calculation.
            Defaults to `scalar_product_hint` from `src_base`.

    Return:
        :py:class:`numpy.ndarray`: Transformation matrix :math:`V` .
    """
    if not src_base.is_compatible_to(dst_base):
        raise TypeError("Source and destination base must be from the same "
                        "type.")

    p_mat = calculate_scalar_product_matrix(dst_base, src_base, scalar_product)

    q_mat = calculate_scalar_product_matrix(dst_base, dst_base, scalar_product)

    # compute V matrix, where V = inv(Q)*P
    v_mat = np.dot(np.linalg.inv(q_mat), p_mat)
    return v_mat


def normalize_base(b1, b2=None):
    r"""
    Takes two :py:class:`.ApproximationBase`'s :math:`\boldsymbol{b}_1` ,
    :math:`\boldsymbol{b}_1` and normalizes them so that
    :math:`\langle\boldsymbol{b}_{1i}\,
    ,\:\boldsymbol{b}_{2i}\rangle = 1`.
    If only one base is given, :math:`\boldsymbol{b}_2`
    defaults to :math:`\boldsymbol{b}_1`.

    Args:
        b1 (:py:class:`.ApproximationBase`): :math:`\boldsymbol{b}_1`
        b2 (:py:class:`.ApproximationBase`): :math:`\boldsymbol{b}_2`

    Raises:
        ValueError: If :math:`\boldsymbol{b}_1`
            and :math:`\boldsymbol{b}_2` are orthogonal.

    Return:
        :py:class:`.ApproximationBase` : if *b2* is None,
        otherwise: Tuple of 2 :py:class:`.ApproximationBase`'s.
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


def generic_scalar_product(b1, b2=None, scalar_product=None):
    """
    Calculates the pairwise scalar product between the elements
    of the :py:class:`.ApproximationBase` *b1* and *b2*.

    Args:
        b1 (:py:class:`.ApproximationBase`): first basis
        b2 (:py:class:`.ApproximationBase`): second basis, if omitted
            defaults to *b1*
        scalar_product (list of callable): Callbacks for product calculation.
            Defaults to `scalar_product_hint` from `b1`.

    Note:
        If *b2* is omitted, the result can be used to normalize
        *b1* in terms of its scalar product.
    """
    if b2 is None:
        b2 = b1

    if type(b1) != type(b2):
        raise TypeError("only arguments of same type allowed.")

    if scalar_product is None:
        scalar_product = b1.scalar_product_hint()

    res = vectorize_scalar_product(b1, b2, scalar_product)

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

    if n_roots == 0:
        # Either no roots have been found or zero roots have been requested
        return np.array([])

    if len(roots) < n_roots:
        raise ValueError("Insufficient number of roots detected. ({0} < {1}) "
                         "Check provided function (see `visualize_roots`) or "
                         "try to increase the search area.".format(
            len(roots), n_roots))

    valid_roots = np.array(roots)

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
        val = func(np.complex(x[0], x[1]))
        return np.array([np.real(val),
                         np.imag(val)])

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
            # check for correct boundaries
            if bounds and not all(bounds == points[[0, -1]]):
                raise ValueError("Given 'bounds' don't fit the provided data.")

            # check for correct length
            if num is not None and len(points) != num:
                raise ValueError("Given 'num' doesn't fit the provided data.")

            # points are given, easy one
            self._values = np.atleast_1d(points)
            self._limits = (self._values.min(), self._values.max())
            self._num = self._values.size

            # check for evenly spaced entries
            if self._num > 1:
                steps = np.diff(self._values)
                equal_steps = np.allclose(steps, steps[0])
                if step:
                    if not equal_steps or step != steps[0]:
                        raise ValueError("Given 'step' doesn't fit the provided "
                                         "data.")
                else:
                    if equal_steps:
                        step = steps[0]
            else:
                step = np.nan
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
            if np.abs(step - self._step)/self._step > 1e-1:
                warnings.warn("desired step-size {} doesn't fit to given "
                              "interval, changing to {}".format(step,
                                                                self._step))
        else:
            raise ValueError("not enough arguments provided!")

        # mimic some ndarray properties
        self.shape = self._values.shape
        self.view = self._values.view

    def __repr__(self):
        return "Domain(bounds={}, step={}, num={})".format(self.bounds,
                                                           self._step,
                                                           self._num)

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

    @property
    def ndim(self):
        return self._values.ndim


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
    This class helps managing any kind of result data.

    The data gained by evaluation of a function is stored together with the
    corresponding points of its evaluation. This way all data needed for
    plotting or other postprocessing is stored in one place.
    Next to the points of the evaluation the names and units of the included
    axes can be stored.
    After initialization an interpolator is set up, so that one can interpolate
    in the result data by using the overloaded :py:meth:`__call__` method.

    Args:
        input_data: (List of) array(s) holding the axes of a regular grid on
            which the evaluation took place.
        output_data: The result of the evaluation.

    Keyword Args:
        input_labels: (List of) labels for the input axes.
        input_units: (List of) units for the input axes.
        name: Name of the generated data set.
        fill_axes: If the dimension of `output_data` is higher than the
            length of the given `input_data` list, dummy entries will be
            appended until the required dimension is reached.
        enable_extrapolation (bool): If True, internal interpolators will allow
            extrapolation. Otherwise, the last giben value will be repeated for
            1D cases and the result will be padded with zeros for cases > 1D.
        fill_value: If invalid data is encountered, it will be replaced with
            this value before interpolation is performed.

    Examples:
        When instantiating 1d EvalData objects, the list can be omitted

        >>> axis = Domain((0, 10), 5)
        >>> data = np.random.rand(5,)
        >>> e_1d = EvalData(axis, data)

        For other cases, input_data has to be a list

        >>> axis1 = Domain((0, 0.5), 5)
        >>> axis2 = Domain((0, 1), 11)
        >>> data = np.random.rand(5, 11)
        >>> e_2d = EvalData([axis1, axis2], data)

        Adding two Instances (if the boundaries fit, the data will be
        interpolated on the more coarse grid.) Same goes for subtraction and
        multiplication.

        >>> e_1 = EvalData(Domain((0, 10), 5), np.random.rand(5,))
        >>> e_2 = EvalData(Domain((0, 10), 10), 100*np.random.rand(5,))
        >>> e_3 = e_1 + e_2
        >>> e_3.output_data.shape
        (5,)

        Interpolate in the output data by calling the object

        >>> e_4 = EvalData(np.array(range(5)), 2*np.array(range(5))))
        >>> e_4.output_data
        array([0, 2, 4, 6, 8])
        >>> e_5 = e_4([2, 5])
        >>> e_5.output_data
        array([4, 8])
        >>> e_5.output_data.size
        2

        one may also give a slice

        >>> e_6 = e_4(slice(1, 5, 2))
        >>> e_6.output_data
        array([2., 6.])
        >>> e_5.output_data.size
        2

        For multi-dimensional interpolation a list has to be provided

        >>> e_7 = e_2d([[.1, .5], [.3, .4, .7)])
        >>> e_7.output_data.shape
        (2, 3)

    """
    def __init__(self, input_data, output_data,
                 input_labels=None, input_units=None,
                 enable_extrapolation=False,
                 fill_axes=False, fill_value=None,
                 name=None):
        # check type and dimensions
        if isinstance(input_data, np.ndarray) and input_data.ndim == 1:
            # accept single array for single dimensional input
            input_data = [input_data]
        elif isinstance(input_data, Domain) and input_data.points.ndim == 1:
            # some goes for domains
            input_data = [input_data]
        else:
            assert isinstance(input_data, list)

        # convert numpy arrays to domains
        input_data = [Domain(points=entry)
                      if isinstance(entry, np.ndarray) else entry
                      for entry in input_data]

        # if a list with names is provided, the dimension must fit
        if input_labels is None:
            input_labels = ["" for i in range(len(input_data))]
        if not isinstance(input_labels, list):
            input_labels = [input_labels]
        assert len(input_labels) == len(input_data)

        # if a list with units is provided, the dimension must fit
        if input_units is None:
            input_units = ["" for i in range(len(input_data))]
        if not isinstance(input_units, list):
            input_units = [input_units]
        assert len(input_units) == len(input_data)

        assert isinstance(output_data, np.ndarray)
        if output_data.size == 0:
            raise ValueError("No initialisation possible with an empty array!")

        if fill_axes:
            # add dummy axes to input_data for missing output dimensions
            dim_diff = output_data.ndim - len(input_data)
            for dim in range(dim_diff):
                input_data.append(Domain(points=np.array(
                    range(output_data.shape[-(dim_diff - dim)]))))
                input_labels.append("")
                input_units.append("")

        # output_data has to contain len(input_data) dimensions
        assert len(input_data) == output_data.ndim

        for dim in range(len(input_data)):
            assert len(input_data[dim]) == output_data.shape[dim]

        self.input_data = input_data
        self.output_data = output_data
        self.min = np.nanmin(output_data)
        self.max = np.nanmax(output_data)

        if len(input_data) == 1:
            if enable_extrapolation:
                fill_val = "extrapolate"
            else:
                fill_val = (output_data[0], output_data[-1])

            self._interpolator = interp1d(
                input_data[0],
                np.ma.fix_invalid(output_data, fill_value=fill_value),
                axis=-1,
                bounds_error=False,
                fill_value=fill_val)
        elif len(input_data) == 2 and output_data.ndim == 2:
            # pure 2d case
            if enable_extrapolation:
                raise ValueError("Extrapolation not supported for 2d data. See "
                                 "https://github.com/scipy/scipy/issues/8099"
                                 "for details.")
            if len(input_data[0]) > 3 and len(input_data[1]) > 3 and False:
                # special treatment for very common case (faster than interp2d)
                # boundary values are used as fill values
                self._interpolator = RectBivariateSpline(
                    *input_data,
                    np.ma.fix_invalid(output_data, fill_value=fill_value)
                )
            else:
                # this will trigger nearest neighbour interpolation
                fill_val = None

                # if enable_extrapolation:
                #     fill_val = None
                # else:
                #     Since the value has to be the same at every border
                #     fill_val = 0

                self._interpolator = interp2d(
                    input_data[0],
                    input_data[1],
                    np.ma.fix_invalid(output_data.T, fill_value=fill_value),
                    bounds_error=False,
                    fill_value=fill_val)
        else:
            if enable_extrapolation:
                fill_val = None
            else:
                # Since the value has to be the same at every border
                fill_val = 0

            self._interpolator = RegularGridInterpolator(
                input_data,
                np.ma.fix_invalid(output_data, fill_value=fill_value),
                bounds_error=False,
                fill_value=fill_val)

        # handle names and units
        self.input_labels = input_labels
        self.input_units = input_units
        self.name = name
        if self.name is None:
            self.name = ""

    def adjust_input_vectors(self, other):
        """
        Check the the inputs vectors of `self` and `other` for compatibility
        (equivalence) and harmonize them if they are compatible.

        The compatibility check is performed for every input_vector in
        particular and examines whether they share the same boundaries.
        and equalize to the minimal discretized axis.
        If the amount of discretization steps between the two instances differs,
        the more precise discretization is interpolated down onto the less
        precise one.

        Args:
            other (:py:class:`.EvalData`): Other EvalData class.

        Returns:
            tuple:

                - (list) - New common input vectors.
                - (numpy.ndarray) - Interpolated self output_data array.
                - (numpy.ndarray) - Interpolated other output_data array.
        """
        assert len(self.input_data) == len(other.input_data)

        if self.input_data == other.input_data:
            return self.input_data, self.output_data, other.output_data

        input_data = []
        for idx in range(len(self.input_data)):
            # check if axis have the same length
            if self.input_data[idx].bounds != other.input_data[idx].bounds:
                raise ValueError("Boundaries of input vector {0} don't match."
                                 " {1} (self) != {2} (other)".format(
                    idx,
                    self.input_data[idx].bounds,
                    other.input_data[idx].bounds
                ))

            # check which axis has the worst discretization
            if len(self.input_data[idx]) <= len(other.input_data[idx]):
                input_data.append(self.input_data[idx])
            else:
                input_data.append(other.input_data[idx])

        # interpolate data
        interpolated_self = self.interpolate(input_data)
        interpolated_other = other.interpolate(input_data)

        return (input_data,
                interpolated_self.output_data,
                interpolated_other.output_data)

    def add(self, other, from_left=True):
        """
        Perform the element-wise addition of the output_data arrays from `self`
        and `other`

        This method is used to support addition by implementing
        __add__ (fromLeft=True) and __radd__(fromLeft=False)).
        If `other**` is a :py:class:`.EvalData`, the `input_data` lists of
        `self` and `other` are adjusted using :py:meth:`.adjust_input_vectors`
        The summation operation is performed on the interpolated output_data.
        If `other` is a :class:`numbers.Number` it is added according to
        numpy's broadcasting rules.

        Args:
            other (:py:class:`numbers.Number` or :py:class:`.EvalData`): Number
                or EvalData object to add to self.
            from_left (bool): Perform the addition from left if True or from
                right if False.

        Returns:
            :py:class:`.EvalData` with adapted input_data and output_data as
            result of the addition.
        """
        if isinstance(other, numbers.Number):
            if from_left:
                output_data = self.output_data + other
            else:
                output_data = other + self.output_data
            return EvalData(input_data=deepcopy(self.input_data),
                            output_data=output_data,
                            name="{} + {}".format(self.name, other))

        elif isinstance(other, EvalData):
            (input_data, self_output_data, other_output_data
             ) = self.adjust_input_vectors(other)

            # add the output arrays
            if from_left:
                output_data = self_output_data + other_output_data
                _name = self.name + " + " + other.name
            else:
                output_data = other_output_data + self_output_data
                _name = other.name + " + " + self.name

            return EvalData(input_data=deepcopy(input_data),
                            output_data=output_data,
                            name=_name)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.add(other, from_left=False)

    def __add__(self, other):
        return self.add(other)

    def sub(self, other, from_left=True):
        """
        Perform the element-wise subtraction of the output_data arrays from
        `self` and `other` .

        This method is used to support subtraction by implementing
        __sub__ (from_left=True) and __rsub__(from_left=False)).
        If `other**` is a :py:class:`.EvalData`, the `input_data` lists of
        `self` and `other` are adjusted using :py:meth:`.adjust_input_vectors`.
        The subtraction operation is performed on the interpolated output_data.
        If `other` is a :class:`numbers.Number` it is handled according to
        numpy's broadcasting rules.

        Args:
            other (:py:class:`numbers.Number` or :py:class:`.EvalData`): Number
                or EvalData object to subtract.
            from_left (boolean): Perform subtraction from left if True or from
                right if False.

        Returns:
            :py:class:`.EvalData` with adapted input_data and output_data as
            result of subtraction.
        """
        if isinstance(other, numbers.Number):
            if from_left:
                output_data = self.output_data - other
            else:
                output_data = other - self.output_data
            return EvalData(input_data=deepcopy(self.input_data),
                            output_data=output_data,
                            name="{} - {}".format(self.name, other))

        elif isinstance(other, EvalData):
            (input_data, self_output_data, other_output_data
             ) = self.adjust_input_vectors(other)

            # subtract the output arrays
            if from_left:
                output_data = self_output_data - other_output_data
                _name = self.name + " - " + other.name
            else:
                output_data = other_output_data - self_output_data
                _name = other.name + " - " + self.name

            return EvalData(input_data=deepcopy(input_data),
                            output_data=output_data,
                            name=_name)
        else:
            return NotImplemented

    def __rsub__(self, other):
        return self.sub(other, from_left=False)

    def __sub__(self, other):
        return self.sub(other)

    def mul(self, other, from_left=True):
        """
        Perform the element-wise multiplication of the output_data arrays from
        `self` and `other` .

        This method is used to support multiplication by implementing
        __mul__ (from_left=True) and __rmul__(from_left=False)).
        If `other**` is a :py:class:`.EvalData`, the `input_data` lists of
        `self` and `other` are adjusted using :py:meth:`.adjust_input_vectors`.
        The multiplication operation is performed on the interpolated
        output_data. If `other` is a :class:`numbers.Number` it is handled
        according to numpy's broadcasting rules.

        Args:
            other (:class:`numbers.Number` or :py:class:`.EvalData`): Factor
                to multiply with.
            from_left boolean: Multiplication from left if True or from right
                if False.

        Returns:
            :py:class:`.EvalData` with adapted input_data and output_data as
            result of multiplication.
        """
        if isinstance(other, numbers.Number):
            if from_left:
                output_data = self.output_data * other
            else:
                output_data = other * self.output_data
            return EvalData(input_data=deepcopy(self.input_data),
                            output_data=output_data,
                            name="{} - {}".format(self.name, other))

        elif isinstance(other, EvalData):
            (input_data, self_output_data, other_output_data
             ) = self.adjust_input_vectors(other)

            # addition der output array
            output_data = other_output_data * self_output_data
            if from_left:
                _name = self.name + " * " + other.name
            else:
                _name = other.name + " * " + self.name

            return EvalData(input_data=deepcopy(input_data),
                            output_data=output_data,
                            name=_name)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.mul(other, from_left=False)

    def __mul__(self, other):
        return self.mul(other)

    def matmul(self, other, from_left=True):
        """
        Perform the matrix multiplication of the output_data arrays from
        `self` and `other` .

        This method is used to support matrix multiplication (@) by implementing
        __matmul__ (from_left=True) and __rmatmul__(from_left=False)).
        If `other**` is a :py:class:`.EvalData`, the `input_data` lists of
        `self` and `other` are adjusted using :py:meth:`.adjust_input_vectors`.
        The matrix multiplication operation is performed on the interpolated
        output_data.
        If `other` is a :class:`numbers.Number` it is handled according to
        numpy's broadcasting rules.

        Args:
            other (:py:class:`EvalData`): Object to multiply with.
            from_left (boolean): Matrix multiplication from left if True or
                from right if False.

        Returns:
            :py:class:`EvalData` with adapted input_data and output_data as
            result of matrix multiplication.
        """
        if isinstance(other, EvalData):
            (input_data, self_output_data, other_output_data
             ) = self.adjust_input_vectors(other)
            if self.output_data.shape != other.output_data.shape:
                raise ValueError("Dimension mismatch")

            if from_left:
                output_data = self_output_data @ other_output_data
                _name = self.name + " @ " + other.name
            else:
                output_data = other_output_data @ self_output_data
                _name = other.name + " @ " + self.name

            return EvalData(input_data=deepcopy(input_data),
                            output_data=output_data,
                            name=_name)
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        return self.matmul(other, from_left=False)

    def __matmul__(self, other):
        return self.matmul(other)

    def __pow__(self, power):
        """
        Raise the elements form `self.output_data` element-wise to `power`.

        Args:
            power (:class:`numbers.Number`): Power to raise to.

        Returns:
            :py:class:`EvalData` with self.input_data and output_data as results
            of the raise operation.
        """
        if isinstance(power, numbers.Number):
            output_data = self.output_data ** power
            return EvalData(input_data=deepcopy(self.input_data),
                            output_data=output_data,
                            name="{} ** {}".format(self.name, power))
        else:
            return NotImplemented

    def sqrt(self):
        """
        Radicate the elements form `self.output_data` element-wise.

        Return:
             :py:class:`EvalData` with self.input_data and output_data as result
             of root calculation.
        """
        output_data = np.sqrt(self.output_data)

        ed = EvalData(input_data=deepcopy(self.input_data),
                      output_data=output_data,
                      name="sqrt({})".format(self.name))
        return ed

    def abs(self):
        """
        Get the absolute value of the elements form `self.output_data` .

        Return:
            :py:class:`EvalData` with self.input_data and output_data as result
            of absolute value calculation.
        """
        output_data = np.abs(self.output_data)

        ed = EvalData(input_data=deepcopy(self.input_data),
                      output_data=output_data,
                      name="abs({})".format(self.name))
        return ed

    def __call__(self, interp_axes, as_eval_data=True):
        """
        Interpolation method for output_data.

        Determines, if a one, two or three dimensional interpolation is used.
        Method can handle slice objects in the pos lists.
        One slice object is allowed per axis list.

        Args:
            interp_axes (list(list)): Axis positions in the form

            - 1D: [axis] with axis=[1,2,3]
            - 2D: [axis1, axis2] with axis1=[1,2,3] and axis2=[0,1,2,3,4]

            as_eval_data (bool): Return the interpolation result as EvalData
                object. If `False`, the output_data array of the results is
                returned.

        Returns:
            :py:class:`EvalData` with pos as input_data and to pos interpolated
            output_data.
        """
        if len(self.input_data) == 1:
            # special case for 1d data where the outermost list can be omitted
            if isinstance(interp_axes, slice):
                interp_axes = [interp_axes]
            if isinstance(interp_axes, list) and \
                    all([isinstance(e, Number) for e in interp_axes]):
                interp_axes = [interp_axes]

        assert isinstance(interp_axes, list)
        dim_err = len(self.input_data) - len(interp_axes)
        assert dim_err >= 0
        interp_axes += [slice(None) for x in range(dim_err)]
        assert len(interp_axes) == len(self.input_data)

        _list = []
        for i, interp_points in enumerate(interp_axes):
            if isinstance(interp_points, slice):
                _entry = self.input_data[i][interp_points]
                if _entry is None:
                    raise ValueError("Quantity resulting from slice is empty!")
            else:
                try:
                    _entry = list(interp_points)
                except TypeError as e:
                    raise ValueError("Coordinates must be given as iterable!")
            _list.append(_entry)

        res = self.interpolate(_list)

        if as_eval_data:
            return res
        else:
            return res.output_data

    def interpolate(self, interp_axis):
        """
        Main interpolation method for output_data.

        If one of the output dimensions is to be interpolated at one single
        point, the dimension of the output will decrease by one.

        Args:
            interp_axis (list(list)): axis positions in the form

            - 1D: axis with axis=[1,2,3]
            - 2D: [axis1, axis2] with axis1=[1,2,3] and axis2=[0,1,2,3,4]

        Returns:
            :py:class:`EvalData` with `interp_axis` as new input_data and
            interpolated output_data.
        """
        assert isinstance(interp_axis, list)
        assert len(interp_axis) == len(self.input_data)

        # check if an axis has been degenerated
        domains = [Domain(points=axis) for axis in interp_axis if len(axis) > 1]

        if len(self.input_data) == 1:
            interpolated_output = self._interpolator(interp_axis[0])
        elif len(self.input_data) == 2:
            interpolated_output = self._interpolator(*interp_axis)
            if isinstance(self._interpolator, interp2d):
                interpolated_output = interpolated_output.T
        else:
            dims = tuple(len(a) for a in interp_axis)
            coords = np.array(
                [a.flatten() for a in np.meshgrid(*interp_axis, indexing="ij")])
            interpolated_output = self._interpolator(coords.T).reshape(dims)

        out_arr = ma.masked_invalid(interpolated_output).squeeze()
        return EvalData(input_data=domains,
                        output_data=out_arr,
                        name=self.name)
