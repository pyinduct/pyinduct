"""
In :py:mod:`pyinduct.placeholder` you find placeholders for symbolic Term
definitions.
"""

import collections
import copy
from abc import ABCMeta
from numbers import Number
import warnings

import numpy as np

from .core import sanitize_input, Base, Function
from .registry import register_base, get_base, is_registered

__all__ = ["Scalars", "ScalarFunction", "TestFunction", "FieldVariable",
           "Input", "ObserverGain",
           "Product",
           "ScalarTerm", "IntegralTerm", "ScalarProductTerm",
           "Placeholder"]


class Placeholder(object):
    """
    Base class that works as a placeholder for terms that are later parsed into
    a canonical form.

    Args:
        data (arbitrary): data to store in the placeholder.
        order (tuple): (temporal_order, spatial_order) derivative orders  that
            are to be applied before evaluation.
        location (numbers.Number): Location to evaluate at before further
            computation.

    Todo:
        convert order and location into attributes with setter and getter
        methods. This will close the gap of unchecked values for order and
        location that can be sneaked in by the copy constructors by
        circumventing code doubling.
    """

    def __init__(self, data, order=(0, 0), location=None):
        self.data = data

        if (not isinstance(order, tuple)
            or any([not isinstance(o, int)
                    or o < 0 for o in order])):
            raise ValueError("invalid derivative order.")
        self.order = order

        if location is not None:
            if location and not isinstance(location, Number):
                raise TypeError("location must be a number")
        self.location = location

    def derivative(self, temp_order=0, spat_order=0):
        """
        Mimics a copy constructor and adds the given derivative orders.

        Note:
            The desired derivative order :code:`order` is added to the original
            order.

        Args:
            temp_order: Temporal derivative order to be added.
            spat_order: Spatial derivative order to be added.

        Returns:
            New :py:class:`.Placeholder` instance with the desired derivative
            order.
        """
        new_obj = copy.deepcopy(self)
        new_obj.order = tuple(der + a
                              for der, a in zip(self.order,
                                                (temp_order, spat_order)))
        return new_obj

    def __call__(self, location):
        """
        Mimics a copy constructor and adds the given location for spatial
        evaluation.

        Args:
            location: Spatial Location to be set.

        Returns:
            New :py:class:`.Placeholder` instance with the desired location.
        """
        new_obj = copy.deepcopy(self)
        new_obj.location = location
        return new_obj


class SpatialPlaceholder(Placeholder):
    """
    Base class for all spatially-only dependent placeholders.
    The deeper meaning of this abstraction layer is to offer an easier to use
    interface.
    """

    def __init__(self, data, order=0, location=None):
        Placeholder.__init__(self, data, order=(0, order), location=location)

    def derive(self, order=1):
        """
        Take the (spatial) derivative of this object.
        Args:
            order: Derivative order.

        Returns:
            :py:class:`.Placeholder`: The derived expression.
        """
        return self.derivative(spat_order=order)


class Scalars(Placeholder):
    """
    Placeholder for scalar values that scale the equation system,
    gained by the projection of the pde onto the test basis.

    Note:
        The arguments *target_term* and *target_form* are used inside the
        parser. For frontend use, just specify the *values*.

    Args:
        values: Iterable object containing the scalars for every k-th equation.
        target_term: Coefficient matrix to :py:func:`.add_to`.
        target_form: Desired weight set.
    """

    def __init__(self, values, target_term=None, target_form=None,
                 test_func_lbl=None):
        if target_term is None:
            target_term = dict(name="f")
        values = np.atleast_2d(values)

        super().__init__(sanitize_input(values, Number))
        self.target_term = target_term
        self.target_form = target_form


class ScalarFunction(SpatialPlaceholder):
    """
    Class that works as a placeholder for spatial functions in an equation.
    An example could be spatial dependent coefficients.

    Args:
        function_label (str): label under which the function is registered
        order (int): spatial derivative order to use
        location: location to evaluate at

    Warn:
        There seems to be a problem when this function is used in combination
        with the :py:class:`.Product` class. Make sure to provide this class as
        first argument to any product you define.

    Todo:
        see warning.

    """

    def __init__(self, function_label, order=0, location=None):
        if not is_registered(function_label):
            raise ValueError("Unknown function label"
                             " '{0}'!".format(function_label))

        super().__init__({"func_lbl": function_label},
                         order=order,
                         location=location)

    @staticmethod
    def from_scalar(scalar, label, **kwargs):
        """
        create a :py:class:`.ScalarFunction` from scalar values.

        Args:
            scalar (array like): Input that is used to generate the
                placeholder. If a number is given, a constant function will be
                created, if it is callable it will be wrapped in a
                :py:class:`.Function` and registered.
            label (string): Label to register the created base.
            **kwargs: All kwargs that are not mentioned below will be passed
                to :py:class:`.Function`.

        Keyword Args:
            order (int): See constructor.
            location (int): See constructor.
            overwrite (bool): See :py:func:`.register_base`

        Returns:
            :py:class:`.ScalarFunction` : Placeholder object that
            can be used in a weak formulation.
        """

        order = kwargs.pop("order", 0)
        loc = kwargs.pop("location", None)
        over = kwargs.pop("overwrite", False)

        if isinstance(scalar, Number):
            f = Function.from_constant(scalar, **kwargs)
        elif isinstance(scalar, Function):
            f = scalar
        elif isinstance(scalar, collections.Callable):
            f = Function(scalar, **kwargs)
        else:
            raise TypeError("Coefficient type not understood.")

        register_base(label, Base(f), overwrite=over)

        return ScalarFunction(label, order, loc)


class Input(Placeholder):
    """
    Class that works as a placeholder for an input of the system.

    Args:
        function_handle (callable): Handle that will be called by the simulation
            unit.
        index (int): If the system's input is vectorial, specify the element to
            be used.
        order (int): temporal derivative order of this term
            (See :py:class:`.Placeholder`).
        exponent (numbers.Number): See :py:class:`.FieldVariable`.

    Note:
        if *order* is nonzero, the callable is expected to return the temporal
        derivatives of the input signal by returning an array of
        ``len(order)+1``.
    """

    def __init__(self, function_handle, index=0, order=0, exponent=1):
        if not isinstance(function_handle, collections.Callable):
            raise TypeError("callable object has to be provided.")
        if not isinstance(index, int) or index < 0:
            raise TypeError("index must be a positive integer.")

        if not isinstance(exponent, Number):
            raise TypeError("exponent must be a number")
        if exponent != 1:
            raise ValueError("Providing exponents that differ from 1 is no "
                             "longer supported.")

        super().__init__(dict(input=function_handle,
                              index=index,
                              exponent=exponent),
                         order=(order, 0))


class ObserverGain(Placeholder):
    """
    Class that works as a placeholder for the observer error gain.

    Args:
        observer_feedback (:py:class:`.ObserverFeedback`): Handle that will be
            called by the simulation unit.
    """
    def __init__(self, observer_feedback):
        super().__init__(dict(obs_fb=observer_feedback))


class TestFunction(SpatialPlaceholder):
    """
    Class that works as a placeholder for test functions in an equation.

    Args:
        function_label (str): Label of the function test base.
        order (int): Spatial derivative order.
        location (Number): Point of evaluation / argument of the function.
        approx_label (str): Label of the approximation test base.
    """

    def __init__(self, function_label, order=0, location=None,
                 approx_label=None):
        if not is_registered(function_label):
            raise ValueError("Unknown function label "
                             "'{0}'!".format(function_label))

        if approx_label is None:
            approx_label = function_label
        elif not isinstance(approx_label, str):
            raise TypeError("only strings allowed as 'approx_label'")

        super().__init__({"func_lbl": function_label, "appr_lbl": approx_label},
                         order, location=location)


class FieldVariable(Placeholder):
    r"""
    Class that represents terms of the systems field variable :math:`x(z, t)`.

    Args:
        function_label (str): Label of shapefunctions to use for approximation,
            see :py:func:`.register_base` for more information about how to
            register an approximation basis.
        order tuple of int: Tuple of temporal_order and spatial_order derivation
            order.
        weight_label (str): Label of weights for which coefficients are to be
            calculated (defaults to function_label).
        location: Where the expression is to be evaluated.
        exponent: Exponent of the term.

    Examples:
        Assuming some shapefunctions have been registered under the label
        ``"phi"`` the following expressions hold:

        - :math:`\frac{\partial^{2}}{\partial t \partial z}x(z, t)`

        >>> x_dtdz = FieldVariable("phi", order=(1, 1))

        - :math:`\frac{\partial^2}{\partial t^2}x(3, t)`

        >>> x_ddt_at_3 = FieldVariable("phi", order=(2, 0), location=3)

        - :math:`\frac{\partial}{\partial t}x^2(z, t)`

        >>> x_dt_squared = FieldVariable("phi", order=(1, 0), exponent=2)
    """

    def __init__(self, function_label, order=(0, 0),
                 weight_label=None, location=None,
                 exponent=1, raised_spatially=False):
        """
        """
        # derivative orders
        if not isinstance(order, tuple) or len(order) > 2:
            raise TypeError("order mus be 2-tuple of int.")
        if any([True for n in order if n < 0]):
            raise ValueError("derivative orders must be positive")

        if location is not None:
            if location and not isinstance(location, Number):
                raise TypeError("location must be a number")

        # basis
        if not is_registered(function_label):
            raise ValueError("Unknown function label "
                             "'{0}'!".format(function_label))
        if weight_label is None:
            weight_label = function_label
        elif not isinstance(weight_label, str):
            raise TypeError("only strings allowed as 'weight_label'")
        if function_label == weight_label:
            self.simulation_compliant = True
        else:
            self.simulation_compliant = False

        self.raised_spatially = raised_spatially

        if not isinstance(exponent, Number):
            raise TypeError("exponent must be a number")
        if exponent != 1:
            raise ValueError("Providing exponents that differ from 1 is no "
                             "longer supported.")

        super().__init__({"func_lbl": function_label,
                          "weight_lbl": weight_label,
                          "exponent": exponent},
                         order=order,
                         location=location)

    def derive(self, *, temp_order=0, spat_order=0):
        """
        Derive the expression to the specified order.

        Args:
            temp_order: Temporal derivative order.
            spat_order: Spatial derivative order.

        Returns:
            :py:class:`.Placeholder`: The derived expression.

        Note:
            This method uses keyword only arguments, which means that a call
            will fail if the arguments are passed by order.

        """
        return self.derivative(temp_order, spat_order)


class TemporalDerivedFieldVariable(FieldVariable):
    def __init__(self, function_label, order, weight_label=None, location=None):
        warnings.warn("TemporalDerivedFieldVariable will deprecated in v0.6.0,"
                      "use FieldVariable.derive(temp_order=order) instead",
                      PendingDeprecationWarning)
        FieldVariable.__init__(self,
                               function_label,
                               (order, 0),
                               weight_label,
                               location)


class SpatialDerivedFieldVariable(FieldVariable):
    def __init__(self, function_label, order, weight_label=None, location=None):
        warnings.warn("SpatialDerivedFieldVariable will deprecated in v0.6.0,"
                      "use FieldVariable.derive(spat_order=order) instead",
                      PendingDeprecationWarning)
        FieldVariable.__init__(self,
                               function_label,
                               (0, order),
                               weight_label,
                               location)


class Product(object):
    """
    Represents a product.

    Args:
        a:
        b:
    """

    def __init__(self, a, b=None):
        # convenience: accept single arguments
        if b is None:  # multiply by one as Default
            self.b_empty = True
            if isinstance(a, Input):
                b = Scalars(np.ones(1))
            if isinstance(a, Scalars):
                if a.target_term["name"] == "E":
                    b = Scalars(np.ones(a.data.T.shape))
                else:
                    b = Scalars(np.ones(a.data.shape))
                    # TODO other Placeholders?
        else:
            self.b_empty = False

        # convert trivial products (arising from simplification)
        if isinstance(a, Product) and a.b_empty:
            a = a.args[0]
        if isinstance(b, Product) and b.b_empty:
            b = b.args[0]

        # check for allowed terms
        if (not isinstance(a, Placeholder)
            or (b is not None and not isinstance(b, Placeholder))):
            raise TypeError("argument not allowed in product")

        a, b = self._simplify_product(a, b)
        if b is None:
            self.b_empty = True

        a, b = self._evaluate_terms(a, b)
        self.args = [a, b]

    @staticmethod
    def _evaluate_terms(a, b):
        # evaluate all terms that can be evaluated
        args = (a, b)
        new_args = []
        for idx, arg in enumerate(args):
            if getattr(arg, "location", None) is not None:
                # evaluate term and add scalar
                # print("WARNING: converting Placeholder that is to be evaluated
                #  into 'Scalars' object.")
                new_args.append(_evaluate_placeholder(arg))
            else:
                new_args.append(arg)
        return new_args

    @staticmethod
    def _simplify_product(a, b):
        # try to simplify expression containing ScalarFunctions
        scalar_func = None
        other_func = None
        for obj1, obj2 in [(a, b), (b, a)]:
            if isinstance(obj1, ScalarFunction):
                scalar_func = obj1
                if isinstance(obj2,
                              (FieldVariable, TestFunction, ScalarFunction)):
                    other_func = obj2
                    break

        if scalar_func and other_func:
            s_func = get_base(scalar_func.data["func_lbl"]).derive(
                scalar_func.order[1]).fractions
            o_func = get_base(other_func.data["func_lbl"]).derive(
                other_func.order[1]).fractions

            if s_func.shape != o_func.shape:
                if s_func.shape[0] == 1:
                    # only one function provided, use it for all others
                    s_func = s_func[[0] * o_func.shape[0]]
                else:
                    raise ValueError("Cannot simplify Product due to dimension "
                                     "mismatch!")

            exp = other_func.data.get("exponent", 1)

            if scalar_func.location is None:
                new_base = Base(np.asarray(
                    [func.raise_to(exp).scale(scale_func)
                     for func, scale_func in zip(o_func, s_func)]))
            else:
                new_base = Base(np.asarray(
                    [func.raise_to(exp).scale(scale_func(scalar_func.location))
                     for func, scale_func in zip(o_func, s_func)]))

            # TODO change name generation to more sane behaviour
            new_name = new_base.fractions.tobytes()
            register_base(new_name, new_base)

            # overwrite spatial derivative order since derivation took place
            if isinstance(other_func, TestFunction):
                a = other_func.__class__(
                    function_label=new_name,
                    order=0,
                    location=other_func.location,
                    approx_label=other_func.data["appr_lbl"])

            elif isinstance(other_func, ScalarFunction):
                a = other_func.__class__(
                    function_label=new_name,
                    order=0,
                    location=other_func.location)

            elif isinstance(other_func, FieldVariable):
                a = copy.deepcopy(other_func)
                a.data["func_lbl"] = new_name
                a.order = (other_func.order[0], 0)

            b = None

        return a, b

    def get_arg_by_class(self, cls):
        """
        Extract element from product that is an instance of cls.

        Args:
            cls:

        Return:
            list:
        """
        return [elem for elem in self.args if isinstance(elem, cls)]


class EquationTerm(object, metaclass=ABCMeta):
    """
    Base class for all accepted terms in a weak formulation.

    Args:
        scale:
        arg:
    """

    def __init__(self, scale, arg):
        if not isinstance(scale, Number):
            raise TypeError("only numbers allowed as scale.")
        self.scale = scale

        # convenience: convert single argument
        if not isinstance(arg, Product):
            if isinstance(arg, Placeholder):
                # arg = Product(arg)
                self.arg = Product(arg, None)
            else:
                raise TypeError("argument not supported.")
        else:
            self.arg = arg


class ScalarTerm(EquationTerm):
    """
    Class that represents a scalar term in a weak equation.

    Args:
        argument:
        scale:
    """

    def __init__(self, argument, scale=1.0):
        EquationTerm.__init__(self, scale, argument)

        if any([True for arg in self.arg.args
                if isinstance(arg, (FieldVariable, TestFunction))]):
            raise ValueError("cannot leave z dependency. specify location to "
                             "evaluate expression.")


class IntegralTerm(EquationTerm):
    """
    Class that represents an integral term in a weak equation.

    Args:
        integrand:
        limits (tuple):
        scale:
    """

    def __init__(self, integrand, limits, scale=1.0):
        EquationTerm.__init__(self, scale, integrand)

        if not any([isinstance(arg, (FieldVariable, TestFunction))
                    for arg in self.arg.args]):
            raise ValueError("nothing to integrate")
        if not isinstance(limits, tuple):
            raise TypeError("limits must be provided as tuple")
        self.limits = limits


class ScalarProductTerm(EquationTerm):
    """
    Class that represents a scalar product in a weak equation.

    Args:
        arg1: Fieldvariable (Shapefunctions) to be projected.
        arg2: Testfunctions to project on.
        scale (Number): Scaling of expression.
    """

    def __init__(self, arg1, arg2,  scale=1.0):
        if not any([isinstance(arg, (FieldVariable, TestFunction))
                    for arg in (arg2, arg2)]):
            raise ValueError("nothing to integrate")
        EquationTerm.__init__(self, scale, (arg1, arg2))


def _evaluate_placeholder(placeholder):
    """
    Evaluates a placeholder object and returns a Scalars object.

    Args:
        placeholder (:py:class:`.Placeholder`):

    Return:
        :py:class:`.Scalars` or NotImplementedError
    """
    if not isinstance(placeholder, Placeholder):
        raise TypeError("only placeholders supported")
    if isinstance(placeholder, (Scalars, Input)):
        raise TypeError("provided type cannot be evaluated")

    fractions = get_base(placeholder.data['func_lbl']).derive(
        placeholder.order[1]).fractions
    location = placeholder.location
    exponent = placeholder.data.get("exponent", 1)
    if getattr(placeholder, "raised_spatially", False):
        exponent = 1

    values = np.atleast_2d([frac.raise_to(exponent)(location)
                            for frac in fractions])
    # TODO full 2d output should be taken care of here but not all information
    # is present for that
    if values.shape[0] > 1 and values.shape[1] > 1 and False:
        print("INFO: 2d input detected, probably some composed input "
              "was used!")
        zero_cnt = np.array([sum((row != 0).astype(int)) for row in values])
        if any(zero_cnt > 1):
            raise ValueError(
                    "Invalid input detected when processing fractions: {} {}"
                    "".format(fractions, zero_cnt))
        print("WARNING: Summing up dimensions")
        values = np.sum(values, axis=1, keepdims=True).T

    if isinstance(placeholder, FieldVariable):
        return Scalars(values,
                       target_term=dict(name="E",
                                        order=placeholder.order[0],
                                        exponent=placeholder.data["exponent"]),
                       target_form=placeholder.data["weight_lbl"])
    elif isinstance(placeholder, TestFunction):
        # target form doesn't matter, since the f vector is added independently
        return Scalars(values.T, target_term=dict(
            name="f",
            test_func_lbl=placeholder.data["func_lbl"],
            test_appr_lbl=placeholder.data["appr_lbl"]))
    else:
        raise NotImplementedError


def get_common_target(scalars):
    """
    Extracts the common target from list of scalars while making sure that
    targets are equivalent.

    Args:
        scalars (:py:class:`.Scalars`):

    Return:
        dict: Common target.
    """
    e_targets = [scalar.target_term for scalar in scalars
                 if scalar.target_term["name"] == "E"]
    if e_targets:
        if len(e_targets) == 1:
            return e_targets[0]

        # more than one E-target, check if all entries are identical
        for key in ["order", "exponent"]:
            entries = [tar[key] for tar in e_targets]
            if entries[1:] != entries[:-1]:
                raise ValueError("mismatch in target terms!")

        return e_targets[0]
    else:
        return dict(name="f")


def get_common_form(placeholders):
    """
    Extracts the common target form from a list of scalars while making sure
    that the given targets are equivalent.

    Args:
        placeholders: Placeholders with possibly differing target forms.

    Return:
        str: Common target form.
    """
    target_form = None
    for member in placeholders["scalars"]:
        if target_form is None:
            target_form = member.target_form
        elif member.target_form is not None:
            if target_form != member.target_form:
                raise ValueError("Variant target forms provided for "
                                 "single entry.")
            target_form = member.target_form

    return target_form


def evaluate_placeholder_function(placeholder, input_values):
    """
    Evaluate a given placeholder object, that contains functions.

    Args:
        placeholder: Instance of :py:class:`.FieldVariable`,
            :py:class:`.TestFunction` or :py:class:`.ScalarFunction`.
        input_values: Values to evaluate at.

    Return:
        :py:obj:`numpy.ndarray` of results.
    """
    if not isinstance(placeholder, (FieldVariable, TestFunction)):
        raise TypeError("Input Object not supported!")

    base = get_base(placeholder.data["func_lbl"]).derive(placeholder.order[1])
    return np.array([func(input_values) for func in base.fractions])

