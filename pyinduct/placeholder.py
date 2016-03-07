from abc import ABCMeta
from numbers import Number

import numpy as np

from .registry import get_base, register_base, is_registered
from .core import sanitize_input
import collections


class Placeholder(object):
    """
    class that works as an placeholder for terms that are later substituted
    """
    def __init__(self, data, order=(0, 0), location=None):
        """
        :param order how many derivations are to be applied before evaluation (t, z)
        :param location to evaluate at before further computation
        """
        self.data = data

        if not isinstance(order, tuple) or any([not isinstance(o, int) or o < 0 for o in order]):
            raise ValueError("invalid derivative order.")
        self.order = order

        if location is not None:
            if location and not isinstance(location, Number):
                raise TypeError("location must be a number")
        self.location = location


class Scalars(Placeholder):
    """
    placeholder for scalars that will be replaced later
    """
    def __init__(self, values, target_term=None, target_form=None):
        values = np.atleast_2d(values)
        Placeholder.__init__(self, sanitize_input(values, Number))
        self.target_term = target_term
        self.target_form = target_form


class ScalarFunction(Placeholder):
    """
    class that works as a placeholder for spatial-functions in an equation such as spatial dependent coefficients
    """
    def __init__(self, function_label, order=0, location=None):
        if not is_registered(function_label):
            raise ValueError("Unknown function label '{0}'!".format(function_label))

        Placeholder.__init__(self, {"func_lbl": function_label}, (0, order), location)


class Input(Placeholder):
    """
    class that works as a placeholder for the input of a system
    """
    def __init__(self, function_handle, order=0):
        if not isinstance(function_handle, collections.Callable):
            raise TypeError("callable object has to be provided.")
        Placeholder.__init__(self, function_handle, order=(order, 0))


class TestFunction(Placeholder):
    """
    class that works as a placeholder for test-functions in an equation
    """
    def __init__(self, function_label, order=0, location=None):
        if not is_registered(function_label):
            raise ValueError("Unknown function label '{0}'!".format(function_label))

        Placeholder.__init__(self, {"func_lbl": function_label}, order=(0, order), location=location)


class FieldVariable(Placeholder):
    """
    class that represents terms of the systems field variable x(z, t).
    since differentiation may occur, order can provide information about which derivative of the field variable shall
    be used.
    """
    def __init__(self, function_label, order=(0, 0), weight_label=None, location=None):
        """
        :param : order tuple of temporal_order and spatial_order
        :param : factor
        :param : location
        """
        if not isinstance(order, tuple) or len(order) > 2:
            raise TypeError("order mus be 2-tuple of int.")
        if any([True for n in order if n < 0]):
            raise ValueError("derivative orders must be positive")
        if sum(order) > 2:
            raise ValueError("only derivatives of order one and two supported")
        if location is not None:
            if location and not isinstance(location, Number):
                raise TypeError("location must be a number")
        if not is_registered(function_label):
            raise ValueError("Unknown function label '{0}'!".format(function_label))
        if weight_label is None:
            weight_label = function_label
        elif not isinstance(weight_label, str):
            raise TypeError("only strings allowed as 'weight_label'")

        Placeholder.__init__(self, {"func_lbl": function_label, "weight_lbl": weight_label},
                             order=order, location=location)


class TemporalDerivedFieldVariable(FieldVariable):
    def __init__(self, function_label, order, weight_label=None, location=None):
        FieldVariable.__init__(self, function_label, (order, 0), weight_label, location)


class SpatialDerivedFieldVariable(FieldVariable):
    def __init__(self, function_label, order, weight_label=None, location=None):
        FieldVariable.__init__(self, function_label, (0, order), weight_label, location)


# TODO: remove
class MixedDerivedFieldVariable(FieldVariable):
    def __init__(self, function_label, weight_label=None, location=None):
        FieldVariable.__init__(self, function_label, (1, 1), weight_label, location)


class Product(object):
    """
    represents a product
    """
    def __init__(self, a, b=None):
        # convenience: accept single arguments
        if b is None:  # multiply by one as Default
            self.b_empty = True
            if isinstance(a, Input):
                b = Scalars(np.ones(1))
            if isinstance(a, Scalars):
                if a.target_term[0] == "E":
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
        if not isinstance(a, Placeholder) or (b is not None and not isinstance(b, Placeholder)):
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
                # print("WARNING: converting Placeholder that is to be evaluated into 'Scalars' object.")
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
                if isinstance(obj2, (FieldVariable, TestFunction, ScalarFunction)):
                    other_func = obj2
                    break

        if scalar_func and other_func:
            s_func = get_base(scalar_func.data["func_lbl"], scalar_func.order[1])
            o_func = get_base(other_func.data["func_lbl"], other_func.order[1])

            if s_func.shape != o_func.shape:
                if s_func.shape[0] == 1:
                    # only one function provided, use it for all others
                    s_func = s_func[[0 for i in range(o_func.shape[0])]]
                else:
                    raise ValueError("Cannot simplify Product due to dimension mismatch!")

            new_func = np.asarray([func.scale(scale_func) for func, scale_func in zip(o_func, s_func)])
            new_name = new_func.tostring()
            new_name = new_func.tobytes()
            register_base(new_name, new_func)

            if isinstance(other_func, (ScalarFunction, TestFunction)):
                a = other_func.__class__(function_label=new_name, order=other_func.order[1],
                                         location=other_func.location)
            elif isinstance(other_func, FieldVariable):
                # overwrite spatial derivative order, since derivation has been performed
                a = FieldVariable(function_label=new_name, weight_label=other_func.data["weight_lbl"],
                                  order=(other_func.order[0], 0), location=other_func.location)
            b = None

        return a, b

    def get_arg_by_class(self, cls):
        """
        extract element from product that is an instance of cls
        :return:
        """
        return [elem for elem in self.args if isinstance(elem, cls)]


class EquationTerm(object, metaclass=ABCMeta):
    """
    base class for all accepted terms in a weak formulation
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
    class that represents a scalar term in a weak equation
    """
    def __init__(self, argument, scale=1.0):
        EquationTerm.__init__(self, scale, argument)

        if any([True for arg in self.arg.args if isinstance(arg, (FieldVariable, TestFunction))]):
            raise ValueError("cannot leave z dependency. specify location to evaluate expression.")


class IntegralTerm(EquationTerm):
    """
    Class that represents an integral term in a weak equation
    """
    def __init__(self, integrand, limits, scale=1.0):
        EquationTerm.__init__(self, scale, integrand)

        if not any([isinstance(arg, (FieldVariable, TestFunction)) for arg in self.arg.args]):
            raise ValueError("nothing to integrate")
        if not isinstance(limits, tuple):
            raise TypeError("limits must be provided as tuple")
        self.limits = limits

# TODO: delete!
class SpatialIntegralTerm(IntegralTerm):
    def __init__(self, integrand, limits, scale=1.0):
        IntegralTerm.__init__(self, integrand, limits, scale)


def _evaluate_placeholder(placeholder):
    """
    evaluates a placeholder object and returns a Scalars object

    :param placeholder:
    :return:
    """
    if not isinstance(placeholder, Placeholder):
        raise TypeError("only placeholders supported")
    if isinstance(placeholder, (Scalars, Input)):
        raise TypeError("provided type cannot be evaluated")

    functions = get_base(placeholder.data['func_lbl'], placeholder.order[1])
    location = placeholder.location
    values = np.atleast_2d([func(location) for func in functions])

    if isinstance(placeholder, FieldVariable):
        return Scalars(values, target_term=("E", placeholder.order[0]), target_form=placeholder.data["weight_lbl"])
    elif isinstance(placeholder, TestFunction):
        # target form does not matter, since the f vector is added independently
        return Scalars(values.T, target_term=("f", 0))
    else:
        raise NotImplementedError


def get_scalar_target(scalars):
    """
    extract target from list of scalars.
    makes sure that targets are equivalent.

    :param scalars:
    :return:
    """
    targets = [elem for elem in [getattr(ph, "target_term", None) for ph in scalars] if elem]
    if targets:
        if targets[1:] != targets[:-1]:
            # since scalars are evaluated separately prefer E for f
            residual = [x for x in targets if x[0] != "f"]
            if len(residual) > 1:
                # different temporal derivatives of state -> not supported
                raise ValueError("target_term of scalars in product must be identical")
        return targets[0]

    return None
