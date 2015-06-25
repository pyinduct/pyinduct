from __future__ import division
from abc import ABCMeta, abstractmethod
import numpy as np
import sympy as sp
from core import Function, sanitize_input, calculate_function_matrix_differential

__author__ = 'Stefan Ecklebe'

class Placeholder(object):
    """
    class that works as an placeholder for functions that are later substituted
    """
    def __init__(self, location=None):
        if location is not None:
            if location and not isinstance(location, (int, long, float)):
                raise TypeError("location must be a number")
        self.location = location

class TestFunction(Placeholder):
    """
    class that works as a placeholder for test-functions in an equation
    """
    def __init__(self, order=0, location=None):
        Placeholder.__init__(self, location)
        self.order = order


class Input(Placeholder):
    """
    class that works as a placeholder for the input of a system
    """
    def __init__(self, order=0):
        Placeholder.__init__(self)
        self.order = order


class FieldVariable(object):
    """
    class that represents terms of the systems field variable x(z, t).
    since differentiation may occur, order can provide information about which derivative of the field variable.
    """
    def __init__(self, order, location=None):
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
            if location and not isinstance(location, (int, long, float)):
                raise TypeError("location must be a number")

        self.order = order
        self.location = location

# TODO add IndexedFactor, to represent fitting factor for every TestFunction

class TemporalDerivedFieldVariable(FieldVariable):
    def __init__(self, order, location=None):
        FieldVariable.__init__(self, (order, 0), location=location)


class SpatialDerivedFieldVariable(FieldVariable):
    def __init__(self, order, location=None):
        FieldVariable.__init__(self, (0, order), location=location)


class MixedDerivedFieldVariable(FieldVariable):
    def __init__(self, location=None):
        FieldVariable.__init__(self, (1, 1), location=location)

class Product:
    """
    represents a product
    """
    def __init__(self, a, b):
        if not isinstance(a, (FieldVariable, Placeholder)) or not isinstance(b, (FieldVariable, Placeholder)):
            raise TypeError("argument not allowed in product")
        self.args = [a, b]

    def get_arg_by_class(self, cls):
        """
        extract element from product that is an instance of cls
        :return:
        """
        return [elem for elem in self.args if isinstance(elem, cls)]

class WeakEquationTerm:
    """
    base class for all accepted terms in a weak formulation
    """
    __metaclass__ = ABCMeta

    def __init__(self, scale, arg):
        if not isinstance(scale, (int, long, float)):
            raise TypeError("only numbers allowed as scale.")
        if not isinstance(arg, (Placeholder, FieldVariable, Product)):
            raise TypeError("argument not supported.")

        self.scale = scale
        self.arg = arg

class ScalarTerm(WeakEquationTerm):
    """
    class that represents a scalar term in a weak equation
    """
    def __init__(self, argument, scale=1.0):
        WeakEquationTerm.__init__(self, scale, argument)

class IntegralTerm(WeakEquationTerm):
    """
    Class that represents an integral term in a weak equation
    """
    def __init__(self, integrand, limits, scale=1.0):
        WeakEquationTerm.__init__(self, scale, integrand)

        # TODO recurse down all possibly nested products and look whether location is set
        if not isinstance(integrand, Product):
            if integrand.location is not None:
                raise ValueError("cannot integrate if integrand has to be evaluated first.")
        if not isinstance(limits, tuple):
            raise TypeError("limits must be provided as tuple")
        self.limits = limits


class SpatialIntegralTerm(IntegralTerm):
    def __init__(self, integrand, limits, scale=1.0):
        IntegralTerm.__init__(self, integrand, limits, scale)



class WeakFormulation:
    """
    this class represents the weak formulation of a spatial problem.
    It can be initialized with several terms (see children of WeakEquationTerm).
    The equation is interpreted as term_0 + term_1 + ... + term_N = 0

    :param terms: single (or list) of object(s) of type WeakEquationTerm
    """
    def __init__(self, terms):
        if isinstance(terms, WeakEquationTerm):
            terms = [terms]
        for term in terms:
            if not isinstance(term, WeakEquationTerm):
                raise TypeError("Only WeakEquationTerm(s) are accepted.")

        self._terms = terms
        self.init_funcs = None
        self.test_funcs = None
        self._E = None
        self._f = None

    def create_ode_system(self, initial_functions, test_functions):
        """
        creates an ode system for the weights x_i based on the weak formulation.
        General assumption is that x is expressed as generalized fourier series using initial_functions and weights x_i.
        :param initial_functions: functions that are used to construct the solution
        :param test_functions: functional base which is used to minimize error
        :return: simulation.ODESystem
        """
        self.init_funcs = sanitize_input(initial_functions, Function)
        self.test_funcs = sanitize_input(test_functions, Function)

        if self.init_funcs.shape != self.test_funcs.shape:
            raise ValueError("dimensions of init- and test-functions do not match.")

        dim = self.init_funcs.shape[0]
        self._f = np.zeros((dim,))
        self._E = [np.zeros((dim, dim)) for i in range(3)]

        # fill elementary matrices
        self._interpret_terms()

        # convert into state space form

    def _interpret_terms(self):
        # handle each term
        for term in self._terms:
            if isinstance(term, ScalarTerm):
                # TODO move cases from Product case into functions and add them below
                if isinstance(term.arg, Product):
                    funcs = term.arg.get_arg_by_class(TestFunction)
                    ders = term.arg.get_arg_by_class(FieldVariable)
                    ins = term.arg.get_arg_by_class(Input)

                    if len(ders) == 1:
                        temp_order = ders[0].order[0]
                        spat_order = ders[0].order[1]
                        der_loc = ders[0].location
                        # TODO handle Input as well
                        if len(funcs) == 1:
                            func_loc = funcs[0].location
                            if der_loc is None or func_loc is None:
                                raise ValueError("scalar term mus be evaluated, should be an integral otherwise.")
                            test_der_order = funcs[0].order
                            result = calculate_function_matrix_differential(self.init_funcs, self.test_funcs,
                                                                            spat_order, test_der_order,
                                                                            locations=(der_loc, func_loc))
                        else:
                            raise NotImplementedError
                        self._E[temp_order] += result*term.scale
                    elif len(ins) == 1:
                        # since product contains two elements and a FieldDerivative was not in, other one is TestFunc
                        assert len(funcs) == 1
                        func_loc = funcs[0].location
                        if func_loc is None:
                            raise ValueError("scalar term mus be evaluated, should be an integral otherwise.")
                        test_der_order = funcs[0].order
                        result = np.asarray([func.derivative(test_der_order)(func_loc) for func in self.test_funcs])
                        self._f += result*term.scale
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError

            elif isinstance(term, IntegralTerm):
                # TODO move cases from Product case into functions and add them below
                if isinstance(term.arg, Product):
                    funcs = term.arg.get_arg_by_class(TestFunction)
                    ders = term.arg.get_arg_by_class(FieldVariable)
                    ins = term.arg.get_arg_by_class(Input)

                    if len(ders) == 1:
                        temp_order = ders[0].order[0]
                        spat_order = ders[0].order[1]
                        # TODO handle Input as well
                        if len(funcs) == 1:
                            test_der_order = funcs[0].order
                            result = calculate_function_matrix_differential(self.init_funcs, self.test_funcs,
                                                                            spat_order, test_der_order)
                        else:
                            raise NotImplementedError
                        self._E[temp_order] += result*term.scale
                    else:
                        raise NotImplementedError

        print("f:")
        print(self._f)
        print("EO:")
        print(self._E[0])
        print("E1:")
        print(self._E[1])
        print("E2:")
        print(self._E[2])


