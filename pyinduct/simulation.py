from __future__ import division
from abc import ABCMeta, abstractmethod
import numpy as np
import sympy as sp
from core import Function, sanitize_input, calculate_function_matrix_differential

__author__ = 'Stefan Ecklebe'


class TestFunction(object):
    """
    class that works as a placeholder for test-functions in an equation
    """
    def __init__(self, order=0):
        self.order = order


class Input(object):
    """
    class that works as a placeholder for the input of a system
    """
    pass


class Factor(object):
    """
    class that represents multiplicative terms with the systems field variable x(z, t).
    factors can be placeholders like TestFunction or input.
    since differentiation may occur, order can provide information about which derivative of the field variable the
    factor belongs to.
    """
    def __init__(self, order, factor, location=None):
        """
        :param : order tuple of temporal_order and spatial_order
        :param : factor
        :param : location
        """
        if not isinstance(order, tuple) or len(order) > 2:
            raise TypeError("order mus be 2-tuple of int.")
        if sum(order) > 2:
            raise ValueError("only derivatives of order one and two supported")
        if not isinstance(factor, (TestFunction, Input)):
            raise TypeError("Only scalars or Functions accepted")
        if location is not None:
            if location and not isinstance(location, (int, long, float)):
                raise TypeError("location must be a number")

        self.order = order
        self.factor = factor
        self.location = location

class TemporalFactor(Factor):
    def __init__(self, order, factor, location=None):
        Factor.__init__(self, (order, 0), factor, location=location)


class SpatialFactor(Factor):
    def __init__(self, order, factor, location=None):
        Factor.__init__(self, (0, order), factor, location=location)


class MixedFactor(Factor):
    def __init__(self, factor, location=None):
        Factor.__init__(self, (1, 1), factor, location=location)


class WeakEquationTerm:
    """
    base class for all accepted terms in a weak formulation
    """

    def __init__(self, scale):
        if not isinstance(scale, (int, long, float)):
            raise TypeError("only numbers allowed as scale.")

        self.scale = scale

class ScalarTerm(WeakEquationTerm):
    """
    class that represents a scalar term in a weak equation
    """
    def __init__(self, factor, scale=1.0):
        WeakEquationTerm.__init__(self, scale)
        if not isinstance(factor, Factor):
            raise TypeError("input not supported")
        self.factor = factor

class IntegralTerm(WeakEquationTerm):
    """
    Class that represents an integral term in a weak equation
    """
    def __init__(self, integrand, limits, scale=1.0):
        WeakEquationTerm.__init__(self, scale)
        if not isinstance(integrand, Factor):
            raise TypeError("integrand must be of type Factor, otherwise compute it yourself!")
        if integrand.location is not None:
            raise ValueError("cannot integrate if integrand has to be evaluated first.")
        if not isinstance(limits, tuple):
            raise TypeError("limits must be provided as tuple")

        self.integrand = integrand
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
        init_funcs = sanitize_input(initial_functions, Function)
        test_funcs = sanitize_input(test_functions, Function)

        if init_funcs.shape != test_funcs.shape:
            raise ValueError("dimensions of init- and test-functions do not match.")

        dim = init_funcs.shape[0]
        self._f = np.zeros((dim,))
        self._E = [np.zeros((dim, dim)) for i in range(3)]

        # fill elementary matrices
        self._interpret_terms()

        # convert into state space form

    def _interpret_terms(self):
        # handle each term
        for term in self._terms:
            if isinstance(term, ScalarTerm):
                pass
            elif isinstance(term, IntegralTerm):
                temp_order = term.integrand.order[0]
                spat_order = term.integrand.order[1]
                # TODO Input, Factor

                if isinstance(term.integrand.factor, TestFunction):
                    test_der_order = term.integrand.factor.order
                    result = term.scale * calculate_function_matrix_differential(self.init_funcs, self.test_funcs,
                                                                                 spat_order,
                                                                                 test_der_order)
                self._E[temp_order] += result

    def _handle_temporal_term(self, term):
        """
        converts terms that contain temporal derivatives into state space matrices
        :param term:
        :return:
        """
        result = None
        if isinstance(term, IntegralTerm):
            if isinstance(term.integrand, Factor):
                order = term.integrand.order
                if isinstance(term.integrand.factor, TestFunction):
                    test_der_order = term.integrand.factor.order
                    result = term.scale*calculate_function_matrix_differential(self.init_funcs, self.test_funcs, 0,
                                                                     test_der_order)
                # TODO Input, Factor

                self._E[order] += result



    def _handle_spatial_term(self, term):
        pass
