from __future__ import division
from abc import ABCMeta, abstractmethod
import numpy as np
import sympy as sp
#from pyinduct import core

__author__ = 'Stefan Ecklebe'

class TestFunction:
    """
    class that works as a placeholder for test-functions in an equation
    """
    pass


class Input:
    """
    class that works as a placeholder for the input of a system
    """
    pass

class DerivativeFactor:
    """
    class that represents a multiplicative terms such as: dx/dt(z, t) * factor at z=location
    factor can either be a scalar number or placeholder like TestFunction
    """
    def __init__(self, kind, order, factor, location=None):
        if kind not in ["spatial", "temporal", "spatial/temporal"]:
            raise ValueError("only temporal and spatial derivatives supported.")
        if not isinstance(order, int):
            raise TypeError("order mus be int.")
        if order < 1 or order > 2:
            raise ValueError("only derivatives of order one and two supported")
        if not isinstance(factor, (int, long, float, TestFunction, Input)):
            raise TypeError("Only scalars or Functions accepted")
        if location is not None:
            if location and not isinstance(location, (int, long, float)):
                raise TypeError("location must be a number")

        self._kind = kind
        self._order = order
        self._factor = factor
        self._location = location

class TemporalDerivativeFactor(DerivativeFactor):
    def __init__(self, order, factor):
        DerivativeFactor.__init__(self, "temporal", order, factor)


class SpatialDerivativeFactor(DerivativeFactor):
    def __init__(self, order, factor):
        DerivativeFactor.__init__(self, "spatial", order, factor)


class MixedDerivativeFactor(DerivativeFactor):
    def __init__(self, factor):
        DerivativeFactor.__init__(self, "spatial/temporal", 1, factor)


class WeakEquationTerm:
    """
    base class for all accepted terms in a weak formulation
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, term):
        pass

class ScalarTerm(WeakEquationTerm):
    """
    class that represents a scalar term in a weak equation
    """
    def __init__(self, term):
        if not isinstance(term, (TestFunction, Input, DerivativeFactor)):
            raise TypeError("input not supported")
        self._term = term

class IntegralTerm(WeakEquationTerm):
    """
    Class that represents an integral term in a weak equation
    """
    def __init__(self, integrand, limits):
        if not isinstance(integrand, DerivativeFactor):
            raise TypeError("Integrand must be of type DerivativeFactor, otherwise compute it yourself!")
        if not isinstance(limits, tuple):
            raise TypeError("limits must be provided as tuple")

        self._integrand = integrand
        self._limits = limits


class SpatialIntegralTerm(IntegralTerm):
    def __init__(self, integrand, limits):
        IntegralTerm.__init__(self, integrand, limits)


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
