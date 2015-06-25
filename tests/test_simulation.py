from __future__ import division
import unittest
import numpy as np
import pyqtgraph as pg
from pyinduct import core as cr, simulation as sim, utils as ut

__author__ = 'Stefan Ecklebe'


class TestFactor(unittest.TestCase):

    def setUp(self):
        pass

    def test_factor(self):
        f = sim.TestFunction()
        i = sim.Input()

        # Factor (Base)
        self.assertRaises(TypeError, sim.FieldVariable, [0, 0])  # list instead of tuple
        self.assertRaises(ValueError, sim.FieldVariable, (3, 0))  # order too high
        self.assertRaises(ValueError, sim.FieldVariable, (0, 3))  # order too high
        self.assertRaises(ValueError, sim.FieldVariable, (2, 2))  # order too high
        self.assertRaises(ValueError, sim.FieldVariable, (-1, 3))  # order negative
        a = sim.FieldVariable((0, 0), 7)
        self.assertEqual((0, 0), a.order)
        self.assertEqual(7, a.location)

    def test_TemporalDerivativeFactor(self):
        pass
        # # TemporalDerivativeFactor
        # self.assertRaises(TypeError, sim.TemporalDerivativeFactor, "high order", "about five")
        # self.assertRaises(ValueError, sim.TemporalDerivativeFactor, 3, 7)
        # a = sim.TemporalDerivativeFactor(2, 7)
        # self.assertEqual("temporal", a.kind)
        # self.assertEqual(2, a.order)
        # self.assertEqual(7, a.factor)
        # self.assertEqual(None, a.location)
        #
        # # SpatialDerivativeFactor
        # self.assertRaises(TypeError, sim.SpatialDerivativeFactor, "high order", "about five")
        # self.assertRaises(ValueError, sim.SpatialDerivativeFactor, 3, 7)
        # a = sim.SpatialDerivativeFactor(2, 7)
        # self.assertEqual("spatial", a.kind)
        # self.assertEqual(2, a.order)
        # self.assertEqual(7, a.factor)
        # self.assertEqual(None, a.location)
        #
        # # MixedDerivativeFactor
        # self.assertRaises(TypeError, sim.MixedDerivativeFactor, "about five")
        # a = sim.MixedDerivativeFactor(7)
        # self.assertEqual("spatial/temporal", a.kind)
        # self.assertEqual(1, a.order)
        # self.assertEqual(7, a.factor)
        # self.assertEqual(None, a.location)

    def test_product(self):
        self.assertRaises(TypeError, sim.Product, cr.Function, cr.Function)
        sim.Product(sim.Input(), sim.Input())
        sim.Product(sim.Input(), sim.TestFunction())
        sim.Product(sim.Input(), sim.TemporalDerivedFieldVariable(1))

class WeakTermsTest(unittest.TestCase):

    def setUp(self):
        self.input = sim.Input()
        self.test_func = sim.TestFunction()
        self.xdt = sim.TemporalDerivedFieldVariable(1)
        self.xdz_at1 = sim.SpatialDerivedFieldVariable(1, 1)
        self.prod = sim.Product(self.input, self.xdt)

    def test_WeakEquationTerm(self):
        self.assertRaises(TypeError, sim.WeakEquationTerm, "eleven", self.input)  # scale is not a number
        self.assertRaises(TypeError, sim.WeakEquationTerm, 1, cr.LagrangeFirstOrder(0, 1, 2))  # arg is invalid
        sim.WeakEquationTerm(1, self.test_func)
        sim.WeakEquationTerm(1, self.xdt)
        t1 = sim.WeakEquationTerm(1, self.input)
        self.assertEqual(t1.scale, 1)
        self.assertEqual(t1.arg, self.input)

    def test_ScalarTerm(self):
        self.assertRaises(TypeError, sim.ScalarTerm, 7)  # factor is number
        self.assertRaises(TypeError, sim.ScalarTerm, cr.Function(np.sin))  # factor is Function
        sim.ScalarTerm(self.input)
        sim.ScalarTerm(self.test_func)
        t1 = sim.ScalarTerm(self.xdt)
        self.assertEqual(t1.scale, 1.0)  # default scale
        self.assertEqual(t1.arg, self.xdt)

    def test_IntegralTerm(self):
        self.assertRaises(TypeError, sim.IntegralTerm, 7, (0, 1))  # integrand is number
        self.assertRaises(TypeError, sim.IntegralTerm, cr.Function(np.sin), (0, 1))  # integrand is Function
        self.assertRaises(ValueError, sim.IntegralTerm, self.xdz_at1, (0, 1))  # integrand is to be evaluated
        self.assertRaises(TypeError, sim.IntegralTerm, self.xdt, [0, 1])  # limits is list

        sim.IntegralTerm(self.test_func, (0, 1))  # integrand is Placeholder
        sim.IntegralTerm(self.input, (0, 1))  # integrand is Placeholder
        sim.IntegralTerm(self.xdt, (0, 1))  # integrand is FieldVariable
        sim.IntegralTerm(self.prod, (0, 1))  # integrand is Product

        t1 = sim.IntegralTerm(self.xdt, (0, 1))
        self.assertEqual(t1.scale, 1.0)  # default scale
        self.assertEqual(t1.arg, self.xdt)
        self.assertEqual(t1.limits, (0, 1))

class WeakFormulationTest(unittest.TestCase):

    def setUp(self):
        self.u = sim.Input()
        self.input_term = sim.ScalarTerm(sim.Product(sim.FieldVariable((0, 0)), self.u))
        self.phi = sim.ScalarTerm(sim.TestFunction())
        self.int1 = sim.IntegralTerm(sim.Product(sim.TemporalDerivedFieldVariable(2), sim.TestFunction()), (0, 1))
        self.int2 = sim.IntegralTerm(sim.Product(sim.SpatialDerivedFieldVariable(1), sim.TestFunction(order=1)), (0, 1))

    def test_weak_form(self):
        self.assertRaises(TypeError, sim.WeakFormulation, ["a", "b"])
        wf = sim.WeakFormulation(self.input_term)
        wf = sim.WeakFormulation([self.input_term, self.phi, self.int1, self.int2])
        func = cr.Function(np.sin, derivative_handles=[np.cos], domain=(0, 1))
        # wf.create_ode_system(func, func)

    def test_WeakFormulation_string(self):
        # enter string with mass equations
        interval = (0, 1)
        int1 = sim.IntegralTerm(sim.Product(sim.TemporalDerivedFieldVariable(2), sim.TestFunction()), interval)
        s1 = sim.ScalarTerm(sim.Product(sim.TemporalDerivedFieldVariable(2, location=0), sim.TestFunction(location=0)))
        int2 = sim.IntegralTerm(sim.Product(sim.SpatialDerivedFieldVariable(1), sim.TestFunction(order=1)), interval)
        s2 = sim.ScalarTerm(sim.Product(sim.Input(), sim.TestFunction(location=1)), -1)
        nodes, ini_funcs = ut.cure_interval(cr.LagrangeFirstOrder, interval, node_count=3)
        string_pde = sim.WeakFormulation([int1, s1, int2, s2])
        string_pde.create_ode_system(ini_funcs, ini_funcs)
