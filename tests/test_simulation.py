from __future__ import division
import unittest
import numpy as np
import pyqtgraph as pg
from pyinduct import core as cr, simulation as sim, utils as ut

__author__ = 'Stefan Ecklebe'


class TestWeakFormulation(unittest.TestCase):

    def setUp(self):
        pass

    def test_bricks(self):
        f = sim.TestFunction()
        i = sim.Input()

        # DerivativeFactor (Base)
        self.assertRaises(TypeError, sim.DerivativeFactor, "z")
        self.assertRaises(TypeError, sim.DerivativeFactor, "temporal", 1, "a")
        self.assertRaises(ValueError, sim.DerivativeFactor, "temporal", 0, 7)  # order between 1 and 2
        self.assertRaises(ValueError, sim.DerivativeFactor, "temporal", 3, 7)
        self.assertRaises(TypeError, sim.DerivativeFactor, "temporal", 1, 7, "over there")
        a = sim.DerivativeFactor("temporal", 1, 7, 10)
        self.assertEqual("temporal", a.kind)
        self.assertEqual(1, a.order)
        self.assertEqual(7, a.factor)
        self.assertEqual(10, a.location)

        # TemporalDerivativeFactor
        self.assertRaises(TypeError, sim.TemporalDerivativeFactor, "high order", "about five")
        self.assertRaises(ValueError, sim.TemporalDerivativeFactor, 3, 7)
        a = sim.TemporalDerivativeFactor(2, 7)
        self.assertEqual("temporal", a.kind)
        self.assertEqual(2, a.order)
        self.assertEqual(7, a.factor)
        self.assertEqual(None, a.location)

        # SpatialDerivativeFactor
        self.assertRaises(TypeError, sim.SpatialDerivativeFactor, "high order", "about five")
        self.assertRaises(ValueError, sim.SpatialDerivativeFactor, 3, 7)
        a = sim.SpatialDerivativeFactor(2, 7)
        self.assertEqual("spatial", a.kind)
        self.assertEqual(2, a.order)
        self.assertEqual(7, a.factor)
        self.assertEqual(None, a.location)

        # MixedDerivativeFactor
        self.assertRaises(TypeError, sim.MixedDerivativeFactor, "about five")
        a = sim.MixedDerivativeFactor(7)
        self.assertEqual("spatial/temporal", a.kind)
        self.assertEqual(1, a.order)
        self.assertEqual(7, a.factor)
        self.assertEqual(None, a.location)

    def test_weak_terms(self):
        # simple number
        self.assertRaises(TypeError, sim.WeakEquationTerm, "eleven")
        t0 = sim.WeakEquationTerm(7)

        # scalar term
        self.assertRaises(TypeError, sim.ScalarTerm, 7)
        t1 = sim.ScalarTerm(sim.Input())
        t1 = sim.ScalarTerm(sim.TestFunction())
        t1 = sim.ScalarTerm(sim.SpatialDerivativeFactor(2, 1))
        t1 = sim.ScalarTerm(sim.TemporalDerivativeFactor(1, 1))
        self.assertRaises(TypeError, sim.ScalarTerm, cr.Function(np.sin))

        # integral term
        self.assertRaises(TypeError, sim.IntegralTerm, 7, [1, 2])
        self.assertRaises(TypeError, sim.IntegralTerm, sim.Input(), [1, 2])
        t2 = sim.IntegralTerm(sim.SpatialDerivativeFactor(1, 1), (0, 1))
        t2 = sim.IntegralTerm(sim.TemporalDerivativeFactor(1, 1), (0, 1))
        t2 = sim.IntegralTerm(sim.MixedDerivativeFactor(1), (0, 1))
        self.assertRaises(TypeError, sim.IntegralTerm, cr.Function(np.sin))

    def test_weak_form(self):
        u = sim.ScalarTerm(sim.Input())
        phi = sim.ScalarTerm(sim.TestFunction())
        int1 = sim.IntegralTerm(sim.TemporalDerivativeFactor(1, 1), (0, 1))

        self.assertRaises(TypeError, sim.WeakFormulation, ["a", "b"])
        wf = sim.WeakFormulation(u)
        wf = sim.WeakFormulation([u, phi, int1])

        # ok, so now lets test with a real equation
        interval = (0, 1)
        terms = [sim.IntegralTerm(sim.TemporalDerivativeFactor(2, sim.TestFunction()), interval),
                 sim.ScalarTerm(sim.TemporalDerivativeFactor(2, sim.TestFunction(), 1)),
                 sim.IntegralTerm(sim.SpatialDerivativeFactor(2, sim.TestFunction()), interval),
                 sim.ScalarTerm(sim.SpatialDerivativeFactor(2, sim.TestFunction(), 1), -1),
                 ]
        nodes, ini_funcs = ut.cure_interval(cr.LagrangeFirstOrder, interval, node_count=10)
        string_pde = sim.WeakFormulation(terms)
        string_pde.create_ode_system(ini_funcs, ini_funcs)
