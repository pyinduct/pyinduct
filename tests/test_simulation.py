from __future__ import division
import unittest
import numpy as np
import pyqtgraph as pg
from pyinduct import core as cr, simulation as sim

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
        self.assertEqual("temporal", a._kind)
        self.assertEqual(1, a._order)
        self.assertEqual(7, a._factor)
        self.assertEqual(10, a._location)

        # TemporalDerivativeFactor
        self.assertRaises(TypeError, sim.TemporalDerivativeFactor, "high order", "about five")
        self.assertRaises(ValueError, sim.TemporalDerivativeFactor, 3, 7)
        a = sim.TemporalDerivativeFactor(2, 7)
        self.assertEqual("temporal", a._kind)
        self.assertEqual(2, a._order)
        self.assertEqual(7, a._factor)
        self.assertEqual(None, a._location)

        # SpatialDerivativeFactor
        self.assertRaises(TypeError, sim.SpatialDerivativeFactor, "high order", "about five")
        self.assertRaises(ValueError, sim.SpatialDerivativeFactor, 3, 7)
        a = sim.SpatialDerivativeFactor(2, 7)
        self.assertEqual("spatial", a._kind)
        self.assertEqual(2, a._order)
        self.assertEqual(7, a._factor)
        self.assertEqual(None, a._location)

        # MixedDerivativeFactor
        self.assertRaises(TypeError, sim.MixedDerivativeFactor, "about five")
        a = sim.MixedDerivativeFactor(7)
        self.assertEqual("spatial/temporal", a._kind)
        self.assertEqual(1, a._order)
        self.assertEqual(7, a._factor)
        self.assertEqual(None, a._location)

    def test_weak_terms(self):
        # abstract base
        self.assertRaises(TypeError, sim.WeakEquationTerm, 7)

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
