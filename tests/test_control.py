from __future__ import division
import unittest
import numpy as np
from pyinduct import core as cr
from pyinduct import control as ct
from pyinduct import placeholder as ph
from pyinduct import utils as ut

__author__ = 'Stefan Ecklebe'


class CollocatedTestCase(unittest.TestCase):

    def setUp(self):

        interval = (0, 1)
        nodes, funcs = ut.cure_interval(cr.LagrangeFirstOrder, interval, 3)
        x_at1 = ph.FieldVariable(funcs, location=1)
        x_dt_at1 = ph.TemporalDerivedFieldVariable(funcs, 1, 1)
        x_dz_at0 = ph.SpatialDerivedFieldVariable(funcs, 1, 0)
        exp_at1 = ph.ScalarFunctions(cr.Function(np.exp), location=1)

        alpha = 2
        self.term1 = ph.ScalarTerm(x_dt_at1, 1 + alpha)
        self.term2 = ph.ScalarTerm(x_dz_at0, 2)
        self.term3 = ph.ScalarTerm(ph.Product(x_at1, exp_at1))

        self.weights = np.hstack([[[1], [1], [1]], [[2], [2], [2]]])

    def test_temp_term(self):
        law = ct.approximate_control_law([self.term1])
        res = law(self.weights)
        self.assertAlmostEqual(res, 6)

    def test_spat_term(self):
        law = ct.approximate_control_law([self.term2])
        res = law(self.weights)
        self.assertAlmostEqual(res, 0)

    def test_product_term(self):
        law = ct.approximate_control_law([self.term3])
        res = law(self.weights)
        self.assertAlmostEqual(res, 1*np.exp(1))


class ContinuousTestCase(unittest.TestCase):

    def setUp(self):
        interval = (0, 1)
        nodes, funcs = ut.cure_interval(cr.LagrangeFirstOrder, interval, 3)
        x = ph.FieldVariable(funcs)
        x_dt = ph.TemporalDerivedFieldVariable(funcs, 1)
        x_dz = ph.SpatialDerivedFieldVariable(funcs, 1)
        exp = ph.ScalarFunctions(cr.Function(np.exp))

        alpha = 2
        self.term1 = ph.IntegralTerm(x_dt, interval, 1 + alpha)
        self.term2 = ph.IntegralTerm(x_dz, interval, 2)
        self.term3 = ph.IntegralTerm(ph.Product(x, exp), interval)

        self.weights = np.hstack([[[1], [1], [1]], [[2], [2], [2]]])

    def test_temp_term(self):
        law = ct.approximate_control_law([self.term1])
        res = law(self.weights)
        self.assertAlmostEqual(res, 6)

    def test_spat_term(self):
        law = ct.approximate_control_law([self.term2])
        res = law(self.weights)
        self.assertAlmostEqual(res, 0)

    def test_product_term(self):
        law = ct.approximate_control_law([self.term3])
        res = law(self.weights)
        # TODO calculate expected result
        # self.assertAlmostEqual(res, 1*np.exp(1))


class SimulationInteractionTestCase(unittest.TestCase):
    # TODO
    pass

