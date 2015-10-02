from __future__ import division
import unittest
import numpy as np
from pyinduct import register_initial_functions
from pyinduct import core as cr, simulation as sim, utils as ut, placeholder as ph

__author__ = 'Stefan Ecklebe'
# TODO Test for all Placeholders


class FieldVariableTest(unittest.TestCase):

    def setUp(self):
        nodes, ini_funcs = ut.cure_interval(cr.LagrangeFirstOrder, (0, 1), node_count=2)
        register_initial_functions("test_funcs", ini_funcs)

    def test_FieldVariable(self):
        self.assertRaises(TypeError, ph.FieldVariable, "test_funcs", [0, 0])  # list instead of tuple
        self.assertRaises(ValueError, ph.FieldVariable, "test_funcs", (3, 0))  # order too high
        self.assertRaises(ValueError, ph.FieldVariable, "test_funcs", (0, 3))  # order too high
        self.assertRaises(ValueError, ph.FieldVariable, "test_funcs", (2, 2))  # order too high
        self.assertRaises(ValueError, ph.FieldVariable, "test_funcs", (-1, 3))  # order negative
        a = ph.FieldVariable("test_funcs", (0, 0), location=7)
        self.assertEqual((0, 0), a.order)
        self.assertEqual("test_funcs", a.data["weight_lbl"])  # default weight label is function label
        self.assertEqual(7, a.location)

    def test_TemporalDerivativeFactor(self):
        # TODO add test cases again
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


class ProductTest(unittest.TestCase):

    def scale(self, z):
            return 2

    def a2(self, z):
            return 5*z

    def setUp(self):
        self.input = ph.Input(np.sin)
        self.phi = cr.Function(np.sin)
        self.psi = cr.Function(np.sin)
        register_initial_functions("funcs", np.array([self.phi, self.psi]))
        self.funcs = ph.TestFunction("funcs")
        self.scale_funcs = ph.ScalarFunctions(np.array([cr.Function(self.scale), cr.Function(self.scale)]))
        nodes, self.ini_funcs = ut.cure_interval(cr.LagrangeFirstOrder, (0, 1), node_count=2)
        register_initial_functions("ini_funcs", self.ini_funcs)
        self.field_var = ph.FieldVariable("ini_funcs")

    def test_product(self):
        self.assertRaises(TypeError, ph.Product, cr.Function, cr.Function)  # only Placeholders allowed
        p1 = ph.Product(self.input, self.funcs)
        p2 = ph.Product(self.funcs, self.field_var)

        # test single argument call
        p3 = ph.Product(self.funcs)
        self.assertAlmostEqual(p3.args[0].data[0](np.pi/2), 1)
        self.assertTrue(p3.b_empty)

        # test automated evaluation of Product with Scaled function
        p4 = ph.Product(self.field_var, self.scale_funcs)
        self.assertTrue(isinstance(p4.args[0], ph.Placeholder))
        self.assertEqual(p4.args[0].data[0](0), self.scale(0)*self.ini_funcs[0](0))
        self.assertEqual(p4.args[0].data[1](1), self.scale(1)*self.ini_funcs[1](1))
        self.assertEqual(p4.args[1], None)
        self.assertTrue(p4.b_empty)

        # test automated simplification of cascaded products
        p5 = ph.Product(ph.Product(self.field_var, self.scale_funcs),
                        ph.Product(self.funcs, self.scale_funcs))
        self.assertEqual(p5.args[0].data[0](0), self.scale(0)*self.ini_funcs[0](0))
        self.assertEqual(p5.args[0].data[1](1), self.scale(1)*self.ini_funcs[1](1))
        self.assertEqual(p5.args[1].data[0](0), self.scale(0)*self.phi(0))
        self.assertEqual(p5.args[1].data[1](1), self.scale(1)*self.psi(1))
        self.assertFalse(p5.b_empty)

        # test methods
        self.assertEqual(p1.get_arg_by_class(ph.Input), [self.input])
        self.assertEqual(p1.get_arg_by_class(ph.TestFunction), [self.funcs])
        self.assertEqual(p2.get_arg_by_class(ph.TestFunction), [self.funcs])
        self.assertEqual(p2.get_arg_by_class(ph.FieldVariable), [self.field_var])


class WeakTermsTest(unittest.TestCase):

    def setUp(self):
        self.input = ph.Input(np.sin)
        self.phi = cr.Function(lambda x: 2*x)
        self.test_func = ph.TestFunction(self.phi)
        nodes, self.ini_funcs = ut.cure_interval(cr.LagrangeFirstOrder, (0, 1), node_count=2)
        self.xdt = ph.TemporalDerivedFieldVariable(self.ini_funcs, 1)
        self.xdz_at1 = ph.SpatialDerivedFieldVariable(self.ini_funcs, 1, 1)
        self.prod = ph.Product(self.input, self.xdt)

    def test_WeakEquationTerm(self):
        self.assertRaises(TypeError, ph.EquationTerm, "eleven", self.input)  # scale is not a number
        self.assertRaises(TypeError, ph.EquationTerm, 1, cr.LagrangeFirstOrder(0, 1, 2))  # arg is invalid
        ph.EquationTerm(1, self.test_func)
        ph.EquationTerm(1, self.xdt)
        t1 = ph.EquationTerm(1, self.input)
        self.assertEqual(t1.scale, 1)
        self.assertEqual(t1.arg.args[0], self.input)  # automatically create Product object if only one arg is provided

    def test_ScalarTerm(self):
        self.assertRaises(TypeError, ph.ScalarTerm, 7)  # factor is number
        self.assertRaises(TypeError, ph.ScalarTerm, cr.Function(np.sin))  # factor is Function
        ph.ScalarTerm(self.input)
        self.assertRaises(ValueError, ph.ScalarTerm, self.test_func)  # integration has to be done
        t1 = ph.ScalarTerm(self.xdz_at1)
        self.assertEqual(t1.scale, 1.0)  # default scale
        # check if automated evaluation works
        self.assertTrue(np.allclose(t1.arg.args[0].data, np.array([0,  1])))

    def test_IntegralTerm(self):
        self.assertRaises(TypeError, ph.IntegralTerm, 7, (0, 1))  # integrand is number
        self.assertRaises(TypeError, ph.IntegralTerm, cr.Function(np.sin), (0, 1))  # integrand is Function
        self.assertRaises(ValueError, ph.IntegralTerm, self.xdz_at1, (0, 1))  # nothing left after evaluation
        self.assertRaises(TypeError, ph.IntegralTerm, self.xdt, [0, 1])  # limits is list

        ph.IntegralTerm(self.test_func, (0, 1))  # integrand is Placeholder
        self.assertRaises(ValueError, ph.IntegralTerm, self.input, (0, 1))  # nothing to do
        ph.IntegralTerm(self.xdt, (0, 1))  # integrand is Placeholder
        ph.IntegralTerm(self.prod, (0, 1))  # integrand is Product

        t1 = ph.IntegralTerm(self.xdt, (0, 1))
        self.assertEqual(t1.scale, 1.0)  # default scale
        self.assertEqual(t1.arg.args[0], self.xdt)  # automated product creation
        self.assertEqual(t1.limits, (0, 1))


class WeakFormulationTest(unittest.TestCase):

    def setUp(self):
        self.u = np.sin
        self.input = ph.Input(self.u)  # control input
        nodes, self.ini_funcs = ut.cure_interval(cr.LagrangeFirstOrder, (0, 1), node_count=3)
        self.phi = ph.TestFunction(self.ini_funcs)  # eigenfunction or something else
        self.dphi = ph.TestFunction(self.ini_funcs, order=1)  # eigenfunction or something else
        self.dphi_at1 = ph.TestFunction(self.ini_funcs, order=1, location=1)  # eigenfunction or something else
        self.field_var = ph.FieldVariable(self.ini_funcs)
        self.field_var_at1 = ph.FieldVariable(self.ini_funcs, location=1)

    def test_init(self):
        self.assertRaises(TypeError, sim.WeakFormulation, ["a", "b"])
        sim.WeakFormulation(ph.ScalarTerm(self.field_var_at1))  # scalar case
        sim.WeakFormulation([ph.ScalarTerm(self.field_var_at1),
                             ph.IntegralTerm(self.field_var, (0, 1))
                             ])  # vector case
