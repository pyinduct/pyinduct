from __future__ import division
import unittest
import numpy as np
import pyqtgraph as pg
from pyinduct import core as cr, simulation as sim, utils as ut

__author__ = 'Stefan Ecklebe'

# TODO Test for Placeholders

class FieldVariableTest(unittest.TestCase):

    def setUp(self):
        nodes, self.ini_funcs = ut.cure_interval(cr.LagrangeFirstOrder, (0, 1), node_count=2)

    def test_FieldVariable(self):
        # Factor (Base)
        self.assertRaises(TypeError, sim.FieldVariable, self.ini_funcs, [0, 0])  # list instead of tuple
        self.assertRaises(ValueError, sim.FieldVariable, self.ini_funcs, (3, 0))  # order too high
        self.assertRaises(ValueError, sim.FieldVariable, self.ini_funcs, (0, 3))  # order too high
        self.assertRaises(ValueError, sim.FieldVariable, self.ini_funcs, (2, 2))  # order too high
        self.assertRaises(ValueError, sim.FieldVariable, self.ini_funcs, (-1, 3))  # order negative
        a = sim.FieldVariable(self.ini_funcs, (0, 0), 7)
        self.assertEqual((0, 0), a.order)
        self.assertEqual(7, a.location)

    def test_TemporalDerivativeFactor(self):
        # TODO
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
    def setUp(self):
        self.input = sim.Input(np.sin)
        phi = cr.Function(np.sin)
        self.funcs = sim.TestFunctions(phi)
        nodes, self.ini_funcs = ut.cure_interval(cr.LagrangeFirstOrder, (0, 1), node_count=2)
        self.field_var = sim.FieldVariable(self.ini_funcs)

    def test_product(self):
        self.assertRaises(TypeError, sim.Product, cr.Function, cr.Function)  # only Placeholders allowed
        p1 = sim.Product(self.input, self.funcs)
        p2 = sim.Product(self.funcs, self.field_var)

        # test methods
        self.assertEqual(p1.get_arg_by_class(sim.Input), [self.input])
        self.assertEqual(p1.get_arg_by_class(sim.TestFunctions), [self.funcs])
        self.assertEqual(p2.get_arg_by_class(sim.TestFunctions), [self.funcs])
        self.assertEqual(p2.get_arg_by_class(sim.FieldVariable), [self.field_var])

class WeakTermsTest(unittest.TestCase):

    def setUp(self):
        self.input = sim.Input(np.sin)
        self.phi = cr.Function(lambda x: 2*x)
        self.test_func = sim.TestFunctions(self.phi)
        nodes, self.ini_funcs = ut.cure_interval(cr.LagrangeFirstOrder, (0, 1), node_count=2)
        self.xdt = sim.TemporalDerivedFieldVariable(self.ini_funcs, 1)
        self.xdz_at1 = sim.SpatialDerivedFieldVariable(self.ini_funcs, 1, 1)
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

    def test_init(self):
        self.assertRaises(TypeError, sim.WeakFormulation, ["a", "b"])
        sim.WeakFormulation(sim.ScalarTerm(sim.FieldVariable()))  # scalar case
        sim.WeakFormulation([sim.ScalarTerm(sim.FieldVariable()),
                             sim.IntegralTerm(sim.FieldVariable(), (0, 1))
                             ])  # vector case

class ParseTest(unittest.TestCase):

    def setUp(self):
        self.u = np.sin
        self.input = sim.Input(self.u)  # control input
        nodes, self.ini_funcs = ut.cure_interval(cr.LagrangeFirstOrder, (0, 1), node_count=3)
        self.phi = sim.TestFunctions(self.ini_funcs)  # eigenfunction or something else
        self.dphi = sim.TestFunctions(self.ini_funcs, order=1)  # eigenfunction or something else
        self.field_var = sim.FieldVariable(self.ini_funcs)
        self.field_var_dz = sim.SpatialDerivedFieldVariable(self.ini_funcs, 1, location=1)
        self.field_var_ddt = sim.TemporalDerivedFieldVariable(self.ini_funcs, 2)
        self.field_var_ddt_at1 = sim.TemporalDerivedFieldVariable(self.ini_funcs, 2, location=1)

        # create all possible kinds of input variables
        self.input_term = sim.ScalarTerm(sim.Product(self.field_var_dz, self.input))
        self.func_term = sim.ScalarTerm(self.phi)
        self.field_term = sim.ScalarTerm(self.field_var)
        self.field_term_dz = sim.ScalarTerm(self.field_var_dz)
        self.field_term_ddt_at1 = sim.ScalarTerm(self.field_var_ddt_at1)
        self.temp_int = sim.IntegralTerm(sim.Product(self.field_var_ddt, self.phi), (0, 1))
        self.spat_int = sim.IntegralTerm(sim.Product(self.field_var_dz, self.dphi), (0, 1))

    def test_Input_term(self):
        wf = sim.WeakFormulation(self.input_term)
        sim.parse_weak_formulation(wf)

    def test_TestFunction_term(self):
        wf = sim.WeakFormulation(self.func_term)
        sim.parse_weak_formulation(wf)

    def test_FieldVariable_term(self):
        wf = sim.WeakFormulation(self.field_term)
        sim.parse_weak_formulation(wf)

        wf = sim.WeakFormulation(self.field_term_ddt_at1)
        sim.parse_weak_formulation(wf)

    def test_Product_term(self):
        wf = sim.WeakFormulation(self.temp_int)
        sim.parse_weak_formulation(wf)

        wf = sim.WeakFormulation(self.spat_int)
        sim.parse_weak_formulation(wf)

    def test_modal_from(self):
        pass

class StateSpaceTests(unittest.TestCase):

    def setUp(self):
        # enter string with mass equations
        interval = (0, 1)
        int1 = sim.IntegralTerm(sim.Product(sim.TemporalDerivedFieldVariable(2), sim.TestFunctions()), interval)
        s1 = sim.ScalarTerm(sim.Product(sim.TemporalDerivedFieldVariable(2, location=0), sim.TestFunctions(location=0)))
        int2 = sim.IntegralTerm(sim.Product(sim.SpatialDerivedFieldVariable(1), sim.TestFunctions(order=1)), interval)
        s2 = sim.ScalarTerm(sim.Product(sim.Input(), sim.TestFunctions(location=1)), -1)
        nodes, ini_funcs = ut.cure_interval(cr.LagrangeFirstOrder, interval, node_count=3)
        string_pde = sim.WeakFormulation([int1, s1, int2, s2])
        self.e_mats, self.f_vec = string_pde.parse_input(ini_funcs, ini_funcs)

    def test_convert_to_state_space(self):
        A, B = sim.convert_to_state_space(self.e_mats, self.f_vec)
        self.assertTrue(np.allclose(A, np.array([[0, 0, 0, 1, 0, 0],
                                                [0, 0, 0, 0, 1, 0],
                                                [0, 0, 0, 0, 0, 1],
                                                [-2.25, 3, -.75, 0, 0, 0],
                                                [7.5, -18, 10.5, 0, 0, 0],
                                                [-3.75, 21, -17.25, 0, 0, 0]])))
        self.assertTrue(np.allclose(B, np.array([0, 0, 0, 0.125, -1.75, 6.875])))


class StringMassTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_it(self):
        # example case which the user will have to perform

        # enter string with mass equations
        interval = (0, 1)
        int1 = sim.IntegralTerm(sim.Product(sim.TemporalDerivedFieldVariable(2), sim.TestFunctions()), interval)
        s1 = sim.ScalarTerm(sim.Product(sim.TemporalDerivedFieldVariable(2, location=0), sim.TestFunctions(location=0)))
        int2 = sim.IntegralTerm(sim.Product(sim.SpatialDerivedFieldVariable(1), sim.TestFunctions(order=1)), interval)
        s2 = sim.ScalarTerm(sim.Product(sim.Input(), sim.TestFunctions(location=1)), -1)

        # cure interval
        nodes, ini_funcs = ut.cure_interval(cr.LagrangeFirstOrder, interval, node_count=3)

        # derive sate-space system
        string_pde = sim.WeakFormulation([int1, s1, int2, s2])
        e_mats, f_vec = string_pde.parse_input(ini_funcs, ini_funcs)
        # TODO assert check

        A, B = sim.convert_to_state_space(e_mats, f_vec)

        # derive initial conditions and simulate system
        def x0(z):
            return 0

        def input_handle(t):
            return np.sin(t)

        start_state = cr.Function(x0)
        initial_weights = cr.project_on_initial_functions(start_state, ini_funcs)
        q0 = np.zeros(2*len(initial_weights))
        q0[0:len(initial_weights)] = initial_weights
        sim.simulate_system(A, B, input_handle, q0, (0, 10))

class CanonicalFormTest(unittest.TestCase):

    def setUp(self):
        self.cf = sim.CanonicalForm()

    def test_add_to(self):
        a = np.eye(5)
        self.cf.add_to(("E", 0), a)
        self.assertTrue(np.array_equal(self.cf._E0, a))
        self.cf.add_to(("E", 0), 5*a)
        self.assertTrue(np.array_equal(self.cf._E0, 6*a))

        b = np.eye(10)
        self.assertRaises(ValueError, self.cf.add_to, ("E", 0), b)
        self.cf.add_to(("E", 2), b)
        self.assertTrue(np.array_equal(self.cf._E2, b))
        self.cf.add_to(("E", 2), 2*b)
        self.assertTrue(np.array_equal(self.cf._E2, 3*b))

        f = np.array(range(5))
        self.assertRaises(ValueError, self.cf.add_to, ("E", 0), f)
        self.cf.add_to(("f", 0), f)
        self.assertTrue(np.array_equal(self.cf._f0, f))
        self.cf.add_to(("f", 0), 2*f)
        self.assertTrue(np.array_equal(self.cf._f0, 3*f))

    def test_get_terms(self):
        self.cf.add_to(("E", 0), np.eye(5))
        self.cf.add_to(("E", 2), 5*np.eye(5))
        terms = self.cf.get_terms()
        self.assertTrue(np.array_equal(terms[0][0], np.eye(5)))
        self.assertTrue(np.array_equal(terms[0][1], np.zeros((5, 5))))
        self.assertTrue(np.array_equal(terms[0][2], 5*np.eye(5)))
        self.assertEqual(terms[1], None)
        self.assertEqual(terms[2], None)
