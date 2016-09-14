import os
import sys
import unittest
from pickle import dump

import numpy as np

from pyinduct import register_base, deregister_base, \
    eigenfunctions as ef, \
    core as cr, \
    simulation as sim, \
    utils as ut, \
    visualization as vis, \
    trajectory as tr, \
    placeholder as ph, \
    shapefunctions as sf

if any([arg in {'discover', 'setup.py', 'test'} for arg in sys.argv]):
    show_plots = False
else:
    # show_plots = True
    show_plots = False

if show_plots:
    import pyqtgraph as pg

    app = pg.QtGui.QApplication([])
else:
    app = None


# TODO Test for Domain


class SimpleInput(sim.SimulationInput):
    """
    the simplest input we can imagine
    """

    def _calc_output(self, **kwargs):
        return 0


class MonotonousInput(sim.SimulationInput):
    """
    an input that ramps up
    """

    def _calc_output(self, **kwargs):
        return dict(output=kwargs["time"])


class CorrectInput(sim.SimulationInput):
    """
    a diligent input
    """

    def _calc_output(self, **kwargs):
        if "time" not in kwargs:
            raise ValueError("mandatory key not found!")
        if "weights" not in kwargs:
            raise ValueError("mandatory key not found!")
        if "weight_lbl" not in kwargs:
            raise ValueError("mandatory key not found!")
        return dict(output=0)


class SimulationInputTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_abstract_funcs(self):
        # raise type error since abstract method is not implemented
        self.assertRaises(TypeError, sim.SimulationInput)

        # method implemented, should work
        u = SimpleInput()

    def test_call_arguments(self):
        a = np.eye(2, 2)
        b = np.array([[0], [1]])
        u = CorrectInput()
        ic = np.zeros((2, 1))
        ss = sim.StateSpace("test", {1: a}, {1: b}, input_handle=u)

        # if caller provides correct kwargs no exception should be raised
        res = sim.simulate_state_space(ss, ic, sim.Domain((0, 1), num=10))

    def test_storage(self):
        a = np.eye(2, 2)
        b = np.array([[0], [1]])
        u = MonotonousInput()
        ic = np.zeros((2, 1))
        ss = sim.StateSpace("test", a, b, input_handle=u)

        # run simulation to fill the internal storage
        domain = sim.Domain((0, 10), step=.1)
        res = sim.simulate_state_space(ss, ic, domain)

        # don't return entries that are not there
        self.assertRaises(KeyError, u.get_results, domain, "Unknown Entry")

        # default key is "output"
        ed = u.get_results(domain)
        ed_explicit = u.get_results(domain, result_key="output")
        self.assertTrue(np.array_equal(ed, ed_explicit))

        # return np.ndarray as default
        self.assertIsInstance(ed, np.ndarray)

        # return EvalData if corresponding flag is set
        self.assertIsInstance(u.get_results(domain, as_eval_data=True), sim.EvalData)

        # TODO interpolation methods and extrapolation errors


class CanonicalFormTest(unittest.TestCase):
    def setUp(self):
        self.cf = sim.CanonicalForm()
        self.u = SimpleInput()

    def test_add_to(self):
        a = np.eye(5)
        self.cf.add_to(dict(name="E", order=0, exponent=1), a)
        self.assertTrue(np.array_equal(self.cf._matrices["E"][0][1], a))
        self.cf.add_to(dict(name="E", order=0, exponent=1), 5 * a)
        self.assertTrue(np.array_equal(self.cf._matrices["E"][0][1], 6 * a))

        b = np.eye(10)
        self.assertRaises(ValueError, self.cf.add_to, dict(name="E", order=0, exponent=1), b)
        self.cf.add_to(dict(name="E", order=2, exponent=1), b)
        self.assertTrue(np.array_equal(self.cf._matrices["E"][2][1], b))

        f = np.atleast_2d(np.array(range(5))).T
        self.assertRaises(ValueError, self.cf.add_to, dict(name="E", order=0, exponent=1), f)
        self.cf.add_to(dict(name="f", order=None, exponent=None), f)
        self.assertTrue(np.array_equal(self.cf._matrices["f"], f))
        # try to add something with derivative or exponent to f: value should end up in f
        self.cf.add_to(dict(name="f", order=None, exponent=None), f)
        self.assertTrue(np.array_equal(self.cf._matrices["f"], 2 * f))

        c = np.atleast_2d(np.array(range(5))).T
        # that one should be easy
        self.cf.add_to(dict(name="G", order=0, exponent=1), c, column=0)
        self.assertTrue(np.array_equal(self.cf._matrices["G"][0][1], c))

        # here G01 as to be expanded
        self.cf.add_to(dict(name="G", order=0, exponent=1), c, column=1)
        self.assertTrue(np.array_equal(self.cf._matrices["G"][0][1], np.hstack((c, c))))

        # here G01 as to be expanded again
        self.cf.add_to(dict(name="G", order=0, exponent=1), c, column=3)
        self.assertTrue(np.array_equal(self.cf._matrices["G"][0][1], np.hstack((c, c, np.zeros_like(c), c))))


class ParseTest(unittest.TestCase):
    def setUp(self):
        # scalars
        self.scalars = ph.Scalars(np.vstack(list(range(3))))

        # inputs
        self.u = np.sin
        self.input = ph.Input(self.u)
        self.input_squared = ph.Input(self.u, exponent=2)

        # scale function
        register_base("heavi", cr.Function(lambda z: 0 if z < 0.5 else (0.5 if z == 0.5 else 1)), overwrite=True)

        nodes, self.ini_funcs = sf.cure_interval(sf.LagrangeFirstOrder,
                                                 (0, 1), node_count=3)

        # TestFunctions
        register_base("ini_funcs", self.ini_funcs, overwrite=True)
        self.phi = ph.TestFunction("ini_funcs")
        self.phi_at0 = ph.TestFunction("ini_funcs", location=0)
        self.phi_at1 = ph.TestFunction("ini_funcs", location=1)
        self.dphi = ph.TestFunction("ini_funcs", order=1)
        self.dphi_at1 = ph.TestFunction("ini_funcs", order=1, location=1)

        # FieldVars
        self.field_var = ph.FieldVariable("ini_funcs")
        self.field_var_squared = ph.FieldVariable("ini_funcs", exponent=2)

        self.odd_weight_field_var = ph.FieldVariable("ini_funcs", weight_label="special_weights")
        self.field_var_at1 = ph.FieldVariable("ini_funcs", location=1)
        self.field_var_at1_squared = ph.FieldVariable("ini_funcs", location=1, exponent=2)

        self.field_var_dz = ph.SpatialDerivedFieldVariable("ini_funcs", 1)
        self.field_var_dz_at1 = ph.SpatialDerivedFieldVariable("ini_funcs", 1, location=1)

        self.field_var_ddt = ph.TemporalDerivedFieldVariable("ini_funcs", 2)
        self.field_var_ddt_at0 = ph.TemporalDerivedFieldVariable("ini_funcs", 2, location=0)
        self.field_var_ddt_at1 = ph.TemporalDerivedFieldVariable("ini_funcs", 2, location=1)

        # create all possible kinds of input variables
        self.input_term1 = ph.ScalarTerm(ph.Product(self.phi_at1, self.input))
        self.input_term1_swapped = ph.ScalarTerm(ph.Product(self.input, self.phi_at1))
        self.input_term1_squared = ph.ScalarTerm(ph.Product(self.input_squared, self.phi_at1))

        self.input_term2 = ph.ScalarTerm(ph.Product(self.dphi_at1, self.input))
        self.func_term = ph.ScalarTerm(self.phi_at1)

        self.input_term3 = ph.IntegralTerm(ph.Product(self.phi, self.input), (0, 1))
        self.input_term3_swapped = ph.IntegralTerm(ph.Product(self.input, self.phi), (0, 1))
        self.input_term3_scaled = ph.IntegralTerm(
            ph.Product(ph.Product(ph.ScalarFunction("heavi"), self.phi), self.input), (0, 1))

        # same goes for field variables
        self.field_term_at1 = ph.ScalarTerm(self.field_var_at1)
        self.field_term_at1_squared = ph.ScalarTerm(self.field_var_at1_squared)
        self.field_term_dz_at1 = ph.ScalarTerm(self.field_var_dz_at1)
        self.field_term_ddt_at1 = ph.ScalarTerm(self.field_var_ddt_at1)

        self.field_int = ph.IntegralTerm(self.field_var, (0, 1))
        self.field_squared_int = ph.IntegralTerm(self.field_var_squared, (0, 1))
        self.field_dz_int = ph.IntegralTerm(self.field_var_dz, (0, 1))
        self.field_ddt_int = ph.IntegralTerm(self.field_var_ddt, (0, 1))

        self.prod_term_fs_at1 = ph.ScalarTerm(
            ph.Product(self.field_var_at1, self.scalars))
        self.prod_int_fs = ph.IntegralTerm(ph.Product(self.field_var, self.scalars), (0, 1))
        self.prod_int_f_f = ph.IntegralTerm(ph.Product(self.field_var, self.phi), (0, 1))
        self.prod_int_f_squared_f = ph.IntegralTerm(ph.Product(self.field_var_squared, self.phi), (0, 1))
        self.prod_int_f_f_swapped = ph.IntegralTerm(ph.Product(self.phi, self.field_var), (0, 1))

        self.prod_int_f_at1_f = ph.IntegralTerm(
            ph.Product(self.field_var_at1, self.phi), (0, 1))
        self.prod_int_f_at1_squared_f = ph.IntegralTerm(
            ph.Product(self.field_var_at1_squared, self.phi), (0, 1))

        self.prod_int_f_f_at1 = ph.IntegralTerm(
            ph.Product(self.field_var, self.phi_at1), (0, 1))
        self.prod_int_f_squared_f_at1 = ph.IntegralTerm(
            ph.Product(self.field_var_squared, self.phi_at1), (0, 1))

        self.prod_term_f_at1_f_at1 = ph.ScalarTerm(
            ph.Product(self.field_var_at1, self.phi_at1))
        self.prod_term_f_at1_squared_f_at1 = ph.ScalarTerm(
            ph.Product(self.field_var_at1_squared, self.phi_at1))

        self.prod_int_fddt_f = ph.IntegralTerm(
            ph.Product(self.field_var_ddt, self.phi), (0, 1))
        self.prod_term_fddt_at0_f_at0 = ph.ScalarTerm(
            ph.Product(self.field_var_ddt_at0, self.phi_at0))

        self.prod_term_f_at1_dphi_at1 = ph.ScalarTerm(
            ph.Product(self.field_var_at1, self.dphi_at1))

        self.temp_int = ph.IntegralTerm(ph.Product(self.field_var_ddt, self.phi), (0, 1))
        self.spat_int = ph.IntegralTerm(ph.Product(self.field_var_dz, self.dphi), (0, 1))
        self.spat_int_asymmetric = ph.IntegralTerm(
            ph.Product(self.field_var_dz, self.phi), (0, 1))

        self.alternating_weights_term = ph.IntegralTerm(self.odd_weight_field_var, (0, 1))

    def test_Input_term(self):
        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.input_term2)).get_terms()
        self.assertTrue(np.allclose(terms["G"][0][1], np.array([[0], [-2], [2]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.input_term1_squared)).get_terms()
        self.assertTrue(np.allclose(terms["G"][0][2], np.array([[0], [0], [1]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.input_term3)).get_terms()
        self.assertTrue(np.allclose(terms["G"][0][1], np.array([[.25], [.5], [.25]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.input_term3_swapped)).get_terms()
        self.assertTrue(np.allclose(terms["G"][0][1], np.array([[.25], [.5], [.25]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.input_term3_scaled)).get_terms()
        self.assertTrue(np.allclose(terms["G"][0][1], np.array([[.0], [.25], [.25]])))

    def test_TestFunction_term(self):
        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.func_term)).get_terms()
        self.assertTrue(np.allclose(terms["f"], np.array([[0], [0], [1]])))

    def test_FieldVariable_term(self):
        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.field_term_at1)).get_terms()
        self.assertTrue(np.allclose(terms["E"][0][1], np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.field_term_at1_squared)).get_terms()
        self.assertTrue(np.allclose(terms["E"][0][2], np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.field_int)).get_terms()
        self.assertTrue(np.allclose(terms["E"][0][1], np.array([[0.25, 0.5, 0.25], [0.25, 0.5, 0.25], [.25, .5, .25]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.field_squared_int)).get_terms()
        self.assertTrue(np.allclose(terms["E"][0][2],
                                    np.array([[1 / 6, 1 / 3, 1 / 6], [1 / 6, 1 / 3, 1 / 6], [1 / 6, 1 / 3, 1 / 6]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.field_term_dz_at1)).get_terms()
        self.assertTrue(np.allclose(terms["E"][0][1], np.array([[0, -2, 2], [0, -2, 2], [0, -2, 2]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.field_dz_int)).get_terms()
        self.assertTrue(np.allclose(terms["E"][0][1], np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.field_term_ddt_at1)).get_terms()
        self.assertTrue(np.allclose(terms["E"][2][1], np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.field_ddt_int)).get_terms()
        self.assertTrue(np.allclose(terms["E"][2][1], np.array([[0.25, 0.5, 0.25], [0.25, 0.5, 0.25], [.25, .5, .25]])))

    def test_Product_term(self):
        # TODO create test functionality that will automatically check if Case is also valid for swapped arguments

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.prod_term_fs_at1)).get_terms()
        self.assertTrue(np.allclose(terms["E"][0][1], np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.prod_int_fs)).get_terms()
        self.assertTrue(np.allclose(terms["E"][0][1], np.array([[0, 0, 0], [0.25, .5, .25], [.5, 1, .5]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.prod_int_f_f)).get_terms()
        self.assertTrue(
            np.allclose(terms["E"][0][1], np.array([[1 / 6, 1 / 12, 0], [1 / 12, 1 / 3, 1 / 12], [0, 1 / 12, 1 / 6]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.prod_int_f_squared_f)).get_terms()
        self.assertTrue(
            np.allclose(terms["E"][0][2], np.array([[1 / 8, 1 / 24, 0], [1 / 24, 1 / 4, 1 / 24], [0, 1 / 24, 1 / 8]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.prod_int_f_f_swapped)).get_terms()
        self.assertTrue(
            np.allclose(terms["E"][0][1], np.array([[1 / 6, 1 / 12, 0], [1 / 12, 1 / 3, 1 / 12], [0, 1 / 12, 1 / 6]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.prod_int_f_at1_f)).get_terms()
        self.assertTrue(np.allclose(terms["E"][0][1], np.array([[0, 0, 0.25], [0, 0, 0.5], [0, 0, .25]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.prod_int_f_at1_squared_f)).get_terms()
        self.assertTrue(np.allclose(terms["E"][0][2], np.array([[0, 0, 0.25], [0, 0, 0.5], [0, 0, .25]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.prod_int_f_f_at1)).get_terms()
        self.assertTrue(np.allclose(terms["E"][0][1], np.array([[0, 0, 0], [0, 0, 0], [0.25, 0.5, .25]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.prod_int_f_squared_f_at1)).get_terms()
        self.assertTrue(np.allclose(terms["E"][0][2], np.array([[0, 0, 0], [0, 0, 0], [1 / 6, 1 / 3, 1 / 6]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.prod_term_f_at1_f_at1)).get_terms()
        self.assertTrue(np.allclose(terms["E"][0][1], np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.prod_term_f_at1_squared_f_at1)).get_terms()
        self.assertTrue(np.allclose(terms["E"][0][2], np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])))

        # more complex terms
        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.prod_int_fddt_f)).get_terms()
        self.assertTrue(
            np.allclose(terms["E"][2][1], np.array([[1 / 6, 1 / 12, 0], [1 / 12, 1 / 3, 1 / 12], [0, 1 / 12, 1 / 6]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.prod_term_fddt_at0_f_at0)).get_terms()
        self.assertTrue(np.allclose(terms["E"][2][1], np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.spat_int)).get_terms()
        self.assertTrue(np.allclose(terms["E"][0][1], np.array([[2, -2, 0], [-2, 4, -2], [0, -2, 2]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.spat_int_asymmetric)).get_terms()
        self.assertTrue(np.allclose(terms["E"][0][1], np.array([[-.5, .5, 0], [-.5, 0, .5], [0, -.5, .5]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.prod_term_f_at1_dphi_at1)).get_terms()
        self.assertTrue(np.allclose(terms["E"][0][1], np.array([[0, 0, 0], [0, 0, -2], [0, 0, 2]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.input_term1)).get_terms()
        self.assertTrue(np.allclose(terms["G"][0][1], np.array([[0], [0], [1]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.input_term1_swapped)).get_terms()
        self.assertTrue(np.allclose(terms["G"][0][1], np.array([[0], [0], [1]])))

    def test_alternating_weights(self):
        self.assertRaises(ValueError, sim.parse_weak_formulation,
                          sim.WeakFormulation([self.alternating_weights_term, self.field_int]))

    def tearDown(self):
        deregister_base("heavi")
        deregister_base("ini_funcs")


class StateSpaceTests(unittest.TestCase):
    def setUp(self):
        self.u = CorrectInput()

        # setup temp and spat domain
        spat_domain = sim.Domain((0, 1), num=3)
        nodes, ini_funcs = sf.cure_interval(sf.LagrangeFirstOrder, spat_domain.bounds, node_count=3)
        register_base("init_funcs", ini_funcs, overwrite=True)

        # enter string with mass equations for testing
        int1 = ph.IntegralTerm(
            ph.Product(ph.TemporalDerivedFieldVariable("init_funcs", 2),
                       ph.TestFunction("init_funcs")), spat_domain.bounds)
        s1 = ph.ScalarTerm(
            ph.Product(ph.TemporalDerivedFieldVariable("init_funcs", 2, location=0),
                       ph.TestFunction("init_funcs", location=0)))
        int2 = ph.IntegralTerm(
            ph.Product(ph.SpatialDerivedFieldVariable("init_funcs", 1),
                       ph.TestFunction("init_funcs", order=1)), spat_domain.bounds)
        s2 = ph.ScalarTerm(
            ph.Product(ph.Input(self.u), ph.TestFunction("init_funcs", location=1)), -1)

        string_pde = sim.WeakFormulation([int1, s1, int2, s2])
        self.cf = sim.parse_weak_formulation(string_pde)
        self.ic = np.zeros((3, 2))

    def test_convert_to_state_space(self):
        ss = self.cf.convert_to_state_space()
        self.assertEqual(ss.A[1].shape, (6, 6))
        self.assertTrue(np.allclose(ss.A[1], np.array([[0, 0, 0, 1, 0, 0],
                                                       [0, 0, 0, 0, 1, 0],
                                                       [0, 0, 0, 0, 0, 1],
                                                       [-2.25, 3, -.75, 0, 0, 0],
                                                       [7.5, -18, 10.5, 0, 0, 0],
                                                       [-3.75, 21, -17.25, 0, 0, 0]])))
        self.assertEqual(ss.B[1].shape, (6, 1))
        self.assertTrue(np.allclose(ss.B[1], np.array([[0], [0], [0], [0.125], [-1.75], [6.875]])))
        self.assertEqual(self.cf.input_function, self.u)

    def tearDown(self):
        deregister_base("init_funcs")


class StringMassTest(unittest.TestCase):
    def setUp(self):

        z_start = 0
        z_end = 1
        z_step = 0.1
        self.dz = sim.Domain(bounds=(z_start, z_end), num=9)

        t_start = 0
        t_end = 10
        t_step = 0.01
        self.dt = sim.Domain(bounds=(t_start, t_end), step=t_step)

        self.params = ut.Parameters
        self.params.node_distance = 0.1
        self.params.m = 1.0
        self.params.order = 8
        self.params.sigma = 1
        self.params.tau = 1

        self.y_end = 10

        self.u = tr.FlatString(0, self.y_end, z_start, z_end, 0, 5, self.params)

        def x(z, t):
            """
            initial conditions for testing
            """
            return 0

        def x_dt(z, t):
            """
            initial conditions for testing
            """
            return 0

        # initial conditions
        self.ic = np.array([
            cr.Function(lambda z: x(z, 0)),  # x(z, 0)
            cr.Function(lambda z: x_dt(z, 0)),  # dx_dt(z, 0)
        ])

    def test_fem(self):
        """
        use best documented fem case to test all steps in simulation process
        """

        # enter string with mass equations
        # nodes, ini_funcs = sf.cure_interval(sf.LagrangeFirstOrder,
        nodes, ini_funcs = sf.cure_interval(sf.LagrangeSecondOrder,
                                            self.dz.bounds, node_count=11)
        register_base("init_funcs", ini_funcs, overwrite=True)
        int1 = ph.IntegralTerm(
            ph.Product(ph.TemporalDerivedFieldVariable("init_funcs", 2),
                       ph.TestFunction("init_funcs")), self.dz.bounds, scale=self.params.sigma * self.params.tau ** 2)
        s1 = ph.ScalarTerm(
            ph.Product(ph.TemporalDerivedFieldVariable("init_funcs", 2, location=0),
                       ph.TestFunction("init_funcs", location=0)), scale=self.params.m)
        int2 = ph.IntegralTerm(
            ph.Product(ph.SpatialDerivedFieldVariable("init_funcs", 1),
                       ph.TestFunction("init_funcs", order=1)), self.dz.bounds, scale=self.params.sigma)
        s2 = ph.ScalarTerm(
            ph.Product(ph.Input(self.u), ph.TestFunction("init_funcs", location=1)), -self.params.sigma)

        # derive sate-space system
        string_pde = sim.WeakFormulation([int1, s1, int2, s2], name="fem_test")
        self.cf = sim.parse_weak_formulation(string_pde)
        ss = self.cf.convert_to_state_space()

        # generate initial conditions for weights
        q0 = np.array([cr.project_on_base(self.ic[idx], ini_funcs) for idx in range(2)]).flatten()

        # simulate
        t, q = sim.simulate_state_space(ss, q0, self.dt)

        # calculate result data
        eval_data = []
        for der_idx in range(2):
            eval_data.append(
                sim.evaluate_approximation("init_funcs", q[:, der_idx * ini_funcs.size:(der_idx + 1) * ini_funcs.size],
                                           t, self.dz))
            eval_data[-1].name = "{0}{1}".format(self.cf.name, "_" + "".join(["d" for x in range(der_idx)])
                                                               + "t" if der_idx > 0 else "")

        # display results
        if show_plots:
            win = vis.PgAnimatedPlot(eval_data[:2], title="fem approx and derivative")
            win2 = vis.PgSurfacePlot(eval_data[0])
            app.exec_()

        # test for correct transition
        self.assertTrue(np.isclose(eval_data[0].output_data[-1, 0], self.y_end, atol=1e-3))

        # save some test data for later use
        root_dir = os.getcwd()
        if root_dir.split(os.sep)[-1] == "tests":
            res_dir = os.sep.join([os.getcwd(), "resources"])
        else:
            res_dir = os.sep.join([os.getcwd(), "tests", "resources"])

        if not os.path.isdir(res_dir):
            os.makedirs(res_dir)

        file_path = os.sep.join([res_dir, "test_data.res"])
        with open(file_path, "w+b") as f:
            dump(eval_data, f)

    def test_modal(self):
        order = 8

        def char_eq(w):
            return w * (np.sin(w) + self.params.m * w * np.cos(w))

        def phi_k_factory(freq, derivative_order=0):
            def eig_func(z):
                return np.cos(freq * z) - self.params.m * freq * np.sin(freq * z)

            def eig_func_dz(z):
                return -freq * (np.sin(freq * z) + self.params.m * freq * np.cos(freq * z))

            def eig_func_ddz(z):
                return freq ** 2 * (-np.cos(freq * z) + self.params.m * freq * np.sin(freq * z))

            if derivative_order == 0:
                return eig_func
            elif derivative_order == 1:
                return eig_func_dz
            elif derivative_order == 2:
                return eig_func_ddz
            else:
                raise ValueError

        # create eigenfunctions
        eig_frequencies = ut.find_roots(char_eq, n_roots=order, grid=np.arange(0, 1e3, 2), rtol=-2)
        print("eigenfrequencies:")
        print(eig_frequencies)

        # create eigen function vectors
        class SWMFunctionVector(cr.ComposedFunctionVector):
            """
            String With Mass Function Vector, necessary due to manipulated scalar product
            """

            @property
            def func(self):
                return self.members["funcs"][0]

            @property
            def scalar(self):
                return self.members["scalars"][0]

        eig_vectors = np.array([SWMFunctionVector(cr.Function(phi_k_factory(eig_frequencies[n]),
                                                              derivative_handles=[
                                                                  phi_k_factory(eig_frequencies[n], der_order)
                                                                  for der_order in range(1, 3)],
                                                              domain=self.dz.bounds,
                                                              nonzero=self.dz.bounds),
                                                  phi_k_factory(eig_frequencies[n])(0))
                                for n in range(order)])

        # normalize eigen vectors
        norm_eig_vectors = cr.normalize_base(eig_vectors)
        norm_eig_funcs = np.array([vec.func for vec in norm_eig_vectors])
        register_base("norm_eig_funcs", norm_eig_funcs, overwrite=True)

        norm_eig_funcs[0](1)

        # debug print eigenfunctions
        if 0:
            func_vals = []
            for vec in eig_vectors:
                func_vals.append(np.vectorize(vec.func)(self.dz))

            norm_func_vals = []
            for func in norm_eig_funcs:
                norm_func_vals.append(np.vectorize(func)(self.dz))

            clrs = ["r", "g", "b", "c", "m", "y", "k", "w"]
            for n in range(1, order + 1, len(clrs)):
                pw_phin_k = pg.plot(title="phin_k for k in [{0}, {1}]".format(n, min(n + len(clrs), order)))
                for k in range(len(clrs)):
                    if k + n > order:
                        break
                    pw_phin_k.plot(x=np.array(self.dz), y=norm_func_vals[n + k - 1], pen=clrs[k])

            app.exec_()

        # create terms of weak formulation
        terms = [ph.IntegralTerm(ph.Product(ph.FieldVariable("norm_eig_funcs", order=(2, 0)),
                                            ph.TestFunction("norm_eig_funcs")),
                                 self.dz.bounds, scale=-1),
                 ph.ScalarTerm(ph.Product(
                     ph.FieldVariable("norm_eig_funcs", order=(2, 0), location=0),
                     ph.TestFunction("norm_eig_funcs", location=0)),
                     scale=-1),
                 ph.ScalarTerm(ph.Product(ph.Input(self.u),
                                          ph.TestFunction("norm_eig_funcs", location=1))),
                 ph.ScalarTerm(
                     ph.Product(ph.FieldVariable("norm_eig_funcs", location=1),
                                ph.TestFunction("norm_eig_funcs", order=1, location=1)),
                     scale=-1),
                 ph.ScalarTerm(ph.Product(ph.FieldVariable("norm_eig_funcs", location=0),
                                          ph.TestFunction("norm_eig_funcs", order=1,
                                                          location=0))),
                 ph.IntegralTerm(ph.Product(ph.FieldVariable("norm_eig_funcs"),
                                            ph.TestFunction("norm_eig_funcs", order=2)),
                                 self.dz.bounds)]
        modal_pde = sim.WeakFormulation(terms, name="swm_lib-modal")
        eval_data = sim.simulate_system(modal_pde, self.ic, self.dt, self.dz, der_orders=(2, 0))

        # display results
        if show_plots:
            win = vis.PgAnimatedPlot(eval_data[0:2], title="modal approx and derivative")
            win2 = vis.PgSurfacePlot(eval_data[0])
            app.exec_()

        deregister_base("norm_eig_funcs")

        # test for correct transition
        self.assertTrue(np.isclose(eval_data[0].output_data[-1, 0], self.y_end, atol=1e-3))

    def tearDown(self):
        pass


class RadFemTrajectoryTest(unittest.TestCase):
    """
    Tests FEM simulation with cr.LagrangeFirstOrder and cr.LagrangeSecondOrder testfunctions
    and generic trajectory generator tr.RadTrajectory
    for the reaction-advection-diffusion equation.
    """

    def setUp(self):
        self.param = [2., -1.5, -3., 2., .5]
        self.a2, self.a1, self.a0, self.alpha, self.beta = self.param

        self.l = 1.
        spatial_disc = 11
        self.dz = sim.Domain(bounds=(0, self.l), num=spatial_disc)

        self.T = 1.
        temporal_disc = 2e2
        self.dt = sim.Domain(bounds=(0, self.T), num=temporal_disc)

        # create test functions
        self.nodes_1, self.ini_funcs_1 = sf.cure_interval(sf.LagrangeFirstOrder,
                                                          self.dz.bounds,
                                                          node_count=spatial_disc)
        register_base("init_funcs_1", self.ini_funcs_1, overwrite=True)
        self.nodes_2, self.ini_funcs_2 = sf.cure_interval(sf.LagrangeSecondOrder,
                                                          self.dz.bounds,
                                                          node_count=spatial_disc)
        register_base("init_funcs_2", self.ini_funcs_2, overwrite=True)

    @unittest.skip  # needs border homogenization to work
    def test_dd(self):
        # TODO adopt this test case
        # trajectory
        bound_cond_type = 'dirichlet'
        actuation_type = 'dirichlet'
        u = tr.RadTrajectory(self.l, self.T, self.param, bound_cond_type, actuation_type)

        # derive state-space system
        rad_pde = ut.get_parabolic_dirichlet_weak_form("init_funcs_2", "init_funcs_2", u, self.param, self.dz.bounds)
        cf = sim.parse_weak_formulation(rad_pde)
        ss = cf.convert_to_state_space()

        # simulate system
        t, q = sim.simulate_state_space(ss, np.zeros(self.ini_funcs_2.shape), self.dt)

        # display results
        if show_plots:
            eval_d = sim.evaluate_approximation("init_funcs_1", q, t, self.dz, spat_order=1)
            win1 = vis.PgAnimatedPlot([eval_d], title="Test")
            win2 = vis.PgSurfacePlot(eval_d)
            app.exec_()

        # TODO add Test here

    @unittest.skip  # needs border homogenization to work
    def test_dd(self):
        # TODO adopt this test case
        # trajectory
        bound_cond_type = 'robin'
        actuation_type = 'dirichlet'
        u = tr.RadTrajectory(self.l, self.T, self.param, bound_cond_type, actuation_type)

        # integral terms
        int1 = ph.IntegralTerm(ph.Product(ph.TemporalDerivedFieldVariable("init_funcs_2", order=1),
                                          ph.TestFunction("init_funcs_2", order=0)), self.dz.bounds)
        int2 = ph.IntegralTerm(ph.Product(ph.SpatialDerivedFieldVariable("init_funcs_2", order=0),
                                          ph.TestFunction("init_funcs_2", order=2)), self.dz.bounds, -self.a2)
        int3 = ph.IntegralTerm(ph.Product(ph.SpatialDerivedFieldVariable("init_funcs_2", order=1),
                                          ph.TestFunction("init_funcs_2", order=0)), self.dz.bounds, -self.a1)
        int4 = ph.IntegralTerm(ph.Product(ph.SpatialDerivedFieldVariable("init_funcs_2", order=0),
                                          ph.TestFunction("init_funcs_2", order=0)), self.dz.bounds, -self.a0)
        # scalar terms from int 2
        s1 = ph.ScalarTerm(ph.Product(ph.SpatialDerivedFieldVariable("init_funcs_2", order=1, location=self.l),
                                      ph.TestFunction("init_funcs_2", order=0, location=self.l)), -self.a2)
        s2 = ph.ScalarTerm(ph.Product(ph.SpatialDerivedFieldVariable("init_funcs_2", order=0, location=0),
                                      ph.TestFunction("init_funcs_2", order=0, location=0)), self.a2 * self.alpha)
        s3 = ph.ScalarTerm(ph.Product(ph.SpatialDerivedFieldVariable("init_funcs_2", order=0, location=0),
                                      ph.TestFunction("init_funcs_2", order=1, location=0)), -self.a2)
        s4 = ph.ScalarTerm(ph.Product(ph.Input(u),
                                      ph.TestFunction("init_funcs_2", order=1, location=self.l)), self.a2)

        # derive state-space system
        rad_pde = sim.WeakFormulation([int1, int2, int3, int4, s1, s2, s3, s4])
        cf = sim.parse_weak_formulation(rad_pde)
        ss = cf.convert_to_state_space()

        # simulate system
        t, q = sim.simulate_state_space(ss, np.zeros(self.ini_funcs_2.shape), self.dt)
        # TODO add test here

    def test_dr(self):
        # trajectory
        bound_cond_type = 'dirichlet'
        actuation_type = 'robin'
        u = tr.RadTrajectory(self.l, self.T, self.param, bound_cond_type, actuation_type)
        # integral terms
        int1 = ph.IntegralTerm(ph.Product(ph.TemporalDerivedFieldVariable("init_funcs_1", order=1),
                                          ph.TestFunction("init_funcs_1", order=0)), self.dz.bounds)
        int2 = ph.IntegralTerm(ph.Product(ph.SpatialDerivedFieldVariable("init_funcs_1", order=1),
                                          ph.TestFunction("init_funcs_1", order=1)), self.dz.bounds, self.a2)
        int3 = ph.IntegralTerm(ph.Product(ph.SpatialDerivedFieldVariable("init_funcs_1", order=0),
                                          ph.TestFunction("init_funcs_1", order=1)), self.dz.bounds, self.a1)
        int4 = ph.IntegralTerm(ph.Product(ph.SpatialDerivedFieldVariable("init_funcs_1", order=0),
                                          ph.TestFunction("init_funcs_1", order=0)), self.dz.bounds, -self.a0)
        # scalar terms from int 2
        s1 = ph.ScalarTerm(ph.Product(ph.SpatialDerivedFieldVariable("init_funcs_1", order=0, location=self.l),
                                      ph.TestFunction("init_funcs_1", order=0, location=self.l)), -self.a1)
        s2 = ph.ScalarTerm(ph.Product(ph.SpatialDerivedFieldVariable("init_funcs_1", order=0, location=self.l),
                                      ph.TestFunction("init_funcs_1", order=0, location=self.l)), self.a2 * self.beta)
        s3 = ph.ScalarTerm(ph.Product(ph.SpatialDerivedFieldVariable("init_funcs_1", order=1, location=0),
                                      ph.TestFunction("init_funcs_1", order=0, location=0)), self.a2)
        s4 = ph.ScalarTerm(ph.Product(ph.Input(u),
                                      ph.TestFunction("init_funcs_1", order=0, location=self.l)), -self.a2)
        # derive state-space system
        rad_pde = sim.WeakFormulation([int1, int2, int3, int4, s1, s2, s3, s4])
        cf = sim.parse_weak_formulation(rad_pde)
        ss = cf.convert_to_state_space()

        # simulate system
        t, q = sim.simulate_state_space(ss, np.zeros(self.ini_funcs_1.shape), self.dt)

        # check if (x'(0,t_end) - 1.) < 0.1
        self.assertLess(np.abs(self.ini_funcs_1[0].derive(1)(sys.float_info.min) * (q[-1, 0] - q[-1, 1])) - 1, 0.1)
        return

    def test_rr(self):
        # trajectory
        bound_cond_type = 'robin'
        actuation_type = 'robin'
        u = tr.RadTrajectory(self.l, self.T, self.param, bound_cond_type, actuation_type)
        # derive state-space system
        rad_pde = ut.get_parabolic_robin_weak_form("init_funcs_1", "init_funcs_1", u, self.param, self.dz.bounds)
        cf = sim.parse_weak_formulation(rad_pde)
        ss = cf.convert_to_state_space()

        # simulate system
        t, q = sim.simulate_state_space(ss, np.zeros(self.ini_funcs_1.shape), self.dt)

        # check if (x(0,t_end) - 1.) < 0.1
        self.assertLess(np.abs(self.ini_funcs_1[0].derive(0)(0) * q[-1, 0]) - 1, 0.1)
        return

    def test_rr_const_traj(self):
        """ check if simulation interface call tr.ConstantTrajectory properly """

        # const trajectory simulation call test
        u = tr.ConstantTrajectory(1)
        # derive state-space system
        rad_pde = ut.get_parabolic_robin_weak_form("init_funcs_1", "init_funcs_1", u, self.param, self.dz.bounds)
        cf = sim.parse_weak_formulation(rad_pde)
        ss = cf.convert_to_state_space()

        # simulate system
        t, q = sim.simulate_state_space(ss, np.zeros(self.ini_funcs_1.shape), self.dt)
        # TODO add a Test here
        return

    def tearDown(self):
        deregister_base("init_funcs_1")
        deregister_base("init_funcs_2")


class RadDirichletModalVsWeakFormulationTest(unittest.TestCase):
    """
    """

    def setUp(self):
        pass

    def test_it(self):
        actuation_type = 'dirichlet'
        bound_cond_type = 'dirichlet'
        param = [1., -2., -1., None, None]
        adjoint_param = ef.SecondOrderEigenfunction.get_adjoint_problem(param)
        a2, a1, a0, _, _ = param

        l = 1.
        spatial_disc = 10
        dz = sim.Domain(bounds=(0, l), num=spatial_disc)

        T = 1.
        temporal_disc = 1e2
        dt = sim.Domain(bounds=(0, T), num=temporal_disc)

        omega = np.array([(i + 1) * np.pi / l for i in range(spatial_disc)])
        eig_values = a0 - a2 * omega ** 2 - a1 ** 2 / 4. / a2
        norm_fak = np.ones(omega.shape) * np.sqrt(2)
        eig_funcs = np.array([ef.SecondOrderDirichletEigenfunction(omega[i], param, dz.bounds, norm_fak[i])
                              for i in range(spatial_disc)])
        register_base("eig_funcs", eig_funcs, overwrite=True)
        adjoint_eig_funcs = np.array([ef.SecondOrderDirichletEigenfunction(omega[i], adjoint_param, dz.bounds,
                                                                           norm_fak[i]) for i in range(spatial_disc)])
        register_base("adjoint_eig_funcs", adjoint_eig_funcs, overwrite=True)

        # derive initial field variable x(z,0) and weights
        start_state = cr.Function(lambda z: 0., domain=(0, l))
        initial_weights = cr.project_on_base(start_state, adjoint_eig_funcs)

        # init trajectory
        u = tr.RadTrajectory(l, T, param, bound_cond_type, actuation_type)

        # determine (A,B) with weak-formulation (pyinduct)

        # derive sate-space system
        rad_pde = ut.get_parabolic_dirichlet_weak_form("eig_funcs", "adjoint_eig_funcs", u, param, dz.bounds)
        cf = sim.parse_weak_formulation(rad_pde)
        ss_weak = cf.convert_to_state_space()

        # determine (A,B) with modal-transfomation
        A = np.diag(eig_values)
        B = -a2 * np.array([adjoint_eig_funcs[i].derive()(l) for i in range(spatial_disc)])
        ss_modal = sim.StateSpace("eig_funcs", A, B, input_handle=u)

        deregister_base("eig_funcs")
        deregister_base("adjoint_eig_funcs")

        # TODO: resolve the big tolerance (rtol=3e-01) between ss_modal.A and ss_weak.A
        # check if ss_modal.(A,B) is close to ss_weak.(A,B)
        self.assertTrue(np.allclose(np.sort(np.linalg.eigvals(ss_weak.A[1])), np.sort(np.linalg.eigvals(ss_modal.A[1])),
                                    rtol=3e-1, atol=0.))
        self.assertTrue(np.allclose(np.array([i[0] for i in ss_weak.B[1]]), ss_modal.B[1]))

        # display results
        if show_plots:
            t, q = sim.simulate_state_space(ss_modal, initial_weights, dt)
            eval_d = sim.evaluate_approximation("eig_funcs", q, t, dz, spat_order=1)
            win2 = vis.PgSurfacePlot(eval_d)
            app.exec_()


class RadRobinModalVsWeakFormulationTest(unittest.TestCase):
    """
    """

    def setUp(self):
        pass

    def test_it(self):
        actuation_type = 'robin'
        bound_cond_type = 'robin'
        param = [2., 1.5, -3., -1., -.5]
        adjoint_param = ef.SecondOrderEigenfunction.get_adjoint_problem(param)
        a2, a1, a0, alpha, beta = param

        l = 1.
        spatial_disc = 10
        dz = sim.Domain(bounds=(0, l), num=spatial_disc)

        T = 1.
        temporal_disc = 1e2
        dt = sim.Domain(bounds=(0, T), num=temporal_disc)
        n = 10

        eig_freq, eig_val = ef.SecondOrderRobinEigenfunction.eigfreq_eigval_hint(param, l, n)

        init_eig_funcs = np.array([ef.SecondOrderRobinEigenfunction(om, param, dz.bounds) for om in eig_freq])
        init_adjoint_eig_funcs = np.array([ef.SecondOrderRobinEigenfunction(om, adjoint_param, dz.bounds)
                                           for om in eig_freq])

        # normalize eigenfunctions and adjoint eigenfunctions
        eig_funcs, adjoint_eig_funcs = cr.normalize_base(init_eig_funcs, init_adjoint_eig_funcs)

        # register eigenfunctions
        register_base("eig_funcs", eig_funcs, overwrite=True)
        register_base("adjoint_eig_funcs", adjoint_eig_funcs, overwrite=True)

        # derive initial field variable x(z,0) and weights
        start_state = cr.Function(lambda z: 0., domain=(0, l))
        initial_weights = cr.project_on_base(start_state, adjoint_eig_funcs)

        # init trajectory
        u = tr.RadTrajectory(l, T, param, bound_cond_type, actuation_type)

        # determine (A,B) with weak-formulation (pyinduct)
        rad_pde = ut.get_parabolic_robin_weak_form("eig_funcs", "adjoint_eig_funcs", u, param, dz.bounds)
        cf = sim.parse_weak_formulation(rad_pde)
        ss_weak = cf.convert_to_state_space()

        # determine (A,B) with modal-transfomation
        A = np.diag(np.real_if_close(eig_val))
        B = a2 * np.array([adjoint_eig_funcs[i](l) for i in range(len(eig_freq))])
        ss_modal = sim.StateSpace("eig_funcs", A, B, input_handle=u)

        deregister_base("eig_funcs")
        deregister_base("adjoint_eig_funcs")

        # check if ss_modal.(A,B) is close to ss_weak.(A,B)
        self.assertTrue(np.allclose(np.sort(np.linalg.eigvals(ss_weak.A[1])), np.sort(np.linalg.eigvals(ss_modal.A[1])),
                                    rtol=1e-05, atol=0.))
        self.assertTrue(np.allclose(np.array([i[0] for i in ss_weak.B[1]]), ss_modal.B[1]))

        # display results
        if show_plots:
            t, q = sim.simulate_state_space(ss_modal, initial_weights, dt)
            eval_d = sim.evaluate_approximation("eig_funcs", q, t, dz, spat_order=1)
            win1 = vis.PgAnimatedPlot([eval_d], title="Test")
            win2 = vis.PgSurfacePlot(eval_d)
            app.exec_()
