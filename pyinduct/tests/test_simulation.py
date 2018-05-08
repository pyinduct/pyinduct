import sys
import unittest
import copy

import numpy as np

import pyinduct as pi
import pyinduct.hyperbolic.feedforward as hff
import pyinduct.parabolic as parabolic
import pyinduct.simulation as sim
from pyinduct.tests import show_plots

# TODO Test for Domain


class SimpleInput(sim.SimulationInput):
    """
    the simplest input we can imagine
    """
    def __init__(self):
        super().__init__("SimpleInput")

    def _calc_output(self, **kwargs):
        return 0


class MonotonousInput(sim.SimulationInput):
    """
    an input that ramps up
    """
    def __init__(self):
        super().__init__("MonotonousInput")

    def _calc_output(self, **kwargs):
        return dict(output=kwargs["time"])


class CorrectInput(sim.SimulationInput):
    """
    a diligent input
    """
    def __init__(self, limits):
        super().__init__(self)
        self.t_min = limits[0]
        self.t_max = limits[1]

    def _calc_output(self, **kwargs):
        if "time" not in kwargs:
            raise ValueError("mandatory key not found!")
        if "weights" not in kwargs:
            raise ValueError("mandatory key not found!")
        if "weight_lbl" not in kwargs:
            raise ValueError("mandatory key not found!")
        return dict(output=0)


class AlternatingInput(sim.SimulationInput):
    """
    a simple alternating input, composed of smooth transitions
    """

    def _calc_output(self, **kwargs):
        t = kwargs["time"] % 2
        if t < 1:
            res = self.tr_up(t)
        else:
            res = self.tr_down(t)

        return dict(output=res - .5)

    def __init__(self):
        super().__init__(self)
        self.tr_up = pi.SmoothTransition(states=(0, 1), interval=(0, 1), method="poly")
        self.tr_down = pi.SmoothTransition(states=(1, 0), interval=(1, 2), method="poly")


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
        u = CorrectInput(limits=(0, 1))
        ic = np.zeros((2, 1))
        ss = sim.StateSpace({1: a}, {0: {1: b}}, input_handle=u)

        # if caller provides correct kwargs no exception should be raised
        res = sim.simulate_state_space(ss, ic, pi.Domain((0, 1), num=10))

    def test_storage(self):
        a = np.eye(2, 2)
        b = np.array([[0], [1]])
        u = MonotonousInput()
        ic = np.zeros((2, 1))
        ss = sim.StateSpace(a, b, input_handle=u)

        # run simulation to fill the internal storage
        domain = pi.Domain((0, 10), step=.1)
        bigger_domain = pi.Domain((-1, 11), step=.1)
        res = sim.simulate_state_space(ss, ic, domain)

        # don't return any entries that aren't there
        self.assertRaises(KeyError, u.get_results, domain, "Unknown Entry")

        # default key is "output"
        ed = u.get_results(domain)
        ed_explicit = u.get_results(domain, result_key="output")
        self.assertTrue(np.array_equal(ed, ed_explicit))

        # return an np.ndarray as default
        self.assertIsInstance(ed, np.ndarray)

        # return EvalData if corresponding flag is set
        self.assertIsInstance(u.get_results(domain, as_eval_data=True),
                              pi.EvalData)

        # raise an error if extrapolation is performed
        self.assertRaises(ValueError, u.get_results, bigger_domain)

        # storage contains values
        self.assertTrue(u._time_storage)
        self.assertTrue(u._value_storage)

        # clear it
        u.clear_cache()

        # storage should be empty
        self.assertFalse(u._time_storage)
        self.assertFalse(u._value_storage)

        # double clearing should work
        u.clear_cache()


class CanonicalFormTest(unittest.TestCase):
    def setUp(self):
        self.cf = sim.CanonicalForm()
        self.u = SimpleInput()

    def test_add_to(self):
        a = np.eye(5)
        self.cf.add_to(dict(name="E", order=0, exponent=1), a)
        self.assertTrue(np.array_equal(self.cf.matrices["E"][0][1], a))
        self.cf.add_to(dict(name="E", order=0, exponent=1), 5 * a)
        self.assertTrue(np.array_equal(self.cf.matrices["E"][0][1], 6 * a))

        b = np.eye(10)
        self.assertRaises(ValueError, self.cf.add_to, dict(name="E", order=0, exponent=1), b)
        self.cf.add_to(dict(name="E", order=2, exponent=1), b)
        self.assertTrue(np.array_equal(self.cf.matrices["E"][2][1], b))

        f = np.atleast_2d(np.array(range(5))).T
        self.assertRaises(ValueError, self.cf.add_to, dict(name="E", order=0, exponent=1), f)
        self.cf.add_to(dict(name="f"), f)
        self.assertTrue(np.array_equal(self.cf.matrices["f"], f))

        # try to add something with derivative or exponent to f: value should end up in f
        self.cf.add_to(dict(name="f"), f)
        self.assertTrue(np.array_equal(self.cf.matrices["f"], 2 * f))

        c = np.atleast_2d(np.array(range(5))).T
        # that one should be easy
        self.cf.add_to(dict(name="G", order=0, exponent=1), c, column=0)
        self.assertTrue(np.array_equal(self.cf.matrices["G"][0][1], c))

        # here G01 as to be expanded
        self.cf.add_to(dict(name="G", order=0, exponent=1), c, column=1)
        self.assertTrue(np.array_equal(self.cf.matrices["G"][0][1], np.hstack((c, c))))

        # here G01 as to be expanded again
        self.cf.add_to(dict(name="G", order=0, exponent=1), c, column=3)
        self.assertTrue(np.array_equal(self.cf.matrices["G"][0][1], np.hstack((c, c, np.zeros_like(c), c))))


class ParseTest(unittest.TestCase):
    def setUp(self):
        # scalars
        self.scalars = pi.Scalars(np.vstack(list(range(3))))

        # inputs
        self.u = np.sin
        self.input = pi.Input(self.u)
        self.input_squared = pi.Input(self.u, exponent=2)

        # scale function
        def heavyside(z):
            if z < 0.5:
                return 0
            elif z == 0.5:
                return .5
            else:
                return 1

        base = pi.Base(pi.Function(heavyside))
        pi.register_base("heavyside_base", base)

        # distributed base
        nodes = pi.Domain((0, 1), num=3)
        self.distributed_base = pi.LagrangeSecondOrder.cure_interval(nodes)
        pi.register_base("distributed_base", self.distributed_base)

        # lumped base
        self.lumped_base = pi.Base([pi.Function.from_constant(1)])
        pi.register_base("lumped_base", self.lumped_base)

        # Test Functions
        self.test_funcs = pi.TestFunction("distributed_base")
        self.test_funcs_at0 = self.test_funcs(0)
        self.test_funcs_at1 = self.test_funcs(1)
        self.test_funcs_dz = self.test_funcs.derive(1)
        self.test_funcs_dz_at1 = self.test_funcs_dz(1)

        # Scalar Functions
        self.scalar_func = pi.ScalarFunction("heavyside_base")

        # Distributed / Field Variables
        self.field_var = pi.FieldVariable("distributed_base")
        self.field_var_squared = pi.FieldVariable("distributed_base",
                                                  exponent=2)
        self.field_var_cubed = pi.FieldVariable("distributed_base",
                                                exponent=3)
        self.field_var_at1 = self.field_var(1)
        self.field_var_at1_squared = pi.FieldVariable("distributed_base",
                                                      location=1,
                                                      exponent=2)

        self.field_var_dz = self.field_var.derive(spat_order=1)
        self.field_var_dz_at1 = self.field_var_dz(1)
        self.field_var_ddt = self.field_var.derive(temp_order=2)
        self.field_var_ddt_at0 = self.field_var_ddt(0)
        self.field_var_ddt_at1 = self.field_var_ddt(1)

        self.odd_weight_field_var = pi.FieldVariable(
            "distributed_base", weight_label="special_weights")

        # Field variable 2
        self.lumped_var = pi.FieldVariable("lumped_base")

        # Construction of Equation Terms

        # input
        self.input_term1 = pi.ScalarTerm(pi.Product(self.test_funcs_at1,
                                                    self.input))
        self.input_term1_squared = pi.ScalarTerm(pi.Product(self.test_funcs_at1,
                                                            self.input_squared))
        self.input_term1_swapped = pi.ScalarTerm(pi.Product(self.input,
                                                            self.test_funcs_at1)
                                                 )

        self.input_term2 = pi.ScalarTerm(pi.Product(self.test_funcs_dz_at1,
                                                    self.input))

        self.input_term3 = pi.IntegralTerm(pi.Product(self.test_funcs,
                                                      self.input),
                                           limits=(0, 1))
        self.input_term3_swapped = pi.IntegralTerm(pi.Product(self.input,
                                                              self.test_funcs),
                                                   limits=(0, 1))
        self.input_term3_scaled = pi.IntegralTerm(
            pi.Product(pi.Product(self.scalar_func, self.test_funcs),
                       self.input),
            limits=(0, 1))

        self.input_term3_scaled_first_half = pi.IntegralTerm(
            pi.Product(pi.Product(self.scalar_func, self.test_funcs),
                       self.input),
            limits=(0, .5))

        self.input_term3_scaled_second_half = pi.IntegralTerm(
            pi.Product(pi.Product(self.scalar_func, self.test_funcs),
                       self.input),
            limits=(.5, 1))

        # pure test function terms
        self.func_term = pi.ScalarTerm(self.test_funcs_at1)

        # pure field variable terms
        self.field_term_at1 = pi.ScalarTerm(self.field_var_at1)
        self.field_term_at1_squared = pi.ScalarTerm(self.field_var_at1_squared)
        self.field_term_dz_at1 = pi.ScalarTerm(self.field_var_dz_at1)
        self.field_term_ddt_at1 = pi.ScalarTerm(self.field_var_ddt_at1)

        self.field_int = pi.IntegralTerm(self.field_var, limits=(0, 1))
        self.field_int_half = pi.IntegralTerm(self.field_var, limits=(0, .5))
        self.field_squared_int = pi.IntegralTerm(self.field_var_squared, (0, 1))
        self.field_dz_int = pi.IntegralTerm(self.field_var_dz, (0, 1))
        self.field_ddt_int = pi.IntegralTerm(self.field_var_ddt, (0, 1))

        # products
        self.prod_term_fs_at1 = pi.ScalarTerm(
            pi.Product(self.field_var_at1, self.scalars))
        self.prod_int_fs = pi.IntegralTerm(pi.Product(self.field_var,
                                                      self.scalars),
                                           (0, 1))
        self.prod_int_f_f = pi.IntegralTerm(pi.Product(self.field_var,
                                                       self.test_funcs),
                                            (0, 1))
        self.prod_int_f_squared_f = pi.IntegralTerm(pi.Product(
            self.field_var_squared,
            self.test_funcs),
            (0, 1))
        self.prod_int_f_f_swapped = pi.IntegralTerm(pi.Product(self.test_funcs,
                                                               self.field_var),
                                                    (0, 1))
        self.prod_int_f_at1_f = pi.IntegralTerm(
            pi.Product(self.field_var_at1, self.test_funcs), (0, 1))
        self.prod_int_f_at1_squared_f = pi.IntegralTerm(
            pi.Product(self.field_var_at1_squared, self.test_funcs), (0, 1))

        self.prod_int_f_f_at1 = pi.IntegralTerm(
            pi.Product(self.field_var, self.test_funcs_at1), (0, 1))
        self.prod_int_f_squared_f_at1 = pi.IntegralTerm(
            pi.Product(self.field_var_squared, self.test_funcs_at1), (0, 1))

        self.prod_term_f_at1_f_at1 = pi.ScalarTerm(
            pi.Product(self.field_var_at1, self.test_funcs_at1))
        self.prod_term_f_at1_squared_f_at1 = pi.ScalarTerm(
            pi.Product(self.field_var_at1_squared, self.test_funcs_at1))

        self.prod_int_fddt_f = pi.IntegralTerm(
            pi.Product(self.field_var_ddt, self.test_funcs), (0, 1))
        self.prod_term_fddt_at0_f_at0 = pi.ScalarTerm(
            pi.Product(self.field_var_ddt_at0, self.test_funcs_at0))

        self.prod_term_f_at1_dphi_at1 = pi.ScalarTerm(
            pi.Product(self.field_var_at1, self.test_funcs_dz_at1))

        self.temp_int = pi.IntegralTerm(pi.Product(self.field_var_ddt,
                                                   self.test_funcs),
                                        limits=(0, 1))
        self.spat_int = pi.IntegralTerm(pi.Product(self.field_var_dz,
                                                   self.test_funcs_dz),
                                        limits=(0, 1))
        self.spat_int_asymmetric = pi.IntegralTerm(pi.Product(self.field_var_dz,
                                                              self.test_funcs),
                                                   limits=(0, 1))

        self.prod_term_tf_at0_lv_at0 = pi.ScalarTerm(
            pi.Product(self.test_funcs(0), self.lumped_var(0)))
        self.prod_term_tf_at0_lv_at0_swapped = pi.ScalarTerm(
            pi.Product(self.lumped_var(0), self.test_funcs(0)))

        self.prod_int_sf_fv = pi.IntegralTerm(pi.Product(self.scalar_func,
                                                         self.field_var),
                                              limits=(0, 1))
        self.prod_int_sf_fv_swapped = pi.IntegralTerm(
            pi.Product(self.field_var, self.scalar_func),
            limits=(0, 1))

        self.alternating_weights_term = pi.IntegralTerm(
            self.odd_weight_field_var,
            limits=(0, 1))

    def test_Input_term(self):
        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.input_term2, name="test"),
            finalize=False).get_static_terms()
        np.testing.assert_array_almost_equal(terms["G"][0][1],
                                             np.array([[0], [-2], [2]]))

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.input_term1_squared, name="test"),
            finalize=False).get_static_terms()
        np.testing.assert_array_almost_equal(terms["G"][0][2],
                                             np.array([[0], [0], [1]]))

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.input_term3, name="test"),
            finalize=False).get_static_terms()
        np.testing.assert_array_almost_equal(terms["G"][0][1],
                                             np.array([[.25], [.5], [.25]]))

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.input_term3_swapped, name="test"),
            finalize=False).get_static_terms()
        np.testing.assert_array_almost_equal(terms["G"][0][1],
                                             np.array([[.25], [.5], [.25]]))

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.input_term3_scaled, name="test"),
            finalize=False).get_static_terms()
        np.testing.assert_array_almost_equal(terms["G"][0][1],
                                             np.array([[.0], [.25], [.25]]))

        terms_fh = sim.parse_weak_formulation(
            sim.WeakFormulation(self.input_term3_scaled_first_half, name="test"),
            finalize=False).get_static_terms()
        np.testing.assert_array_almost_equal(terms_fh["G"][0][1],
                                             np.array([[.0], [.0], [.0]]))

        terms_sh = sim.parse_weak_formulation(
            sim.WeakFormulation(self.input_term3_scaled_second_half, name="test"),
            finalize=False).get_static_terms()
        np.testing.assert_array_almost_equal(terms_sh["G"][0][1],
                                             np.array([[.0], [.25], [.25]]))

    def test_TestFunction_term(self):
        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.func_term, name="test"),
            finalize=False).get_static_terms()
        np.testing.assert_array_almost_equal(terms["f"],
                                             np.array([[0], [0], [1]]))

    def test_FieldVariable_term(self):
        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.field_term_at1, name="test"),
            finalize=False).get_dynamic_terms()["distributed_base"]
        np.testing.assert_array_almost_equal(terms["E"][0][1],
                                             np.array([[0, 0, 1]]))

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.field_term_at1_squared, name="test"),
            finalize=False).get_dynamic_terms()["distributed_base"]
        np.testing.assert_array_almost_equal(terms["E"][0][2],
                                             np.array([[0, 0, 1]]))

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.field_term_ddt_at1, name="test"),
            finalize=False).get_dynamic_terms()["distributed_base"]
        np.testing.assert_array_almost_equal(terms["E"][2][1],
                                             np.array([[0, 0, 1]]))

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.field_term_dz_at1, name="test"),
            finalize=False).get_dynamic_terms()["distributed_base"]
        np.testing.assert_array_almost_equal(terms["E"][0][1],
                                             np.array([[0, -2, 2]]))

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.field_int, name="test"),
            finalize=False).get_dynamic_terms()["distributed_base"]
        np.testing.assert_array_almost_equal(terms["E"][0][1],
                                             np.array([[.25, .5, .25]]))

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.field_int_half, name="test"),
            finalize=False).get_dynamic_terms()["distributed_base"]
        np.testing.assert_array_almost_equal(terms["E"][0][1],
                                             np.array([[.25, .25, 0]]))

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.field_squared_int, name="test"),
            finalize=False).get_dynamic_terms()["distributed_base"]
        np.testing.assert_array_almost_equal(terms["E"][0][2],
                                             np.array([[1 / 6, 1 / 3, 1 / 6]]))

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.field_dz_int, name="test"),
            finalize=False).get_dynamic_terms()["distributed_base"]
        np.testing.assert_array_almost_equal(terms["E"][0][1],
                                             np.array([[-1, 0, 1]]))

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.field_ddt_int, name="test"),
            finalize=False).get_dynamic_terms()["distributed_base"]
        np.testing.assert_array_almost_equal(terms["E"][2][1],
                                             np.array([[.25, .5, .25]]))

    def test_Product_term(self):
        # TODO create test functionality that will automatically check if Case
        # is also valid for swapped arguments

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.prod_term_fs_at1, name="test"),
            finalize=False).get_dynamic_terms()["distributed_base"]
        np.testing.assert_array_almost_equal(terms["E"][0][1],
                                             np.array([[0, 0, 0],
                                                       [0, 0, 1],
                                                       [0, 0, 2]]))

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.prod_int_fs, name="test"),
            finalize=False).get_dynamic_terms()["distributed_base"]
        np.testing.assert_array_almost_equal(terms["E"][0][1],
                                             np.array([[0, 0, 0],
                                                       [0.25, .5, .25],
                                                       [.5, 1, .5]]))

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.prod_int_f_f, name="test"),
            finalize=False).get_dynamic_terms()["distributed_base"]
        np.testing.assert_array_almost_equal(terms["E"][0][1],
                                             np.array([[1 / 6, 1 / 12, 0],
                                                       [1 / 12, 1 / 3, 1 / 12],
                                                       [0, 1 / 12, 1 / 6]]))

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.prod_int_f_squared_f, name="test"),
            finalize=False).get_dynamic_terms()["distributed_base"]
        np.testing.assert_array_almost_equal(terms["E"][0][2],
                                             np.array([[1 / 8, 1 / 24, 0],
                                                       [1 / 24, 1 / 4, 1 / 24],
                                                       [0, 1 / 24, 1 / 8]]))

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.prod_int_f_f_swapped, name="test"),
            finalize=False).get_dynamic_terms()["distributed_base"]
        np.testing.assert_array_almost_equal(terms["E"][0][1],
                                             np.array([[1 / 6, 1 / 12, 0],
                                                       [1 / 12, 1 / 3, 1 / 12],
                                                       [0, 1 / 12, 1 / 6]]))

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.prod_int_f_at1_f, name="test"),
            finalize=False).get_dynamic_terms()["distributed_base"]
        np.testing.assert_array_almost_equal(terms["E"][0][1],
                                             np.array([[0, 0, 0.25],
                                                       [0, 0, 0.5],
                                                       [0, 0, .25]]))

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.prod_int_f_at1_squared_f, name="test"),
            finalize=False).get_dynamic_terms()["distributed_base"]
        np.testing.assert_array_almost_equal(terms["E"][0][2],
                                             np.array([[0, 0, 0.25],
                                                       [0, 0, 0.5],
                                                       [0, 0, .25]]))

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.prod_int_f_f_at1, name="test"),
            finalize=False).get_dynamic_terms()["distributed_base"]
        np.testing.assert_array_almost_equal(terms["E"][0][1],
                                             np.array([[0, 0, 0],
                                                       [0, 0, 0],
                                                       [0.25, 0.5, .25]]))

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.prod_int_f_squared_f_at1, name="test"),
            finalize=False).get_dynamic_terms()["distributed_base"]
        np.testing.assert_array_almost_equal(terms["E"][0][2],
                                             np.array([[0, 0, 0],
                                                       [0, 0, 0],
                                                       [1 / 6, 1 / 3, 1 / 6]]))

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.prod_term_f_at1_f_at1, name="test"),
            finalize=False).get_dynamic_terms()["distributed_base"]
        np.testing.assert_array_almost_equal(terms["E"][0][1],
                                             np.array([[0, 0, 0],
                                                       [0, 0, 0],
                                                       [0, 0, 1]]))

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.prod_term_f_at1_squared_f_at1, name="test"),
            finalize=False).get_dynamic_terms()["distributed_base"]
        np.testing.assert_array_almost_equal(terms["E"][0][2],
                                             np.array([[0, 0, 0],
                                                       [0, 0, 0],
                                                       [0, 0, 1]]))

        # more complex terms
        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.prod_int_fddt_f, name="test"),
            finalize=False).get_dynamic_terms()["distributed_base"]
        np.testing.assert_array_almost_equal(terms["E"][2][1],
                                             np.array([[1 / 6, 1 / 12, 0],
                                                       [1 / 12, 1 / 3, 1 / 12],
                                                       [0, 1 / 12, 1 / 6]]))

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.prod_term_fddt_at0_f_at0, name="test"),
            finalize=False).get_dynamic_terms()["distributed_base"]
        np.testing.assert_array_almost_equal(terms["E"][2][1],
                                             np.array([[1, 0, 0],
                                                       [0, 0, 0],
                                                       [0, 0, 0]]))

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.spat_int, name="test"),
            finalize=False).get_dynamic_terms()["distributed_base"]
        np.testing.assert_array_almost_equal(terms["E"][0][1],
                                             np.array([[2, -2, 0],
                                                       [-2, 4, -2],
                                                       [0, -2, 2]]))

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.spat_int_asymmetric, name="test"),
            finalize=False).get_dynamic_terms()["distributed_base"]
        np.testing.assert_array_almost_equal(terms["E"][0][1],
                                             np.array([[-.5, .5, 0],
                                                       [-.5, 0, .5],
                                                       [0, -.5, .5]]))

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.prod_term_f_at1_dphi_at1, name="test"),
            finalize=False).get_dynamic_terms()["distributed_base"]
        np.testing.assert_array_almost_equal(terms["E"][0][1],
                                             np.array([[0, 0, 0],
                                                       [0, 0, -2],
                                                       [0, 0, 2]]))

        desired = np.array([[0, 0.25, 0.25]])
        terms1 = sim.parse_weak_formulation(
            sim.WeakFormulation(self.prod_int_sf_fv, name="test"),
            finalize=False).get_dynamic_terms()["distributed_base"]
        np.testing.assert_array_almost_equal(terms1["E"][0][1], desired)

        terms2 = sim.parse_weak_formulation(
            sim.WeakFormulation(self.prod_int_sf_fv_swapped, name="test"),
            finalize=False).get_dynamic_terms()["distributed_base"]
        np.testing.assert_array_almost_equal(terms2["E"][0][1], desired)

        desired = np.array([[1], [0], [0]])
        terms1 = sim.parse_weak_formulation(
            sim.WeakFormulation(self.prod_term_tf_at0_lv_at0, name="test"),
            finalize=False).get_dynamic_terms()["lumped_base"]
        np.testing.assert_array_almost_equal(terms1["E"][0][1], desired)

        terms2 = sim.parse_weak_formulation(
            sim.WeakFormulation(self.prod_term_tf_at0_lv_at0_swapped, name="test"),
            finalize=False).get_dynamic_terms()["lumped_base"]
        np.testing.assert_array_almost_equal(terms2["E"][0][1], desired)

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.input_term1, name="test"),
            finalize=False).get_static_terms()
        np.testing.assert_array_almost_equal(terms["G"][0][1],
                                             np.array([[0], [0], [1]]))

        terms = sim.parse_weak_formulation(
            sim.WeakFormulation(self.input_term1_swapped, name="test"),
            finalize=False).get_static_terms()
        np.testing.assert_array_almost_equal(terms["G"][0][1],
                                             np.array([[0], [0], [1]]))

    def test_alternating_weights(self):
        self.assertRaises(ValueError, sim.parse_weak_formulation,
                          sim.WeakFormulation([self.alternating_weights_term,
                                               self.field_int],
                                              name=""))

    def tearDown(self):
        pi.deregister_base("heavyside_base")
        pi.deregister_base("distributed_base")
        pi.deregister_base("lumped_base")


class StateSpaceTests(unittest.TestCase):
    def setUp(self):

        # setup temp and spat domain
        self.time_domain = pi.Domain((0, 1), num=10)
        node_cnt = 3
        spat_domain = pi.Domain((0, 1), num=node_cnt)
        lag_base = pi.LagrangeFirstOrder.cure_interval(spat_domain)
        pi.register_base("swm_base", lag_base)

        # input
        self.u = CorrectInput(limits=(0, 10))
        # self.u = CorrectInput(limits=self.time_domain.bounds)

        field_var = pi.FieldVariable("swm_base")
        field_var_ddt = field_var.derive(temp_order=2)
        field_var_dz = field_var.derive(spat_order=1)

        psi = pi.TestFunction("swm_base")
        psi_dz = psi.derive(1)

        # enter string with mass equations
        int1 = pi.IntegralTerm(pi.Product(field_var_ddt, psi),
                               spat_domain.bounds)
        s1 = pi.ScalarTerm(pi.Product(field_var_ddt(0), psi(0)))
        int2 = pi.IntegralTerm(pi.Product(field_var_dz, psi_dz),
                               spat_domain.bounds)
        s2 = pi.ScalarTerm(pi.Product(pi.Input(self.u), psi(1)), -1)

        string_eq = sim.WeakFormulation([int1, s1, int2, s2], name="swm")
        self.ce = sim.parse_weak_formulation(string_eq)
        self.ic = np.zeros((6, ))

    def test_convert_to_state_space(self):
        ss = sim.create_state_space(self.ce)
        self.assertEqual(ss.A[1].shape, (6, 6))
        np.testing.assert_array_almost_equal(ss.A[1], np.array([[0, 0, 0, 1, 0, 0],
                                                                [0, 0, 0, 0, 1, 0],
                                                                [0, 0, 0, 0, 0, 1],
                                                                [-2.25, 3, -.75, 0, 0, 0],
                                                                [7.5, -18, 10.5, 0, 0, 0],
                                                                [-3.75, 21, -17.25, 0, 0, 0]]))
        self.assertEqual(ss.B[0][1].shape, (6, 1))
        np.testing.assert_array_almost_equal(ss.B[0][1], np.array([[0], [0], [0], [0.125], [-1.75], [6.875]]))
        self.assertEqual(self.ce.input_function, self.u)

    def test_simulate_state_space(self):
        """
        using the diligent input this test makes sure, that the solver doesn't evaluate the provided input outside
        the given time domain
        """
        ss = sim.create_state_space(self.ce)
        t, q = sim.simulate_state_space(ss, self.ic, self.time_domain)

        # print(self.u._time_storage)
        # print(self.time_domain.points)
        # print(t.points)

        # check that the demanded time range has been simulated
        np.testing.assert_array_almost_equal(t.points, self.time_domain.points)

    def tearDown(self):
        pi.deregister_base("swm_base")


class StringMassTest(unittest.TestCase):
    example_data = None

    def create_test_data(self):
        if self.example_data is None:
            self.setUp()
            self.test_fem()
            self.tearDown()
        return copy.copy(self.example_data)

    def setUp(self):
        z_start = 0
        z_end = 1
        z_step = 0.1
        self.dz = pi.Domain(bounds=(z_start, z_end), num=9)

        t_start = 0
        t_end = 10
        t_step = 0.01
        self.dt = pi.Domain(bounds=(t_start, t_end), step=t_step)

        self.params = pi.Parameters
        self.params.node_distance = 0.1
        self.params.m = 1.0
        self.params.order = 8
        self.params.sigma = 1
        self.params.tau = 1

        self.y_end = 10

        self.u = hff.FlatString(0, self.y_end, z_start, z_end, 0, 5, self.params)

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
            pi.Function(lambda z: x(z, 0)),  # x(z, 0)
            pi.Function(lambda z: x_dt(z, 0)),  # dx_dt(z, 0)
        ])

    def test_fem(self):
        """
        use best documented fem case to test all steps in simulation process
        """
        # enter string with mass equations
        nodes = pi.Domain(self.dz.bounds, num=11)
        fem_base = pi.LagrangeFirstOrder.cure_interval(nodes)
        pi.register_base("fem_base", fem_base)

        field_var = pi.FieldVariable("fem_base")
        field_var_ddt = field_var.derive(temp_order=2)
        field_var_dz = field_var.derive(spat_order=1)

        psi = pi.TestFunction("fem_base")
        psi_dz = psi.derive(1)

        # enter string with mass equations
        int1 = pi.IntegralTerm(pi.Product(field_var_ddt, psi),
                               self.dz.bounds,
                               scale=self.params.sigma)

        s1 = pi.ScalarTerm(pi.Product(field_var_ddt(0), psi(0)),
                           scale=self.params.m)

        int2 = pi.IntegralTerm(pi.Product(field_var_dz, psi_dz),
                               self.dz.bounds,
                               scale=self.params.sigma)

        s2 = pi.ScalarTerm(pi.Product(pi.Input(self.u), psi(1)),
                           scale=-self.params.sigma)

        # derive sate-space system
        string_pde = sim.WeakFormulation([int1, s1, int2, s2], name="fem_test")
        self.cf = sim.parse_weak_formulation(string_pde)
        ss = sim.create_state_space(self.cf)

        # generate initial conditions for weights
        q0 = np.array([pi.project_on_base(self.ic[idx], fem_base)
                       for idx in range(2)]).flatten()

        # simulate
        t, q = sim.simulate_state_space(ss, q0, self.dt)

        # calculate result data
        eval_data = []
        for der_idx in range(2):
            eval_data.append(sim.evaluate_approximation(
                "fem_base",
                q[:, der_idx*fem_base.fractions.size:(der_idx + 1)*fem_base.fractions.size],
                t, self.dz))
            eval_data[-1].name = "{0}{1}".format(self.cf.name, "_" + "".join(
                ["d" for x in range(der_idx)]) + "t" if der_idx > 0 else "")

        # display results
        if show_plots:
            win = pi.PgAnimatedPlot(eval_data[:2],
                                    title="fem approx and derivative")
            win2 = pi.PgSurfacePlot(eval_data[0])
            pi.show(show_mpl=False)

        # test for correct transition
        self.assertAlmostEqual(eval_data[0].output_data[-1, 0],
                               self.y_end,
                               places=3)

        # save some test data for later use
        self.example_data = eval_data

        pi.deregister_base("fem_base")

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
        eig_frequencies = pi.find_roots(char_eq,
                                        grid=np.arange(0, 1e3, 2),
                                        n_roots=order,
                                        rtol=1e-2)
        print("eigenfrequencies:")
        print(eig_frequencies)

        # create eigen function vectors
        class SWMFunctionVector(pi.ComposedFunctionVector):
            """
            String With Mass Function Vector, necessary due to manipulated scalar product
            """
            def __init__(self, function, function_at_0):
                super().__init__(function, function_at_0)

            @property
            def func(self):
                return self.members["funcs"][0]

            @property
            def scalar(self):
                return self.members["scalars"][0]

        eig_vectors = np.array([SWMFunctionVector(pi.Function(phi_k_factory(eig_frequencies[n]),
                                                              derivative_handles=[
                                                                  phi_k_factory(eig_frequencies[n], der_order)
                                                                  for der_order in range(1, 3)],
                                                              domain=self.dz.bounds,
                                                              nonzero=self.dz.bounds),
                                                  phi_k_factory(eig_frequencies[n])(0))
                                for n in range(order)])
        composed_modal_base = pi.Base(eig_vectors)

        # normalize base
        norm_comp_mod_base = pi.normalize_base(composed_modal_base)
        norm_mod_base = pi.Base(np.array([vec.func for vec in norm_comp_mod_base.fractions]))
        pi.register_base("norm_modal_base", norm_mod_base, overwrite=True)

        # debug print eigenfunctions
        if 0:
            func_vals = []
            for vec in eig_vectors:
                func_vals.append(np.vectorize(vec.func)(self.dz))

            norm_func_vals = []
            for func in norm_mod_base.fractions:
                norm_func_vals.append(np.vectorize(func)(self.dz))

            clrs = ["r", "g", "b", "c", "m", "y", "k", "w"]
            for n in range(1, order + 1, len(clrs)):
                pw_phin_k = pg.plot(title="phin_k for k in [{0}, {1}]".format(n, min(n + len(clrs), order)))
                for k in range(len(clrs)):
                    if k + n > order:
                        break
                    pw_phin_k.plot(x=np.array(self.dz), y=norm_func_vals[n + k - 1], pen=clrs[k])

            pi.show(show_mpl=False)

        # create terms of weak formulation
        terms = [pi.IntegralTerm(pi.Product(pi.FieldVariable("norm_modal_base", order=(2, 0)),
                                            pi.TestFunction("norm_modal_base")),
                                 self.dz.bounds, scale=-1),
                 pi.ScalarTerm(pi.Product(
                     pi.FieldVariable("norm_modal_base", order=(2, 0), location=0),
                     pi.TestFunction("norm_modal_base", location=0)),
                     scale=-1),
                 pi.ScalarTerm(pi.Product(pi.Input(self.u),
                                          pi.TestFunction("norm_modal_base", location=1))),
                 pi.ScalarTerm(
                     pi.Product(pi.FieldVariable("norm_modal_base", location=1),
                                pi.TestFunction("norm_modal_base", order=1, location=1)),
                     scale=-1),
                 pi.ScalarTerm(pi.Product(pi.FieldVariable("norm_modal_base", location=0),
                                          pi.TestFunction("norm_modal_base", order=1,
                                                          location=0))),
                 pi.IntegralTerm(pi.Product(pi.FieldVariable("norm_modal_base"),
                                            pi.TestFunction("norm_modal_base", order=2)),
                                 self.dz.bounds)]
        modal_pde = sim.WeakFormulation(terms, name="swm_lib-modal")

        # simulate
        eval_data = sim.simulate_system(modal_pde, self.ic, self.dt, self.dz, derivative_orders=(1, 0))

        # display results
        if show_plots:
            win = pi.PgAnimatedPlot(eval_data[0:2], title="modal approx and derivative")
            win2 = pi.PgSurfacePlot(eval_data[0])
            pi.show(show_mpl=False)

        pi.deregister_base("norm_modal_base")

        # test for correct transition
        self.assertTrue(np.isclose(eval_data[0].output_data[-1, 0], self.y_end, atol=1e-3))

    def tearDown(self):
        pass


class MultipleODETest(unittest.TestCase):
    def desired_test_pr12(self):
        """
        Let us consider the system of ordinary differential equations

            x1^(3)(t) = x2(t) + u(t)
            x2^(1)(t) = x1^(2)(t) + u(t).

        Desired state space model for x = (x1, x1^(1), x1^(2), x2)^T

                    [   0   1   0   0   ]       [0]
           x^(1) =  [   0   0   1   0   ] x +   [0] u.
                    [   0   0   0   1   ]       [1]
                    [   0   0   1   0   ]       [1]
        """
        a_desired = np.array([[0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1],
                              [0, 0, 1, 0]])
        b_desired = np.array([[0], [0], [1], [1]])

        dummy_domain = pi.Domain(bounds=(-1, 1), num=2)
        dummy_point = 0

        pi.register_base("base_1", pi.Base(
            pi.Function.from_constant(1, domain=dummy_domain.bounds)))
        pi.register_base("base_2", pi.Base(
            pi.Function.from_constant(1, domain=dummy_domain.bounds)))

        x1 = pi.FieldVariable("base_1")(dummy_point)
        x2 = pi.FieldVariable("base_2")(dummy_point)
        u = pi.Input(pi.ConstantTrajectory(0))

        weak_form_1 = pi.WeakFormulation([
            pi.ScalarTerm(x1.derive(temp_order=3), scale=-1),
            pi.ScalarTerm(x2),
            pi.ScalarTerm(u)
        ], name="sys_1")

        weak_form_2 = pi.WeakFormulation([
            pi.ScalarTerm(x2.derive(temp_order=1), scale=-1),
            pi.ScalarTerm(x1.derive(temp_order=2)),
            pi.ScalarTerm(u)
        ], name="sys_2", dominant_lbl="base_2")

        weak_forms = [weak_form_1, weak_form_2]
        canonical_equations = [pi.parse_weak_formulation(form)
                                    for form in weak_forms]

        state_space_form = pi.create_state_space(canonical_equations)

        np.testing.assert_array_almost_equal(state_space_form.A[1], a_desired)
        np.testing.assert_array_almost_equal(state_space_form.B[0][1], b_desired)


class MultiplePDETest(unittest.TestCase):
    """
    This TestCase covers the implementation of the parsing and simulation of
    coupled pde systems.
    """

    def setUp(self):
        l1 = 1
        l2 = 4
        l3 = 6
        l4 = 10

        self.dz1 = pi.Domain(bounds=(0, l1), num=100)
        self.dz2 = pi.Domain(bounds=(l1, l2), num=100)
        self.dz3 = pi.Domain(bounds=(l2, l3), num=100)
        self.dz4 = pi.Domain(bounds=(l3, l4), num=100)

        t_start = 0
        t_end = 10
        t_step = 0.01
        self.dt = pi.Domain(bounds=(t_start, t_end), step=t_step)

        v1 = 1
        v2 = 2
        v3 = 3
        mass = 1

        def x(z, t):
            """
            initial conditions for testing
            """
            return 0

        # initial conditions
        fx = pi.Function(lambda z: x(z, 0))
        self.ic1 = np.array([fx])
        self.ic2 = np.array([fx])
        self.ic3 = np.array([fx])
        self.ic4 = np.array([fx, fx])

        # weak formulations
        nodes1 = pi.Domain(self.dz1.bounds, num=3)
        nodes2 = pi.Domain(self.dz2.bounds, num=3)
        nodes3 = pi.Domain(self.dz3.bounds, num=3)
        nodes4 = pi.Domain(self.dz4.bounds, num=15)

        base1 = pi.LagrangeFirstOrder.cure_interval(nodes1)
        base2 = pi.LagrangeFirstOrder.cure_interval(nodes2)
        base3 = pi.LagrangeFirstOrder.cure_interval(nodes3)
        base4 = pi.LagrangeFirstOrder.cure_interval(nodes4)

        pi.register_base("base_1", base1)
        pi.register_base("base_2", base2)
        pi.register_base("base_3", base3)
        pi.register_base("base_4", base4)

        traj = AlternatingInput()
        u = pi.Input(traj)

        x1 = pi.FieldVariable("base_1")
        psi_1 = pi.TestFunction("base_1")
        self.weak_form_1 = pi.WeakFormulation([
            pi.IntegralTerm(pi.Product(x1.derive(temp_order=1), psi_1), limits=self.dz1.bounds),
            pi.IntegralTerm(pi.Product(x1, psi_1.derive(1)), limits=self.dz1.bounds, scale=-v1),
            pi.ScalarTerm(pi.Product(u, psi_1(0)), scale=-v1),
            pi.ScalarTerm(pi.Product(x1(l1), psi_1(l1)), scale=v1),
        ], name="sys_1")

        x2 = pi.FieldVariable("base_2")
        psi_2 = pi.TestFunction("base_2")
        self.weak_form_2 = pi.WeakFormulation([
            pi.IntegralTerm(pi.Product(x2.derive(temp_order=1), psi_2), limits=self.dz2.bounds),
            pi.IntegralTerm(pi.Product(x2, psi_2.derive(1)), limits=self.dz2.bounds, scale=-v2),
            pi.ScalarTerm(pi.Product(x1(l1), psi_2(l1)), scale=-v2),
            pi.ScalarTerm(pi.Product(x2(l2), psi_2(l2)), scale=v2),
        ], name="sys_2")

        x3 = pi.FieldVariable("base_3")
        psi_3 = pi.TestFunction("base_3")
        self.weak_form_3 = pi.WeakFormulation([
            pi.IntegralTerm(pi.Product(x3.derive(temp_order=1), psi_3), limits=self.dz3.bounds),
            pi.IntegralTerm(pi.Product(x3, psi_3.derive(1)), limits=self.dz3.bounds, scale=-v3),
            pi.ScalarTerm(pi.Product(x2(l2), psi_3(l2)), scale=-v3),
            pi.ScalarTerm(pi.Product(x3(l3), psi_3(l3)), scale=v3),
        ], name="sys_3")

        x4 = pi.FieldVariable("base_4")
        psi_4 = pi.TestFunction("base_4")
        self.weak_form_4 = pi.WeakFormulation([
            pi.IntegralTerm(pi.Product(x4.derive(temp_order=2), psi_4), limits=self.dz4.bounds, scale=-1),
            pi.IntegralTerm(pi.Product(x4.derive(spat_order=1), psi_4.derive(1)), limits=self.dz4.bounds, scale=-1),
            pi.ScalarTerm(pi.Product(x4.derive(temp_order=2)(l4), psi_4(l4)), scale=-mass),
            pi.ScalarTerm(pi.Product(x3(l3), psi_4(l3)), scale=-1),
        ], name="sys_4")

    def test_single_system(self):
        results = pi.simulate_system(self.weak_form_1, self.ic1, self.dt, self.dz1)
        win = pi.PgAnimatedPlot(results)
        if show_plots:
            pi.show(show_mpl=False)

    def test_coupled_system(self):
        """
        test the coupled system
        """
        weak_forms = [self.weak_form_1, self.weak_form_2]
        ics = {self.weak_form_1.name: self.ic1, self.weak_form_2.name: self.ic2}
        spat_domains = {self.weak_form_1.name: self.dz1, self.weak_form_2.name: self.dz2}
        derivatives = {self.weak_form_1.name: (0, 0), self.weak_form_2.name: (0, 0)}

        res = pi.simulate_systems(weak_forms, ics, self.dt, spat_domains, derivatives)
        win = pi.PgAnimatedPlot(res)

        if show_plots:
            pi.show(show_mpl=False)
            del win

    def test_triple_system(self):
        """
        three coupled systems
        """
        weak_forms = [self.weak_form_1, self.weak_form_2, self.weak_form_3]
        ics = {self.weak_form_1.name: self.ic1,
               self.weak_form_2.name: self.ic2,
               self.weak_form_3.name: self.ic3}
        spat_domains = {self.weak_form_1.name: self.dz1,
                        self.weak_form_2.name: self.dz2,
                        self.weak_form_3.name: self.dz3}
        derivatives = {self.weak_form_1.name: (0, 0),
                       self.weak_form_2.name: (0, 0),
                       self.weak_form_3.name: (0, 0)}

        res = pi.simulate_systems(weak_forms, ics, self.dt, spat_domains, derivatives)
        win = pi.PgAnimatedPlot(res)

        if show_plots:
            pi.show(show_mpl=False)
            del win

    def test_triple_system_with_swm(self):
        """
        three coupled systems where the output at l4 is the input for
        a string with mass
        """
        weak_forms = [self.weak_form_1, self.weak_form_2, self.weak_form_3,
                      self.weak_form_4]
        ics = {self.weak_form_1.name: self.ic1,
               self.weak_form_2.name: self.ic2,
               self.weak_form_3.name: self.ic3,
               self.weak_form_4.name: self.ic4}
        spat_domains = {self.weak_form_1.name: self.dz1,
                        self.weak_form_2.name: self.dz2,
                        self.weak_form_3.name: self.dz3,
                        self.weak_form_4.name: self.dz4}
        derivatives = {self.weak_form_1.name: (0, 0),
                       self.weak_form_2.name: (0, 0),
                       self.weak_form_3.name: (0, 0),
                       self.weak_form_4.name: (1, 1)}

        res = pi.simulate_systems(weak_forms, ics, self.dt, spat_domains, derivatives)
        win = pi.PgAnimatedPlot(res)

        if show_plots:
            pi.show(show_mpl=False)
            del win

    def tearDown(self):
        pi.deregister_base("base_1")
        pi.deregister_base("base_2")
        pi.deregister_base("base_3")
        pi.deregister_base("base_4")


class RadFemTrajectoryTest(unittest.TestCase):
    """
    Test FEM simulation with pi.LagrangeFirstOrder and pi.LagrangeSecondOrder
    test functions and generic trajectory generator RadTrajectory for the
    reaction-advection-diffusion equation.
    """
    def setUp(self):
        self.param = [2., -1.5, -3., 2., .5]
        self.a2, self.a1, self.a0, self.alpha, self.beta = self.param

        self.l = 1.
        spatial_disc = 11
        self.dz = pi.Domain(bounds=(0, self.l), num=spatial_disc)

        self.T = 1.
        temporal_disc = 50
        self.dt = pi.Domain(bounds=(0, self.T), num=temporal_disc)

        # create test functions
        self.nodes_1 = pi.Domain(self.dz.bounds, num=spatial_disc)
        self.base1 = pi.LagrangeFirstOrder.cure_interval(self.nodes_1)
        pi.register_base("base_1", self.base_1)

        self.nodes_2 = pi.Domain(self.dz.bounds, num=spatial_disc)
        self.base2 = pi.LagrangeSecondOrder.cure_interval(self.nodes_1)
        pi.register_base("base_2", self.base_2)

    @unittest.skip  # needs border homogenization to work
    def test_dd(self):
        # TODO adopt this test case
        # trajectory
        bound_cond_type = 'dirichlet'
        actuation_type = 'dirichlet'
        u = parabolic.RadFeedForward(self.l,
                                     self.T,
                                     self.param,
                                     bound_cond_type, actuation_type)

        # derive state-space system
        rad_pde = parabolic.get_parabolic_dirichlet_weak_form("base_2",
                                                              "base_2",
                                                              u,
                                                              self.param,
                                                              self.dz.bounds)
        ce = sim.parse_weak_formulation(rad_pde)
        ss = sim.create_state_space(ce)

        # simulate system
        t, q = sim.simulate_state_space(ss,
                                        np.zeros(self.base_2.shape),
                                        self.dt)

        # display results
        if show_plots:
            eval_d = sim.evaluate_approximation("base_1",
                                                q,
                                                t,
                                                self.dz,
                                                spat_order=1)
            win1 = pi.PgAnimatedPlot([eval_d], title="Test")
            win2 = pi.PgSurfacePlot(eval_d)
            pi.show(show_mpl=False)

        # TODO add Test here
        return t, q

    @unittest.skip  # needs border homogenization to work
    def test_dd(self):
        # TODO adopt this test case
        # trajectory
        bound_cond_type = 'robin'
        actuation_type = 'dirichlet'
        u = parabolic.RadFeedForward(self.l, self.T, self.param, bound_cond_type, actuation_type)

        # integral terms
        int1 = pi.IntegralTerm(pi.Product(pi.TemporalDerivedFieldVariable("base_2", order=1),
                                          pi.TestFunction("base_2", order=0)), self.dz.bounds)
        int2 = pi.IntegralTerm(pi.Product(pi.SpatialDerivedFieldVariable("base_2", order=0),
                                          pi.TestFunction("base_2", order=2)), self.dz.bounds, -self.a2)
        int3 = pi.IntegralTerm(pi.Product(pi.SpatialDerivedFieldVariable("base_2", order=1),
                                          pi.TestFunction("base_2", order=0)), self.dz.bounds, -self.a1)
        int4 = pi.IntegralTerm(pi.Product(pi.SpatialDerivedFieldVariable("base_2", order=0),
                                          pi.TestFunction("base_2", order=0)), self.dz.bounds, -self.a0)
        # scalar terms from int 2
        s1 = pi.ScalarTerm(pi.Product(pi.SpatialDerivedFieldVariable("base_2", order=1, location=self.l),
                                      pi.TestFunction("base_2", order=0, location=self.l)), -self.a2)
        s2 = pi.ScalarTerm(pi.Product(pi.SpatialDerivedFieldVariable("base_2", order=0, location=0),
                                      pi.TestFunction("base_2", order=0, location=0)), self.a2 * self.alpha)
        s3 = pi.ScalarTerm(pi.Product(pi.SpatialDerivedFieldVariable("base_2", order=0, location=0),
                                      pi.TestFunction("base_2", order=1, location=0)), -self.a2)
        s4 = pi.ScalarTerm(pi.Product(pi.Input(u),
                                      pi.TestFunction("base_2", order=1, location=self.l)), self.a2)

        # derive state-space system
        rad_pde = sim.WeakFormulation([int1, int2, int3, int4, s1, s2, s3, s4], name="rad_pde")
        ce = sim.parse_weak_formulation(rad_pde)
        ss = sim.create_state_space(ce)

        # simulate system
        t, q = sim.simulate_state_space(ss, np.zeros(self.base_2.shape), self.dt)
        # TODO add test here

        return t, q

    @unittest.skip  # needs border homogenization to work
    def test_dr(self):
        # trajectory
        bound_cond_type = 'dirichlet'
        actuation_type = 'robin'
        u = parabolic.RadFeedForward(self.l,
                                                 self.T,
                                                 self.param,
                                                 bound_cond_type,
                                                 actuation_type)

        # integral terms
        int1 = pi.IntegralTerm(pi.Product(pi.TemporalDerivedFieldVariable("base_1", order=1),
                                          pi.TestFunction("base_1", order=0)), self.dz.bounds)
        int2 = pi.IntegralTerm(pi.Product(pi.SpatialDerivedFieldVariable("base_1", order=1),
                                          pi.TestFunction("base_1", order=1)), self.dz.bounds, self.a2)
        int3 = pi.IntegralTerm(pi.Product(pi.SpatialDerivedFieldVariable("base_1", order=0),
                                          pi.TestFunction("base_1", order=1)), self.dz.bounds, self.a1)
        int4 = pi.IntegralTerm(pi.Product(pi.SpatialDerivedFieldVariable("base_1", order=0),
                                          pi.TestFunction("base_1", order=0)), self.dz.bounds, -self.a0)
        # scalar terms from int 2
        s1 = pi.ScalarTerm(pi.Product(pi.SpatialDerivedFieldVariable("base_1", order=0, location=self.l),
                                      pi.TestFunction("base_1", order=0, location=self.l)), -self.a1)
        s2 = pi.ScalarTerm(pi.Product(pi.SpatialDerivedFieldVariable("base_1", order=0, location=self.l),
                                      pi.TestFunction("base_1", order=0, location=self.l)), self.a2 * self.beta)
        s3 = pi.ScalarTerm(pi.Product(pi.SpatialDerivedFieldVariable("base_1", order=1, location=0),
                                      pi.TestFunction("base_1", order=0, location=0)), self.a2)
        s4 = pi.ScalarTerm(pi.Product(pi.Input(u),
                                      pi.TestFunction("base_1", order=0, location=self.l)), -self.a2)
        rad_pde = sim.WeakFormulation([int1, int2, int3, int4, s1, s2, s3, s4], "rad_pde")

        # derive state-space system
        ce = sim.parse_weak_formulation(rad_pde)
        ss = sim.create_state_space(ce)

        # simulate system
        t, q = sim.simulate_state_space(ss, np.zeros(self.base_1.fractions.shape), self.dt)

        # check if (x'(0,t_end) - 1.) < 0.1
        self.assertLess(np.abs(self.base_1.fractions[0].derive(1)(sys.float_info.min) * (q[-1, 0] - q[-1, 1])) - 1, 0.1)

    def test_rr(self):
        # trajectory
        bound_cond_type = 'robin'
        actuation_type = 'robin'
        u = parabolic.RadFeedForward(self.l, self.T, self.param, bound_cond_type, actuation_type)

        # derive state-space system
        rad_pde, extra_labels = parabolic.get_parabolic_robin_weak_form("base_1", "base_1", u, self.param, self.dz.bounds)
        ce = sim.parse_weak_formulation(rad_pde)
        ss = sim.create_state_space(ce)

        # simulate system
        t, q = sim.simulate_state_space(ss, np.zeros(self.base_1.fractions.shape), self.dt)

        for lbl in extra_labels:
            pi.deregister_base(lbl)

        # check if (x(0,t_end) - 1.) < 0.1
        self.assertLess(np.abs(self.base_1.fractions[0].derive(0)(0) * q[-1, 0]) - 1, 0.1)

    def test_rr_const_trajectory(self):
        # TODO if it is only testing ConstantTrajectory should it better be moved to test_visualization ?
        # const trajectory simulation call test
        u = pi.ConstantTrajectory(1)

        # derive state-space system
        rad_pde, extra_labels = pi.parabolic.get_parabolic_robin_weak_form("base_1", "base_1", u, self.param, self.dz.bounds)
        ce = sim.parse_weak_formulation(rad_pde)
        ss = sim.create_state_space(ce)

        # simulate system
        t, q = sim.simulate_state_space(ss, np.zeros(self.base_1.fractions.shape), self.dt)

        # deregister extra labels
        for lbl in extra_labels:
            pi.deregister_base(lbl)

        # TODO add a Test here

    def tearDown(self):
        pi.deregister_base("base_1")
        pi.deregister_base("base_2")


class RadDirichletModalVsWeakFormulationTest(unittest.TestCase):
    """
    """
    def test_comparison(self):
        actuation_type = 'dirichlet'
        bound_cond_type = 'dirichlet'
        param = [1., -2., -1., None, None]
        adjoint_param = pi.SecondOrderEigenfunction.get_adjoint_problem(param)
        a2, a1, a0, _, _ = param

        l = 1.
        spatial_disc = 10
        dz = pi.Domain(bounds=(0, l), num=spatial_disc)

        t_end = 1.
        temporal_disc = 50
        dt = pi.Domain(bounds=(0, t_end), num=temporal_disc)

        (omega, eig_values
         ) = pi.SecondOrderDirichletEigenfunction.eigfreq_eigval_hint(
            param=param,
            l=dz.bounds[-1],
            n_roots=spatial_disc)
        norm_fak = np.ones(omega.shape) * np.sqrt(2)
        eig_base = pi.Base([pi.SecondOrderDirichletEigenfunction(omega[i],
                                                                 param,
                                                                 dz.bounds[-1],
                                                                 norm_fak[i])
                            for i in range(spatial_disc)])
        pi.register_base("eig_base", eig_base)

        adjoint_eig_base = pi.Base(
            [pi.SecondOrderDirichletEigenfunction(omega[i],
                                                  adjoint_param,
                                                  dz.bounds[-1],
                                                  norm_fak[i])
             for i in range(spatial_disc)])
        pi.register_base("adjoint_eig_base", adjoint_eig_base)

        # derive initial field variable x(z,0) and weights
        start_state = pi.Function(lambda z: 0., domain=(0, l))
        initial_weights = pi.project_on_base(start_state, adjoint_eig_base)

        # init trajectory
        u = parabolic.RadFeedForward(l,
                                     t_end,
                                     param,
                                     bound_cond_type,
                                     actuation_type)

        # ------------- determine (A,B) with weak-formulation (pyinduct)
        # derive sate-space system
        rad_pde = \
            parabolic.get_parabolic_dirichlet_weak_form("eig_base",
                                                        "adjoint_eig_base",
                                                        u, param, dz.bounds)
        ce = sim.parse_weak_formulation(rad_pde)
        ss_weak = sim.create_state_space(ce)

        # ------------- determine (A,B) with modal transformation
        a_mat = np.diag(eig_values)
        b_mat = -a2 * np.atleast_2d(
            [fraction(l) for fraction in adjoint_eig_base.derive(1).fractions]).T
        ss_modal = sim.StateSpace(a_mat, b_mat, input_handle=u)

        # check if ss_modal.(A,B) is close to ss_weak.(A,B)
        np.testing.assert_array_almost_equal(
            np.sort(np.linalg.eigvals(ss_weak.A[1])),
            np.sort(np.linalg.eigvals(ss_modal.A[1])))
        np.testing.assert_array_almost_equal(ss_weak.B[0][1], ss_modal.B[0][1])

        # TODO can the result be tested?

        # display results
        if show_plots:
            t, q = sim.simulate_state_space(ss_modal, initial_weights, dt)
            eval_d = sim.evaluate_approximation("eig_base",
                                                q,
                                                t,
                                                dz,
                                                spat_order=0)
            win2 = pi.PgSurfacePlot(eval_d)
            pi.show(show_mpl=False)

        pi.deregister_base("eig_base")
        pi.deregister_base("adjoint_eig_base")


class RadRobinModalVsWeakFormulationTest(unittest.TestCase):
    """
    """
    def test_comparison(self):
        actuation_type = 'robin'
        bound_cond_type = 'robin'
        param = [2., 1.5, -3., -1., -.5]
        adjoint_param = pi.SecondOrderEigenfunction.get_adjoint_problem(param)
        a2, a1, a0, alpha, beta = param

        l = 1.
        spatial_disc = 10
        dz = pi.Domain(bounds=(0, l), num=spatial_disc)

        t_end = 1.
        temporal_disc = 50
        dt = pi.Domain(bounds=(0, t_end), num=temporal_disc)
        n = 10

        eig_freq, eig_val = parabolic.compute_rad_robin_eigenfrequencies(param,
                                                                         l,
                                                                         n)

        init_eig_base = pi.Base(
            [pi.SecondOrderRobinEigenfunction(om, param, dz.bounds[-1])
             for om in eig_freq])

        init_adjoint_eig_base = pi.Base(
            [pi.SecondOrderRobinEigenfunction(om, adjoint_param, dz.bounds[-1])
             for om in eig_freq])

        # normalize eigenfunctions and adjoint eigenfunctions
        eig_base, adjoint_eig_base = pi.normalize_base(init_eig_base,
                                                       init_adjoint_eig_base)

        # register bases
        pi.register_base("eig_base", eig_base)
        pi.register_base("adjoint_eig_base", adjoint_eig_base)

        # derive initial field variable x(z,0) and weights
        start_state = pi.Function(lambda z: 0., domain=(0, l))
        initial_weights = pi.project_on_base(start_state, adjoint_eig_base)

        # init trajectory
        u = parabolic.RadFeedForward(l, t_end, param, bound_cond_type, actuation_type)

        # determine pair (A, B) by weak-formulation (pyinduct)
        rad_pde, extra_labels = parabolic.get_parabolic_robin_weak_form("eig_base", "adjoint_eig_base", u, param, dz.bounds)
        ce = sim.parse_weak_formulation(rad_pde)
        ss_weak = sim.create_state_space(ce)

        # determine pair (A, B) by modal transformation
        a_mat = np.diag(np.real_if_close(eig_val))
        b_mat = a2 * np.atleast_2d([fraction(l) for fraction in adjoint_eig_base.fractions]).T
        ss_modal = sim.StateSpace(a_mat, b_mat, input_handle=u)

        # check if ss_modal.(A,B) is close to ss_weak.(A,B)
        np.testing.assert_array_almost_equal(np.sort(np.linalg.eigvals(ss_weak.A[1])), np.sort(np.linalg.eigvals(ss_modal.A[1])),
                                             decimal=5)
        np.testing.assert_array_almost_equal(ss_weak.B[0][1], ss_modal.B[0][1])

        # display results
        if show_plots:
            t_end, q = sim.simulate_state_space(ss_modal, initial_weights, dt)
            eval_d = sim.evaluate_approximation("eig_base", q, t_end, dz, spat_order=1)
            win1 = pi.PgAnimatedPlot([eval_d], title="Test")
            win2 = pi.PgSurfacePlot(eval_d)
            pi.show(show_mpl=False)

        pi.deregister_base(extra_labels[0])
        pi.deregister_base(extra_labels[1])
        pi.deregister_base("eig_base")
        pi.deregister_base("adjoint_eig_base")


class EvaluateApproximationTestCase(unittest.TestCase):
    def setUp(self):
        self.node_cnt = 5
        self.time_step = 1e-1
        self.dates = pi.Domain((0, 10), step=self.time_step)
        self.spat_dom = pi.Domain((0, 1), num=self.node_cnt)

        # create initial functions
        self.nodes = pi.Domain(self.spat_dom.bounds, num=self.node_cnt)
        self.funcs = pi.LagrangeSecondOrder.cure_interval(self.nodes)
        pi.register_base("approx_funcs", self.funcs, overwrite=True)

        # create a slow rising, nearly horizontal line
        self.weights = np.array(list(range(
            self.node_cnt * self.dates.points.size))).reshape(
            (self.dates.points.size, len(self.nodes)))

    def test_eval_helper(self):
        eval_data = sim.evaluate_approximation("approx_funcs",
                                               self.weights,
                                               self.dates,
                                               self.spat_dom,
                                               1)
        if show_plots:
            p = pi.PgAnimatedPlot(eval_data)
            pi.show(show_mpl=False)
            del p

    def tearDown(self):
        pass


class SetDominantLabel(unittest.TestCase):

    def setUp(self):
        self.limits = (0, 1)
        domain = pi.Domain(bounds=self.limits, num=100)
        nodes = pi.Domain(domain.bounds, num=3)
        base = pi.LagrangeSecondOrder.cure_interval(nodes)

        pi.register_base("base_1", base)
        pi.register_base("base_2", base)
        pi.register_base("base_3", base)

        self.x1 = pi.FieldVariable("base_1")
        self.psi_1 = pi.TestFunction("base_1")
        self.x2 = pi.FieldVariable("base_2")
        self.psi_2 = pi.TestFunction("base_2")
        self.x3 = pi.FieldVariable("base_3")
        self.psi_3 = pi.TestFunction("base_3")

    def test_valid(self):
        weak_form_1 = pi.WeakFormulation([
            pi.IntegralTerm(
                pi.Product(self.x1.derive(temp_order=1), self.psi_1),
                limits=self.limits)],
            name="sys_1")
        weak_form_2 = pi.WeakFormulation([
            pi.IntegralTerm(
                pi.Product(self.x2.derive(temp_order=1), self.psi_2),
                limits=self.limits),
        ], name="sys_2")
        weak_form_3 = pi.WeakFormulation([
            pi.IntegralTerm(pi.Product(self.x3.derive(temp_order=1), self.psi_3),
                            limits=self.limits),
            pi.ScalarTerm(pi.Product(self.x3(0), self.psi_3(0))),
        ], name="sys_3")
        ces = sim.parse_weak_formulations([weak_form_1, weak_form_2,
                                           weak_form_3])
        sim.set_dominant_labels(ces)
        for i, ce in zip(range(3), ces):
            self.assertEqual("base_{}".format(i + 1), ce.dominant_lbl)

    def test_non_valid_algebraic(self):
        weak_form_1 = pi.WeakFormulation([
            pi.IntegralTerm(
                pi.Product(self.x1.derive(temp_order=0), self.psi_1),
                limits=self.limits),
            pi.ScalarTerm(pi.Product(self.x2(0), self.psi_1(0))),
        ], name="sys_1")
        weak_form_2 = pi.WeakFormulation([
            pi.IntegralTerm(pi.Product(self.x2.derive(temp_order=1), self.psi_2),
                            limits=self.limits),
            pi.ScalarTerm(pi.Product(self.x2(0), self.psi_2(0))),
        ], name="sys_2")
        ces = sim.parse_weak_formulations([weak_form_1, weak_form_2])
        self.assertRaises(ValueError, sim.set_dominant_labels, ces)

    def test_non_valid_max_order_uniqueness(self):
        weak_form_1 = pi.WeakFormulation([
            pi.IntegralTerm(
                pi.Product(self.x1.derive(temp_order=4), self.psi_1),
                limits=self.limits),
        ], name="sys_1")
        weak_form_2 = pi.WeakFormulation([
            pi.IntegralTerm(
                pi.Product(self.x1.derive(temp_order=4), self.psi_1),
                limits=self.limits),
            pi.IntegralTerm(
                pi.Product(self.x2.derive(temp_order=1), self.psi_2),
                limits=self.limits),
        ], name="sys_2")
        weak_form_3 = pi.WeakFormulation([
            pi.IntegralTerm(pi.Product(self.x3.derive(temp_order=1), self.psi_3),
                            limits=self.limits),
            pi.ScalarTerm(pi.Product(self.x3(0), self.psi_3(0))),
        ], name="sys_3")
        ces = sim.parse_weak_formulations([weak_form_1, weak_form_2,
                                           weak_form_3])
        self.assertRaises(ValueError, sim.set_dominant_labels, ces)

    def test_non_valid_not_enough_labels(self):
        weak_form_1 = pi.WeakFormulation([
            pi.IntegralTerm(
                pi.Product(self.x1.derive(temp_order=4), self.psi_1),
                limits=self.limits),
        ], name="sys_1")
        weak_form_2 = pi.WeakFormulation([
            pi.IntegralTerm(
                pi.Product(self.x1.derive(temp_order=4), self.psi_1),
                limits=self.limits),
        ], name="sys_2")
        ces = sim.parse_weak_formulations([weak_form_1, weak_form_2])
        self.assertRaises(ValueError, sim.set_dominant_labels, ces)

    def test_wrong_dominant_labels(self):
        weak_form_1 = pi.WeakFormulation([
            pi.IntegralTerm(
                pi.Product(self.x1.derive(temp_order=4), self.psi_1),
                limits=self.limits),
        ], name="sys_1", dominant_lbl="base_2")
        weak_form_2 = pi.WeakFormulation([
            pi.IntegralTerm(
                pi.Product(self.x2.derive(temp_order=4), self.psi_1),
                limits=self.limits),
        ], name="sys_2", dominant_lbl="base_1")
        ces = sim.parse_weak_formulations([weak_form_1, weak_form_2])
        self.assertWarns(UserWarning, sim.set_dominant_labels, ces)

    def tearDown(self):
        pi.deregister_base("base_1")
        pi.deregister_base("base_2")
        pi.deregister_base("base_3")
