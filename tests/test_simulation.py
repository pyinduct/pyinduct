from __future__ import division
import unittest
import os
from cPickle import dumps

import numpy as np
import sys

from pyinduct import register_functions, eigenfunctions as ef,\
    core as cr, simulation as sim, utils as ut, visualization as vis, trajectory as tr
import pyinduct.placeholder as ph
import pyinduct as pi
import pyinduct.shapefunctions


if any([arg == 'discover' for arg in sys.argv]):
    show_plots = False
else:
    show_plots = True
    # show_plots = False

if show_plots:
    import pyqtgraph as pg
    app = pg.QtGui.QApplication([])
else:
    app = None


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


class ParseTest(unittest.TestCase):

    def setUp(self):
        # scalars
        self.scalars = ph.Scalars(np.vstack(range(3)))

        # inputs
        self.u = np.sin
        self.input = ph.Input(self.u)  # control input

        # TestFunctions
        nodes, self.ini_funcs = pyinduct.shapefunctions.cure_interval(pyinduct.shapefunctions.LagrangeFirstOrder,
                                                                      (0, 1), node_count=3)
        register_functions("ini_funcs", self.ini_funcs, overwrite=True)
        self.phi = ph.TestFunction("ini_funcs")  # eigenfunction or something else
        self.phi_at0 = ph.TestFunction("ini_funcs", location=0)  # eigenfunction or something else
        self.phi_at1 = ph.TestFunction("ini_funcs", location=1)  # eigenfunction or something else
        self.dphi = ph.TestFunction("ini_funcs", order=1)  # eigenfunction or something else
        self.dphi_at1 = ph.TestFunction("ini_funcs", order=1, location=1)  # eigenfunction or something else

        # FieldVars
        self.field_var = ph.FieldVariable("ini_funcs")
        self.odd_weight_field_var = ph.FieldVariable("ini_funcs", weight_label="special_weights")
        self.field_var_at1 = ph.FieldVariable("ini_funcs", location=1)
        self.field_var_dz = ph.SpatialDerivedFieldVariable("ini_funcs", 1)
        self.field_var_dz_at1 = ph.SpatialDerivedFieldVariable("ini_funcs", 1, location=1)
        self.field_var_ddt = ph.TemporalDerivedFieldVariable("ini_funcs", 2)
        self.field_var_ddt_at0 = ph.TemporalDerivedFieldVariable("ini_funcs", 2, location=0)
        self.field_var_ddt_at1 = ph.TemporalDerivedFieldVariable("ini_funcs", 2, location=1)

        # create all possible kinds of input variables
        self.input_term1 = ph.ScalarTerm(ph.Product(self.phi_at1, self.input))
        self.input_term1_swapped = ph.ScalarTerm(ph.Product(self.input, self.phi_at1))

        self.input_term2 = ph.ScalarTerm(ph.Product(self.dphi_at1, self.input))
        self.func_term = ph.ScalarTerm(self.phi_at1)

        self.field_term_at1 = ph.ScalarTerm(self.field_var_at1)
        self.field_term_dz_at1 = ph.ScalarTerm(self.field_var_dz_at1)
        self.field_term_ddt_at1 = ph.ScalarTerm(self.field_var_ddt_at1)

        self.field_int = ph.IntegralTerm(self.field_var, (0, 1))
        self.field_dz_int = ph.IntegralTerm(self.field_var_dz, (0, 1))
        self.field_ddt_int = ph.IntegralTerm(self.field_var_ddt, (0, 1))

        self.prod_term_fs_at1 = ph.ScalarTerm(
            ph.Product(self.field_var_at1, self.scalars))
        self.prod_int_fs = ph.IntegralTerm(ph.Product(self.field_var, self.scalars), (0, 1))
        self.prod_int_f_f = ph.IntegralTerm(ph.Product(self.field_var, self.phi), (0, 1))
        self.prod_int_f_f_swapped = ph.IntegralTerm(ph.Product(self.phi, self.field_var), (0, 1))
        self.prod_int_f_at1_f = ph.IntegralTerm(
            ph.Product(self.field_var_at1, self.phi), (0, 1))
        self.prod_int_f_f_at1 = ph.IntegralTerm(
            ph.Product(self.field_var, self.phi_at1), (0, 1))
        self.prod_term_f_at1_f_at1 = ph.ScalarTerm(
            ph.Product(self.field_var_at1, self.phi_at1))

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
        self.assertEqual(terms[0], None)  # E0
        self.assertEqual(terms[1], None)  # f
        self.assertTrue(np.allclose(terms[2][0], np.array([[0], [-2], [2]])))  # g

    def test_TestFunction_term(self):
        wf = sim.WeakFormulation(self.func_term)
        sim.parse_weak_formulation(wf)

    def test_FieldVariable_term(self):
        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.field_term_at1)).get_terms()
        self.assertTrue(np.allclose(terms[0][0], np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.field_int)).get_terms()
        self.assertTrue(np.allclose(terms[0][0], np.array([[0.25, 0.5, 0.25], [0.25, 0.5, 0.25], [.25, .5, .25]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.field_term_dz_at1)).get_terms()
        self.assertTrue(np.allclose(terms[0][0], np.array([[0, -2, 2], [0, -2, 2], [0, -2, 2]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.field_dz_int)).get_terms()
        self.assertTrue(np.allclose(terms[0][0], np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.field_term_ddt_at1)).get_terms()
        self.assertTrue(np.allclose(terms[0][0], np.zeros((3, 3))))
        self.assertTrue(np.allclose(terms[0][1], np.zeros((3, 3))))
        self.assertTrue(np.allclose(terms[0][2], np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.field_ddt_int)).get_terms()
        self.assertTrue(np.allclose(terms[0][0], np.zeros((3, 3))))
        self.assertTrue(np.allclose(terms[0][1], np.zeros((3, 3))))
        self.assertTrue(np.allclose(terms[0][2], np.array([[0.25, 0.5, 0.25], [0.25, 0.5, 0.25], [.25, .5, .25]])))

    def test_Product_term(self):
        # TODO create test functionality that will automatically check if Case is also valid for swapped arguments
        # terms = sim.parse_weak_formulation(sim.WeakFormulation(self.prod_term_fs_at1)).get_terms()
        # self.assertTrue(np.allclose(terms[0][0], np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.prod_int_fs)).get_terms()
        self.assertTrue(np.allclose(terms[0][0], np.array([[0, 0, 0], [0.25, .5, .25], [.5, 1, .5]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.prod_int_f_f)).get_terms()
        self.assertTrue(np.allclose(terms[0][0], np.array([[1/6, 1/12, 0], [1/12, 1/3, 1/12], [0, 1/12, 1/6]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.prod_int_f_f_swapped)).get_terms()
        self.assertTrue(np.allclose(terms[0][0], np.array([[1/6, 1/12, 0], [1/12, 1/3, 1/12], [0, 1/12, 1/6]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.prod_int_f_at1_f)).get_terms()
        self.assertTrue(np.allclose(terms[0][0], np.array([[0, 0, 0.25], [0, 0, 0.5], [0, 0, .25]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.prod_int_f_f_at1)).get_terms()
        self.assertTrue(np.allclose(terms[0][0], np.array([[0, 0, 0], [0, 0, 0], [0.25, 0.5, .25]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.prod_term_f_at1_f_at1)).get_terms()
        self.assertTrue(np.allclose(terms[0][0], np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])))

        # more complex terms
        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.prod_int_fddt_f)).get_terms()
        self.assertTrue(np.allclose(terms[0][0], np.zeros((3, 3))))
        self.assertTrue(np.allclose(terms[0][1], np.zeros((3, 3))))
        self.assertTrue(np.allclose(terms[0][2], np.array([[1/6, 1/12, 0], [1/12, 1/3, 1/12], [0, 1/12, 1/6]])))
        self.assertEqual(terms[1], None)  # f
        self.assertEqual(terms[2], None)  # g

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.prod_term_fddt_at0_f_at0)).get_terms()
        self.assertTrue(np.allclose(terms[0][0], np.zeros((3, 3))))
        self.assertTrue(np.allclose(terms[0][1], np.zeros((3, 3))))
        self.assertTrue(np.allclose(terms[0][2], np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])))
        self.assertEqual(terms[1], None)  # f
        self.assertEqual(terms[2], None)  # g

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.spat_int)).get_terms()
        self.assertTrue(np.allclose(terms[0][0], np.array([[2, -2, 0], [-2, 4, -2], [0, -2, 2]])))
        self.assertEqual(terms[1], None)  # f
        self.assertEqual(terms[2], None)  # g

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.spat_int_asymmetric)).get_terms()
        self.assertTrue(np.allclose(terms[0][0], np.array([[-.5, .5, 0], [-.5, 0, .5], [0, -.5, .5]])))
        self.assertEqual(terms[1], None)  # f
        self.assertEqual(terms[2], None)  # g

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.prod_term_f_at1_dphi_at1)).get_terms()
        self.assertTrue(np.allclose(terms[0][0], np.array([[0, 0, 0], [0, 0, -2], [0, 0, 2]])))
        self.assertEqual(terms[1], None)  # f
        self.assertEqual(terms[2], None)  # g

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.input_term1)).get_terms()
        self.assertEqual(terms[0], None)  # E
        self.assertEqual(terms[1], None)  # f
        self.assertTrue(np.allclose(terms[2][0], np.array([[0], [0], [1]])))

        terms = sim.parse_weak_formulation(sim.WeakFormulation(self.input_term1_swapped)).get_terms()
        self.assertEqual(terms[0], None)  # E
        self.assertEqual(terms[1], None)  # f
        self.assertTrue(np.allclose(terms[2][0], np.array([[0], [0], [1]])))

    def test_alternating_weights(self):
        self.assertRaises(ValueError, sim.parse_weak_formulation,
                          sim.WeakFormulation([self.alternating_weights_term, self.field_int]))


class StateSpaceTests(unittest.TestCase):

    def setUp(self):
        # enter string with mass equations
        self.u = cr.Function(lambda x: 0)
        interval = (0, 1)
        nodes, ini_funcs = pyinduct.shapefunctions.cure_interval(pyinduct.shapefunctions.LagrangeFirstOrder, interval, node_count=3)
        register_functions("init_funcs", ini_funcs, overwrite=True)
        int1 = ph.IntegralTerm(
            ph.Product(ph.TemporalDerivedFieldVariable("init_funcs", 2),
                       ph.TestFunction("init_funcs")), interval)
        s1 = ph.ScalarTerm(
            ph.Product(ph.TemporalDerivedFieldVariable("init_funcs", 2, location=0),
                       ph.TestFunction("init_funcs", location=0)))
        int2 = ph.IntegralTerm(
            ph.Product(ph.SpatialDerivedFieldVariable("init_funcs", 1),
                       ph.TestFunction("init_funcs", order=1)), interval)
        s2 = ph.ScalarTerm(
            ph.Product(ph.Input(self.u), ph.TestFunction("init_funcs", location=1)), -1)

        string_pde = sim.WeakFormulation([int1, s1, int2, s2])
        self.cf = sim.parse_weak_formulation(string_pde)

    def test_convert_to_state_space(self):
        ss = self.cf.convert_to_state_space()
        self.assertTrue(np.allclose(ss.A, np.array([[0, 0, 0, 1, 0, 0],
                                                    [0, 0, 0, 0, 1, 0],
                                                    [0, 0, 0, 0, 0, 1],
                                                    [-2.25, 3, -.75, 0, 0, 0],
                                                    [7.5, -18, 10.5, 0, 0, 0],
                                                    [-3.75, 21, -17.25, 0, 0, 0]])))
        self.assertTrue(np.allclose(ss.B, np.array([[0], [0], [0], [0.125], [-1.75], [6.875]])))
        self.assertEqual(self.cf.input_function, self.u)


class StringMassTest(unittest.TestCase):

    def setUp(self):

        z_start = 0
        z_end = 1
        t_start = 0
        t_end = 10
        self.z_step = 0.1
        self.t_step = 0.01
        self.t_values = np.arange(t_start, t_end+self.t_step, self.t_step)
        self.z_values = np.arange(z_start, z_end+self.z_step, self.z_step)
        self.node_distance = 0.1
        self.mass = 2.0
        self.order = 8
        self.temp_interval = (t_start, t_end)
        self.spat_interval = (z_start, z_end)

        self.u = tr.FlatString(0, 10, 0, 5, m=self.mass)

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
        nodes, ini_funcs = pyinduct.shapefunctions.cure_interval(pyinduct.shapefunctions.LagrangeSecondOrder, self.spat_interval, node_count=10)
        register_functions("init_funcs", ini_funcs, overwrite=True)
        int1 = ph.IntegralTerm(
            ph.Product(ph.TemporalDerivedFieldVariable("init_funcs", 2),
                       ph.TestFunction("init_funcs")), self.spat_interval)
        s1 = ph.ScalarTerm(
            ph.Product(ph.TemporalDerivedFieldVariable("init_funcs", 2, location=0),
                       ph.TestFunction("init_funcs", location=0)), scale=self.mass)
        int2 = ph.IntegralTerm(
            ph.Product(ph.SpatialDerivedFieldVariable("init_funcs", 1),
                       ph.TestFunction("init_funcs", order=1)), self.spat_interval)
        s2 = ph.ScalarTerm(
            ph.Product(ph.Input(self.u), ph.TestFunction("init_funcs", location=1)), -1)

        # derive sate-space system
        string_pde = sim.WeakFormulation([int1, s1, int2, s2], name="fem_test")
        self.cf = sim.parse_weak_formulation(string_pde)
        ss = self.cf.convert_to_state_space()

        # generate initial conditions for weights
        q0 = np.array([cr.project_on_base(self.ic[idx], ini_funcs) for idx in range(2)]).flatten()

        # simulate
        t, q = sim.simulate_state_space(ss, self.cf.input_function, q0, self.temp_interval)

        # calculate result data
        eval_data = []
        for der_idx in range(2):
            eval_data.append(ut.evaluate_approximation(q[:, der_idx*ini_funcs.size:(der_idx+1)*ini_funcs.size],
                                                       "init_funcs", t, self.spat_interval, self.z_step))
            eval_data[-1].name = "{0}{1}".format(self.cf.name, "_"+"".join(["d" for x in range(der_idx)])
                                                               + "t" if der_idx > 0 else "")

        # display results
        if show_plots:
            win = vis.PgAnimatedPlot(eval_data[:2], title="fem approx and derivative")
            win2 = vis.PgSurfacePlot(eval_data[0])
            app.exec_()

        # TODO add test expression
        # TODO dump in pyinduct/tests/ressources
        file_path = os.sep.join(["resources", "test_data.res"])
        if not os.path.isdir("resources"):
            os.makedirs("resources")
        with open(file_path, "w") as f:
            f.write(dumps(eval_data))

    def test_modal(self):
        order = 8

        def char_eq(w):
            return w * (np.sin(w) + self.mass * w * np.cos(w))

        def phi_k_factory(freq, derivative_order=0):
            def eig_func(z):
                return np.cos(freq * z) - self.mass * freq * np.sin(freq * z)

            def eig_func_dz(z):
                return -freq * (np.sin(freq * z) + self.mass * freq * np.cos(freq * z))

            def eig_func_ddz(z):
                return freq ** 2 * (-np.cos(freq * z) + self.mass * freq * np.sin(freq * z))

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
        print eig_frequencies

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

        eig_vectors = []
        for n in range(order):
            eig_vectors.append(SWMFunctionVector(cr.Function(phi_k_factory(eig_frequencies[n]),
                                                             derivative_handles=[
                                                                 phi_k_factory(eig_frequencies[n], der_order)
                                                                 for der_order in range(1, 3)],
                                                             domain=self.spat_interval,
                                                             nonzero=self.spat_interval),
                                                 phi_k_factory(eig_frequencies[n])(0)))

        # normalize eigen vectors
        norm_eig_vectors = [cr.normalize_function(vec) for vec in eig_vectors]
        norm_eig_funcs = np.array([vec.func for vec in norm_eig_vectors])
        register_functions("norm_eig_funcs", norm_eig_funcs, overwrite=True)

        norm_eig_funcs[0](1)

        # debug print eigenfunctions
        if 0:
            func_vals = []
            for vec in eig_vectors:
                func_vals.append(np.vectorize(vec.func)(self.z_values))

            norm_func_vals = []
            for func in norm_eig_funcs:
                norm_func_vals.append(np.vectorize(func)(self.z_values))

            clrs = ["r", "g", "b", "c", "m", "y", "k", "w"]
            for n in range(1, order + 1, len(clrs)):
                pw_phin_k = pg.plot(title="phin_k for k in [{0}, {1}]".format(n, min(n + len(clrs), order)))
                for k in range(len(clrs)):
                    if k + n > order:
                        break
                    pw_phin_k.plot(x=self.z_values, y=norm_func_vals[n + k - 1], pen=clrs[k])

            app.exec_()

        # create terms of weak formulation
        terms = [ph.IntegralTerm(ph.Product(ph.FieldVariable("norm_eig_funcs", order=(2, 0)),
                                            ph.TestFunction("norm_eig_funcs")),
                                 self.spat_interval, scale=-1),
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
                                 self.spat_interval)]
        modal_pde = sim.WeakFormulation(terms, name="swm_lib-modal")
        eval_data = sim.simulate_system(modal_pde, self.ic, self.temp_interval, self.t_step,
                                        self.spat_interval, self.z_step)

        # display results
        if show_plots:
            win = vis.PgAnimatedPlot(eval_data[0:2], title="modal approx and derivative")
            win2 = vis.PgSurfacePlot(eval_data[0])
            app.exec_()

        # TODO add test expression

    def tearDown(self):
        pass

    # TODO test "forbidden" terms like derivatives on the borders


class RadFemTrajectoryTest(unittest.TestCase):
    """
    Tests FEM simulation with cr.LagrangeFirstOrder and cr.LagrangeSecondOrder testfunctions
    and generic trajectory generator tr.RadTrajectory
    for the reaction-advection-diffusion equation.
    """
    def setUp(self):
        pass

    def test_it(self):
        param = [2., -1.5, -3., 2., .5]
        a2, a1, a0, alpha, beta = param
        l = 1.
        spatial_domain = (0, l)
        spatial_disc = 10
        T = 1.
        temporal_domain = (0, T)
        temporal_disc = 2e2

        # create test functions
        nodes_1, ini_funcs_1 = pyinduct.shapefunctions.cure_interval(pyinduct.shapefunctions.LagrangeFirstOrder,
                                                spatial_domain,
                                                node_count=spatial_disc)
        register_functions("init_funcs_1", ini_funcs_1, overwrite=True)
        nodes_2, ini_funcs_2 = pyinduct.shapefunctions.cure_interval(pyinduct.shapefunctions.LagrangeSecondOrder,
                                                spatial_domain,
                                                node_count=int(spatial_disc))
        register_functions("init_funcs_2", ini_funcs_2, overwrite=True)

        def test_dd():
            # trajectory
            bound_cond_type = 'dirichlet'
            actuation_type = 'dirichlet'
            u = tr.RadTrajectory(l, T, param, bound_cond_type, actuation_type)
            # derive state-space system
            rad_pde = ut.get_parabolic_dirichlet_weak_form("init_funcs_2", "init_funcs_2", u, param, spatial_domain)
            cf = sim.parse_weak_formulation(rad_pde)
            ss = cf.convert_to_state_space()
            # simulate system
            t, q = sim.simulate_state_space(ss, cf.input_function, np.zeros(ini_funcs_2.shape),
                                            temporal_domain, time_step=T/temporal_disc)

            return t, q

        def test_rd():
            # trajectory
            bound_cond_type = 'robin'
            actuation_type = 'dirichlet'
            u = tr.RadTrajectory(l, T, param, bound_cond_type, actuation_type)
            # integral terms
            int1 = ph.IntegralTerm(ph.Product(ph.TemporalDerivedFieldVariable("init_funcs_2", order=1),
                                              ph.TestFunction("init_funcs_2", order=0)), spatial_domain)
            int2 = ph.IntegralTerm(ph.Product(ph.SpatialDerivedFieldVariable("init_funcs_2", order=0),
                                              ph.TestFunction("init_funcs_2", order=2)), spatial_domain, -a2)
            int3 = ph.IntegralTerm(ph.Product(ph.SpatialDerivedFieldVariable("init_funcs_2", order=1),
                                              ph.TestFunction("init_funcs_2", order=0)), spatial_domain, -a1)
            int4 = ph.IntegralTerm(ph.Product(ph.SpatialDerivedFieldVariable("init_funcs_2", order=0),
                                              ph.TestFunction("init_funcs_2", order=0)), spatial_domain, -a0)
            # scalar terms from int 2
            s1 = ph.ScalarTerm(ph.Product(ph.SpatialDerivedFieldVariable("init_funcs_2", order=1, location=l),
                                          ph.TestFunction("init_funcs_2", order=0, location=l)), -a2)
            s2 = ph.ScalarTerm(ph.Product(ph.SpatialDerivedFieldVariable("init_funcs_2", order=0, location=0),
                                          ph.TestFunction("init_funcs_2", order=0, location=0)), a2*alpha)
            s3 = ph.ScalarTerm(ph.Product(ph.SpatialDerivedFieldVariable("init_funcs_2", order=0, location=0),
                                          ph.TestFunction("init_funcs_2", order=1, location=0)), -a2)
            s4 = ph.ScalarTerm(ph.Product(ph.Input(u),
                                          ph.TestFunction("init_funcs_2", order=1, location=l)), a2)
            # derive state-space system
            rad_pde = sim.WeakFormulation([int1, int2, int3, int4, s1, s2, s3, s4])
            cf = sim.parse_weak_formulation(rad_pde)
            ss = cf.convert_to_state_space()
            # simulate system
            t, q = sim.simulate_state_space(ss, cf.input_function, np.zeros(ini_funcs_2.shape),
                                            temporal_domain, time_step=T/temporal_disc)

            return t, q

        def test_dr():
            # trajectory
            bound_cond_type = 'dirichlet'
            actuation_type = 'robin'
            u = tr.RadTrajectory(l, T, param, bound_cond_type, actuation_type)
            # integral terms
            int1 = ph.IntegralTerm(ph.Product(ph.TemporalDerivedFieldVariable("init_funcs_1", order=1),
                                              ph.TestFunction("init_funcs_1", order=0)), spatial_domain)
            int2 = ph.IntegralTerm(ph.Product(ph.SpatialDerivedFieldVariable("init_funcs_1", order=1),
                                              ph.TestFunction("init_funcs_1", order=1)), spatial_domain, a2)
            int3 = ph.IntegralTerm(ph.Product(ph.SpatialDerivedFieldVariable("init_funcs_1", order=0),
                                              ph.TestFunction("init_funcs_1", order=1)), spatial_domain, a1)
            int4 = ph.IntegralTerm(ph.Product(ph.SpatialDerivedFieldVariable("init_funcs_1", order=0),
                                              ph.TestFunction("init_funcs_1", order=0)), spatial_domain, -a0)
            # scalar terms from int 2
            s1 = ph.ScalarTerm(ph.Product(ph.SpatialDerivedFieldVariable("init_funcs_1", order=0, location=l),
                                          ph.TestFunction("init_funcs_1", order=0, location=l)), -a1)
            s2 = ph.ScalarTerm(ph.Product(ph.SpatialDerivedFieldVariable("init_funcs_1", order=0, location=l),
                                          ph.TestFunction("init_funcs_1", order=0, location=l)), a2*beta)
            s3 = ph.ScalarTerm(ph.Product(ph.SpatialDerivedFieldVariable("init_funcs_1", order=1, location=0),
                                          ph.TestFunction("init_funcs_1", order=0, location=0)), a2)
            s4 = ph.ScalarTerm(ph.Product(ph.Input(u),
                                          ph.TestFunction("init_funcs_1", order=0, location=l)), -a2)
            # derive state-space system
            rad_pde = sim.WeakFormulation([int1, int2, int3, int4, s1, s2, s3, s4])
            cf = sim.parse_weak_formulation(rad_pde)
            ss = cf.convert_to_state_space()

            # simulate system
            t, q = sim.simulate_state_space(ss, cf.input_function, np.zeros(ini_funcs_1.shape),
                                            temporal_domain, time_step=T/temporal_disc)

            # check if (x'(0,t_end) - 1.) < 0.1
            self.assertLess(np.abs(ini_funcs_1[0].derive(1)(sys.float_info.min) * (q[-1, 0] - q[-1, 1])) - 1, 0.1)
            return t, q

        def test_rr():
            # trajectory
            bound_cond_type = 'robin'
            actuation_type = 'robin'
            u = tr.RadTrajectory(l, T, param, bound_cond_type, actuation_type)
            # derive state-space system
            rad_pde = ut.get_parabolic_robin_weak_form("init_funcs_1", "init_funcs_1", u, param, spatial_domain)
            cf = sim.parse_weak_formulation(rad_pde)
            ss = cf.convert_to_state_space()

            # simulate system
            t, q = sim.simulate_state_space(ss, cf.input_function, np.zeros(ini_funcs_1.shape),
                                            temporal_domain, time_step=T/temporal_disc)

            # check if (x(0,t_end) - 1.) < 0.1
            self.assertLess(np.abs(ini_funcs_1[0].derive(0)(0) * q[-1, 0]) - 1, 0.1)
            return t, q

        t, q = test_dr()
        _, _ = test_rr()
        # TODO: fit LagrangeSecondOrder to test_dd and test_rd
        # t, q = test_dd()
        # t, q = test_rd()

        # display results
        if show_plots:
            eval_d = ut.evaluate_approximation(q, "init_funcs_1", t, spatial_domain, l/spatial_disc)
            # eval_dd = ut.evaluate_approximation(q, np.array([i.derive(1) for i in init_funcs_1]), t, spatial_domain, l/spatial_disc)
            win1 = vis.PgAnimatedPlot([eval_d], title="Test")
            win2 = vis.PgSurfacePlot(eval_d)
            app.exec_()


class RadDirichletModalVsWeakFormulationTest(unittest.TestCase):
    """
    """
    def setUp(self):
        pass

    def test_it(self):
        actuation_type = 'dirichlet'
        bound_cond_type = 'dirichlet'
        param = [1., -2., -1., None, None]
        adjoint_param = ef.get_adjoint_rad_evp_param(param)
        a2, a1, a0, _, _ = param
        l = 1.; spatial_domain = (0, l); spatial_disc = 10
        T = 1.; temporal_domain = (0, T); temporal_disc = 1e2
        n = 10

        omega = np.array([(i+1)*np.pi/l for i in xrange(n)])
        eig_values = a0 - a2*omega**2 - a1**2/4./a2
        norm_fak = np.ones(omega.shape)*np.sqrt(2)
        eig_funcs = np.array([ef.SecondOrderDirichletEigenfunction(omega[i], param, spatial_domain, norm_fak[i]) for i in range(n)])
        register_functions("eig_funcs", eig_funcs, overwrite=True)
        adjoint_eig_funcs = np.array([ef.SecondOrderDirichletEigenfunction(omega[i], adjoint_param, spatial_domain, norm_fak[i]) for i in range(n)])
        register_functions("adjoint_eig_funcs", adjoint_eig_funcs, overwrite=True)

        # derive initial field variable x(z,0) and weights
        start_state = cr.Function(lambda z: 0., domain=(0, l))
        initial_weights = cr.project_on_base(start_state, adjoint_eig_funcs)

        # init trajectory
        u = tr.RadTrajectory(l, T, param, bound_cond_type, actuation_type)

        # determine (A,B) with weak-formulation (pyinduct)

        # derive sate-space system
        rad_pde = ut.get_parabolic_dirichlet_weak_form("eig_funcs", "adjoint_eig_funcs", u, param, spatial_domain)
        cf = sim.parse_weak_formulation(rad_pde)
        ss_weak = cf.convert_to_state_space()

        # determine (A,B) with modal-transfomation
        A = np.diag(eig_values)
        B = -a2*np.array([adjoint_eig_funcs[i].derive()(l) for i in xrange(n)])
        ss_modal = sim.StateSpace("eig_funcs", A, B)

        # TODO: resolve the big tolerance (rtol=3e-01) between ss_modal.A and ss_weak.A
        # check if ss_modal.(A,B) is close to ss_weak.(A,B)
        self.assertTrue(np.allclose(np.sort(np.linalg.eigvals(ss_weak.A)), np.sort(np.linalg.eigvals(ss_modal.A)), rtol=3e-01, atol=0.))
        self.assertTrue(np.allclose(np.array([i[0] for i in ss_weak.B]), ss_modal.B))

        # display results
        if show_plots:
            t, q = sim.simulate_state_space(ss_modal, u, initial_weights, temporal_domain, time_step=T/temporal_disc)
            eval_d = ut.evaluate_approximation(q, "eig_funcs", t, spatial_domain, l/spatial_disc)
            # eval_dd = ut.evaluate_approximation(q, np.array([i.derive(1) for i in rad_eig_funcs]), t, spatial_domain, l/spatial_disc)
            # win1 = vis.PgAnimatedPlot([eval_d, eval_dd], title="Test")
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
        adjoint_param = ef.get_adjoint_rad_evp_param(param)
        a2, a1, a0, alpha, beta = param
        l = 1.; spatial_domain = (0, l); spatial_disc = 10
        T = 1.; temporal_domain = (0, T); temporal_disc = 1e2
        n = 10

        eig_freq, eig_val = ef.compute_rad_robin_eigenfrequencies(param, l, n)

        init_eig_funcs = np.array([ef.SecondOrderRobinEigenfunction(om, param, spatial_domain) for om in eig_freq])
        init_adjoint_eig_funcs = np.array([ef.SecondOrderRobinEigenfunction(om, adjoint_param, spatial_domain) for om in eig_freq])

        # TODO: "vectorize" cr.normalize
        # normalize eigenfunctions and adjoint eigenfunctions
        adjoint_and_eig_funcs = [cr.normalize_function(init_eig_funcs[i], init_adjoint_eig_funcs[i]) for i in range(n)]
        eig_funcs = np.array([f_tuple[0] for f_tuple in adjoint_and_eig_funcs])
        adjoint_eig_funcs = np.array([f_tuple[1] for f_tuple in adjoint_and_eig_funcs])

        # register eigenfunctions
        register_functions("eig_funcs", eig_funcs, overwrite=True)
        register_functions("adjoint_eig_funcs", adjoint_eig_funcs, overwrite=True)

        # derive initial field variable x(z,0) and weights
        start_state = cr.Function(lambda z: 0., domain=(0, l))
        initial_weights = cr.project_on_base(start_state, adjoint_eig_funcs)

        # init trajectory
        u = tr.RadTrajectory(l, T, param, bound_cond_type, actuation_type)

        # determine (A,B) with weak-formulation (pyinduct)
        rad_pde = ut.get_parabolic_robin_weak_form("eig_funcs", "adjoint_eig_funcs", u, param, spatial_domain)
        cf = sim.parse_weak_formulation(rad_pde)
        ss_weak = cf.convert_to_state_space()

        # determine (A,B) with modal-transfomation
        A = np.diag(np.real_if_close(eig_val))
        B = a2*np.array([adjoint_eig_funcs[i](l) for i in xrange(len(eig_freq))])
        ss_modal = sim.StateSpace("eig_funcs", A, B)

        # check if ss_modal.(A,B) is close to ss_weak.(A,B)
        self.assertTrue(np.allclose(np.sort(np.linalg.eigvals(ss_weak.A)), np.sort(np.linalg.eigvals(ss_modal.A)), rtol=1e-05, atol=0.))
        self.assertTrue(np.allclose(np.array([i[0] for i in ss_weak.B]), ss_modal.B))

        # display results
        if show_plots:
            t, q = sim.simulate_state_space(ss_modal, u, initial_weights, temporal_domain, time_step=T/temporal_disc)
            eval_d = ut.evaluate_approximation(q, "eig_funcs", t, spatial_domain, l/spatial_disc)
            # eval_dd = ut.evaluate_approximation(q, np.array([i.derive(1) for i in eig_funcs]), t, spatial_domain, l/spatial_disc)
            win1 = vis.PgAnimatedPlot([eval_d], title="Test")
            win2 = vis.PgSurfacePlot(eval_d)
            app.exec_()

    # TODO Tests for SimulationInput
