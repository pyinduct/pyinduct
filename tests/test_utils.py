from __future__ import division
import unittest
import sys

import numpy as np
import pyqtgraph as pg

from pyinduct import register_functions, \
    core as cr, \
    shapefunctions as sh, \
    utils as ut, \
    visualization as vt, \
    placeholder as ph


if any([arg == 'discover' for arg in sys.argv]):
    show_plots = False
else:
    show_plots = True
    app = pg.QtGui.QApplication([])


class FindRootsTestCase(unittest.TestCase):

    def setUp(self):
        def _char_equation(omega):
            return omega * (np.sin(omega) + omega * np.cos(omega))

        def _univar_equation(x):
            return [np.cos(x[0]), np.cos(4*x[1])]

        def _cmplx_equation(lamda):
            if lamda == 0:
                return 0
            return lamda**2 + 9

        self.char_eq = _char_equation
        self.univar_eq = _univar_equation
        self.cmplx_eq = _cmplx_equation

        self.n_roots = 10
        self.grid = np.arange(0, 50, 1)
        self.rtol = -1

    def test_in_fact_roots(self):
        roots = ut.find_roots(self.char_eq, self.n_roots, self.grid, self.rtol)
        for root in roots:
            self.assertAlmostEqual(self.char_eq(root), 0)

    def test_enough_roots(self):
        roots = ut.find_roots(self.char_eq, self.n_roots, self.grid, self.rtol)
        self.assertEqual(len(roots), self.n_roots)

    def test_rtol(self):
        roots = ut.find_roots(self.char_eq, self.n_roots, self.grid, self.rtol, show_plot=True)
        self.assertGreaterEqual(np.log10(min(np.abs(np.diff(roots)))), self.rtol)

    def test_in_area(self):
        roots = ut.find_roots(self.char_eq, self.n_roots, self.grid, self.rtol)
        for root in roots:
            self.assertTrue(root >= 0.)

    def test_error_raiser(self):
        return
        float_num = -1.
        int_num = 0
        to_small_area_end = 1e-3

        self.assertRaises(TypeError, ut.find_roots, int_num, self.n_roots, self.grid, self.rtol)
        self.assertRaises(TypeError, ut.find_roots, self.char_eq, float_num, self.grid, self.rtol)
        self.assertRaises(TypeError, ut.find_roots, self.char_eq, self.n_roots, self.grid, self.rtol,
                          points_per_root=float_num)
        self.assertRaises(TypeError, ut.find_roots, self.char_eq, self.n_roots, self.grid, self.rtol,
                          show_plots=int_num)
        self.assertRaises(TypeError, ut.find_roots, self.char_eq, self.n_roots, self.grid, float_num)

        self.assertRaises(ValueError, ut.find_roots, self.char_eq, self.n_roots, int_num, self.rtol)
        self.assertRaises(ValueError, ut.find_roots, self.char_eq, self.n_roots, self.grid, self.rtol, atol=int_num)
        self.assertRaises(ValueError, ut.find_roots, self.char_eq, int_num, self.grid, self.rtol)
        self.assertRaises(ValueError, ut.find_roots, self.char_eq, self.n_roots, self.grid, self.rtol,
                          points_per_root=int_num)
        self.assertRaises(ValueError, ut.find_roots, self.char_eq, self.n_roots, float_num, self.rtol)
        self.assertRaises(ValueError, ut.find_roots, self.char_eq, self.n_roots, self.grid, self.rtol,
                          atol=float_num)
        self.assertRaises(ValueError, ut.find_roots, self.char_eq, self.n_roots, to_small_area_end, self.rtol)

    def test_debug_plot(self):
        if show_plots:
            self.roots = ut.find_roots(self.char_eq, self.n_roots, self.grid, rtol=self.rtol,
                                       show_plot=True)

    def test_cmplx_func(self):
        grid = [np.arange(-10, 10), np.arange(-5, 5)]
        roots = ut.find_roots(self.cmplx_eq, 3, grid, -1, show_plot=True, complex=True)
        self.assertTrue(np.allclose([self.cmplx_eq(root) for root in roots], [0]*len(roots)))
        print(roots)

    def test_n_dim_func(self):
        grid = np.array([range(10), range(10)])
        roots = ut.find_roots(self.univar_eq, self.n_roots, grid, self.rtol,
                              show_plot=True)
        print(roots)

    def tearDown(self):
        pass


class EvaluatePlaceholderFunctionTestCase(unittest.TestCase):

    def setUp(self):
        self.psi = cr.Function(np.sin)
        register_functions("funcs", self.psi, overwrite=True)
        self.funcs = ph.TestFunction("funcs")

    def test_eval(self):
        eval_values = np.array(range(10))
        res = ut.evaluate_placeholder_function(self.funcs, eval_values)
        self.assertTrue(np.allclose(self.psi(eval_values), res))


class EvaluateApproximationTestCase(unittest.TestCase):

    def setUp(self):

        self.node_cnt = 5
        self.time_step = 1e-1
        self.dates = np.arange(0, 10, self.time_step)
        self.spat_int = (0, 1)
        self.nodes = np.linspace(self.spat_int[0], self.spat_int[1], self.node_cnt)

        # create initial functions
        self.nodes, self.funcs = sh.cure_interval(sh.LagrangeFirstOrder, self.spat_int, node_count=self.node_cnt)
        register_functions("approx_funcs", self.funcs, overwrite=True)

        # create a slow rising, nearly horizontal line
        self.weights = np.array(range(self.node_cnt*self.dates.size)).reshape((self.dates.size, self.nodes.size))

    def test_eval_helper(self):
        eval_data = ut.evaluate_approximation("approx_funcs", self.weights, self.dates, self.spat_int, .1)
        if show_plots:
            p = vt.PgAnimatedPlot(eval_data)
            app.exec_()
            del p

    def tearDown(self):
        pass
