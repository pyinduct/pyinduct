import sys
import unittest

import core
import numpy as np
import os

from pyinduct import register_base, deregister_base, \
    core as cr, \
    simulation as sim, \
    shapefunctions as sh, \
    utils as ut, \
    visualization as vt, \
    placeholder as ph

if any([arg in {'discover', 'setup.py', 'test'} for arg in sys.argv]):
    show_plots = False
else:
    # show_plots = True
    show_plots = False

if show_plots:
    import pyqtgraph as pg

    app = pg.QtGui.QApplication([])


class ParamsTestCase(unittest.TestCase):

    def test_init(self):
        p = ut.Parameters(a=10, b=12, c="high")
        self.assertTrue(p.a == 10)
        self.assertTrue(p.b == 12)
        self.assertTrue(p.c == "high")


class FindRootsTestCase(unittest.TestCase):
    def setUp(self):
        def _char_equation(omega):
            return omega * (np.sin(omega) + omega * np.cos(omega))

        def _univar_equation(x):
            return [np.cos(x[0]), np.cos(4 * x[1])]

        def _cmplx_equation(lamda):
            if lamda == 0:
                return 0
            return lamda ** 2 + 9

        self.char_eq = _char_equation
        self.univar_eq = _univar_equation
        self.cmplx_eq = _cmplx_equation

        self.n_roots = 10
        self.small_grid = np.arange(0, 1, 1)
        self.grid = np.arange(0, 50, 1)
        self.rtol = -1

    def test_in_fact_roots(self):
        roots = core.find_roots(self.char_eq, self.n_roots, self.grid, self.rtol)
        for root in roots:
            self.assertAlmostEqual(self.char_eq(root), 0)

    def test_enough_roots(self):
        # small area -> not enough roots -> Exception
        self.assertRaises(ValueError, core.find_roots, self.char_eq, self.n_roots, self.small_grid, self.rtol)

        roots = core.find_roots(self.char_eq, self.n_roots, self.grid, self.rtol)
        self.assertEqual(len(roots), self.n_roots)

    def test_rtol(self):
        roots = core.find_roots(self.char_eq, self.n_roots, self.grid, self.rtol, show_plot=show_plots)
        self.assertGreaterEqual(np.log10(min(np.abs(np.diff(roots)))), self.rtol)

    def test_in_area(self):
        roots = core.find_roots(self.char_eq, self.n_roots, self.grid, self.rtol)
        for root in roots:
            self.assertTrue(root >= 0.)

    @unittest.skip  # doesn't match the new signature
    def test_error_raiser(self):
        float_num = -1.
        int_num = 0
        to_small_area_end = 1e-3

        self.assertRaises(TypeError, core.find_roots, int_num, self.n_roots, self.grid, self.rtol)
        self.assertRaises(TypeError, core.find_roots, self.char_eq, float_num, self.grid, self.rtol)
        self.assertRaises(TypeError, core.find_roots, self.char_eq, self.n_roots, self.grid, self.rtol,
                          points_per_root=float_num)
        self.assertRaises(TypeError, core.find_roots, self.char_eq, self.n_roots, self.grid, self.rtol,
                          show_plots=int_num)
        self.assertRaises(TypeError, core.find_roots, self.char_eq, self.n_roots, self.grid, float_num)

        self.assertRaises(ValueError, core.find_roots, self.char_eq, self.n_roots, int_num, self.rtol)
        self.assertRaises(ValueError, core.find_roots, self.char_eq, self.n_roots, self.grid, self.rtol, atol=int_num)
        self.assertRaises(ValueError, core.find_roots, self.char_eq, int_num, self.grid, self.rtol)
        self.assertRaises(ValueError, core.find_roots, self.char_eq, self.n_roots, self.grid, self.rtol,
                          points_per_root=int_num)
        self.assertRaises(ValueError, core.find_roots, self.char_eq, self.n_roots, float_num, self.rtol)
        self.assertRaises(ValueError, core.find_roots, self.char_eq, self.n_roots, self.grid, self.rtol,
                          atol=float_num)
        self.assertRaises(ValueError, core.find_roots, self.char_eq, self.n_roots, to_small_area_end, self.rtol)

    def test_debug_plot(self):
        if show_plots:
            self.roots = core.find_roots(self.char_eq, self.n_roots, self.grid, rtol=self.rtol,
                                         show_plot=show_plots)

    def test_cmplx_func(self):
        grid = [np.arange(-10, 10), np.arange(-5, 5)]
        roots = core.find_roots(self.cmplx_eq, 3, grid, -1, show_plot=show_plots, complex=True)
        self.assertTrue(np.allclose([self.cmplx_eq(root) for root in roots], [0] * len(roots)))
        print(roots)

    def test_n_dim_func(self):
        grid = np.array([list(range(10)), list(range(10))])
        roots = core.find_roots(self.univar_eq, self.n_roots, grid, self.rtol,
                                show_plot=show_plots)
        print(roots)

    def tearDown(self):
        pass


class EvaluatePlaceholderFunctionTestCase(unittest.TestCase):
    def setUp(self):
        self.f = np.cos
        self.psi = cr.Function(np.sin)
        register_base("funcs", self.psi, overwrite=True)
        self.funcs = ph.TestFunction("funcs")

    def test_eval(self):
        eval_values = np.array(list(range(10)))

        # supply a non-placeholder
        self.assertRaises(TypeError, ut.evaluate_placeholder_function, self.f, eval_values)

        # check for correct results
        res = ut.evaluate_placeholder_function(self.funcs, eval_values)
        self.assertTrue(np.allclose(self.psi(eval_values), res))

    def tearDown(self):
        deregister_base("funcs")


class EvaluateApproximationTestCase(unittest.TestCase):
    def setUp(self):
        self.node_cnt = 5
        self.time_step = 1e-1
        self.dates = np.arange(0, 10, self.time_step)
        self.spat_int = (0, 1)
        self.nodes = np.linspace(self.spat_int[0], self.spat_int[1], self.node_cnt)

        # create initial functions
        self.nodes, self.funcs = sh.cure_interval(sh.LagrangeFirstOrder, self.spat_int, node_count=self.node_cnt)
        register_base("approx_funcs", self.funcs, overwrite=True)

        # create a slow rising, nearly horizontal line
        self.weights = np.array(list(range(self.node_cnt * self.dates.size))).reshape(
            (self.dates.size, len(self.nodes)))

    def test_eval_helper(self):
        eval_data = sim.evaluate_approximation("approx_funcs", self.weights, self.dates, self.spat_int, 1)
        if show_plots:
            p = vt.PgAnimatedPlot(eval_data)
            app.exec_()
            del p

    def tearDown(self):
        pass


class CreateDirTestCase(unittest.TestCase):
    existing_file = "already_a_file_there"
    existing_dir = "already_there"
    new_dir = "not_yet_there"

    def setUp(self):
        # check if test directories already exist and stop if they do
        for name in [self.existing_dir, self.new_dir]:
            dir_name = os.sep.join([os.getcwd(), name])
            if os.path.exists(dir_name):
                self.fail("test directory already exists, tests cannot be run.")

    def test_existing_file(self):
        # create a file with directory name
        dir_name = os.sep.join([os.getcwd(), self.existing_dir])
        with open(dir_name, "w") as f:
            pass

        self.assertRaises(FileExistsError, ut.create_dir, self.existing_dir)
        os.remove(dir_name)

    def test_existing_dir(self):
        dir_name = os.sep.join([os.getcwd(), self.existing_dir])
        os.makedirs(dir_name)
        ret = ut.create_dir(self.existing_dir)
        self.assertTrue(os.path.exists(dir_name))  # do not remove the directory
        self.assertEqual(ret, dir_name)  # return abs path of created dir
        os.rmdir(dir_name)

    def test_non_existing_dir(self):
        dir_name = os.sep.join([os.getcwd(), self.new_dir])
        ret = ut.create_dir(self.new_dir)
        self.assertTrue(os.path.exists(dir_name))  # directory should be created
        self.assertEqual(ret, dir_name)  # return abs path of created dir

        os.rmdir(dir_name)


class CreateVideoTestCase(unittest.TestCase):

    @unittest.skip("unfinished test case that requires ffmpeg")
    def test_creation(self):
        # TODO generate test data first!
        ut.create_animation("./animation_output/Test_Plot_21_55_32_%03d_.png")
