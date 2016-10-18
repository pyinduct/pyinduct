import os
import sys
import unittest

import numpy as np
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


class EvaluatePlaceholderFunctionTestCase(unittest.TestCase):
    def setUp(self):
        self.f = np.cos
        self.psi = cr.Function(np.sin)
        register_base("funcs", cr.Base(self.psi), overwrite=True)
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
