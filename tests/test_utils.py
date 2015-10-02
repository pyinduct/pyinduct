from __future__ import division
import unittest
import numpy as np
from pyinduct import core, utils, visualization
import pyqtgraph as pg

__author__ = 'Stefan Ecklebe'


show_plots = True
# show_plots = False


class CureTestCase(unittest.TestCase):
    def setUp(self):
        self.node_cnt = 3
        self.nodes = np.linspace(0, 2, self.node_cnt)
        self.dz = (2 - 0) / (self.node_cnt-1)
        self.test_functions = np.array([core.LagrangeFirstOrder(0, 0, 1),
                                        core.LagrangeFirstOrder(0, 1, 2),
                                        core.LagrangeFirstOrder(1, 2, 2)])

    def test_init(self):
        self.assertRaises(TypeError, utils.cure_interval, np.sin, [2, 3])
        self.assertRaises(TypeError, utils.cure_interval, np.sin, (2, 3))
        self.assertRaises(ValueError, utils.cure_interval, core.LagrangeFirstOrder, (0, 2))
        self.assertRaises(ValueError, utils.cure_interval, core.LagrangeFirstOrder, (0, 2), 2, 1)

    def test_rest(self):
        nodes1, funcs1 = utils.cure_interval(core.LagrangeFirstOrder, (0, 2), node_count=self.node_cnt)
        self.assertTrue(np.allclose(nodes1, self.nodes))
        nodes2, funcs2 = utils.cure_interval(core.LagrangeFirstOrder, (0, 2), element_length=self.dz)
        self.assertTrue(np.allclose(nodes2, self.nodes))

        for i in range(self.test_functions.shape[0]):
            self.assertEqual(self.test_functions[i].nonzero, funcs1[i].nonzero)
            self.assertEqual(self.test_functions[i].nonzero, funcs2[i].nonzero)


class FindRootsTestCase(unittest.TestCase):

    def setUp(self):
        self.app = pg.QtGui.QApplication([])

        def _char_equation(omega):
            return omega * (np.sin(omega) + omega * np.cos(omega))

        self.char_eq = _char_equation

    def test_feasible(self):
        roots = utils.find_roots(self.char_eq, 1e1, (0, 1e3))
        self.assertEqual(len(roots), 1e1)
        for root in roots:
            self.assertAlmostEqual(self.char_eq(root), 0)

        if show_plots:
            points = np.arange(0, 100, .1)
            values = self.char_eq(points)
            pw = pg.plot(title="char equation roots")
            pw.plot(points, values)
            pw.plot(roots, self.char_eq(roots), pen=None, symbolPen=pg.mkPen("g"))
            self.app.exec_()

    def tearDown(self):
        del self.app


class EvaluateApproximationTestCase(unittest.TestCase):

    def setUp(self):
        self.app = pg.QtGui.QApplication([])
        self.node_cnt = 5
        self.time_step = 1e-3
        self.dates = np.arange(0, 10, self.time_step)
        self.spat_int = (0, 10)
        self.nodes = np.linspace(self.spat_int[0], self.spat_int[1], self.node_cnt)

        # create initial functions
        self.nodes, self.funcs = utils.cure_interval(core.LagrangeFirstOrder, self.spat_int, node_count=self.node_cnt)

        # create a slow rising, nearly horizontal line
        self.weights = np.array(range(self.node_cnt*self.dates.size)).reshape((self.dates.size, self.nodes.size))

    def test_eval_helper(self):
        eval_data = utils.evaluate_approximation(self.weights, self.funcs, self.dates, self.spat_int, .1)
        if show_plots:
            p = visualization.AnimatedPlot(eval_data)
            self.app.exec_()

    def tearDown(self):
        del self.app


