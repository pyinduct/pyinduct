from __future__ import division
import unittest
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import sys

import pyinduct.shapefunctions

__author__ = 'Stefan Ecklebe'

# show_plots = True
show_plots = False
app = None

if not any([arg == 'discover' for arg in sys.argv]):
    import pyqtgraph as pg
    app = pg.QtGui.QApplication([])
    show_plots = False


class LagrangeFirstOrderTestCase(unittest.TestCase):
    def test_init(self):
        self.assertRaises(ValueError, pyinduct.shapefunctions.LagrangeFirstOrder, 1, 0, 0)
        self.assertRaises(ValueError, pyinduct.shapefunctions.LagrangeFirstOrder, 0, 1, 0)

    def test_edge_cases(self):
        p1 = pyinduct.shapefunctions.LagrangeFirstOrder(0, 0, 1)
        self.assertEqual(p1.domain, [(-np.inf, np.inf)])
        self.assertEqual(p1.nonzero, [(0, 1)])
        self.assertEqual(p1(-.5), 0)
        self.assertEqual(p1(0), 1)
        self.assertEqual(p1(0.5), 0.5)
        self.assertEqual(p1(1), 0)
        self.assertEqual(p1(1.5), 0)

        p2 = pyinduct.shapefunctions.LagrangeFirstOrder(0, 1, 1)
        self.assertEqual(p2.domain, [(-np.inf, np.inf)])
        self.assertEqual(p2.nonzero, [(0, 1)])
        self.assertEqual(p2(-.5), 0)
        self.assertEqual(p2(0), 0)
        self.assertEqual(p2(0.5), 0.5)
        self.assertEqual(p2(1), 1)
        self.assertEqual(p2(1.5), 0)

    def test_interior_case(self):
        p2 = pyinduct.shapefunctions.LagrangeFirstOrder(0, 1, 2)
        self.assertEqual(p2.domain, [(-np.inf, np.inf)])
        self.assertEqual(p2.nonzero, [(0, 2)])
        self.assertEqual(p2(0), 0)
        self.assertEqual(p2(0.5), 0.5)
        self.assertEqual(p2(1), 1)
        self.assertEqual(p2(1.5), .5)
        self.assertEqual(p2(2), 0)

        # verify equality to zero anywhere outside of nonzero
        self.assertEqual(p2(-1e3), 0)
        self.assertEqual(p2(1e3), 0)

        # integral over whole nonzero area of self**2
        # self.assertEqual(p1.quad_int(), 2/3)


class CureTestCase(unittest.TestCase):

    def setUp(self):
        self.node_cnt = 3
        self.nodes = np.linspace(0, 2, self.node_cnt)
        self.dz = (2 - 0) / (self.node_cnt-1)
        self.test_functions = np.array([pyinduct.shapefunctions.LagrangeFirstOrder(0, 0, 1),
                                        pyinduct.shapefunctions.LagrangeFirstOrder(0, 1, 2),
                                        pyinduct.shapefunctions.LagrangeFirstOrder(1, 2, 2)])

    def test_init(self):
        self.assertRaises(TypeError, pyinduct.shapefunctions.cure_interval, np.sin, [2, 3])
        self.assertRaises(TypeError, pyinduct.shapefunctions.cure_interval, np.sin, (2, 3))
        self.assertRaises(ValueError, pyinduct.shapefunctions.cure_interval, pyinduct.shapefunctions.LagrangeFirstOrder, (0, 2))
        self.assertRaises(ValueError, pyinduct.shapefunctions.cure_interval, pyinduct.shapefunctions.LagrangeFirstOrder, (0, 2), 2, 1)

    def test_rest(self):
        nodes1, funcs1 = pyinduct.shapefunctions.cure_interval(pyinduct.shapefunctions.LagrangeFirstOrder, (0, 2), node_count=self.node_cnt)
        self.assertTrue(np.allclose(nodes1, self.nodes))
        nodes2, funcs2 = pyinduct.shapefunctions.cure_interval(pyinduct.shapefunctions.LagrangeFirstOrder, (0, 2), element_length=self.dz)
        self.assertTrue(np.allclose(nodes2, self.nodes))

        for i in range(self.test_functions.shape[0]):
            self.assertEqual(self.test_functions[i].nonzero, funcs1[i].nonzero)
            self.assertEqual(self.test_functions[i].nonzero, funcs2[i].nonzero)

    def test_lagrange_2nd_order(self):
        nodes, funcs = pyinduct.shapefunctions.cure_interval(pyinduct.shapefunctions.LagrangeSecondOrder, (0, 1), node_count=2)
        self.assertTrue(np.allclose(np.diag(np.ones(len(funcs))),
                                    np.array([funcs[i](nodes) for i in range(len(funcs))])))
        if show_plots:
            fig = plt.figure(figsize=(14, 6), facecolor='white')
            mpl.rcParams.update({'font.size': 50})
            plt.xticks(nodes)
            plt.yticks([0, 1])
            z = np.linspace(0,1,1000)
            [plt.plot(z, fun.derive(0)(z)) for fun in funcs]; plt.grid(True); plt.show()
