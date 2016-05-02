import sys
import unittest
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

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

    def test_visual(self):
        """
        verify by visual feedback
        """
        der_order = 1
        # func_type = pi.LagrangeFirstOrder
        func_type = pi.LagrangeSecondOrder

        dz = pi.Domain((0, 1), step=.001)
        dt = pi.Domain((0, 0), num=1)

        nodes, funcs = pi.cure_interval(func_type, dz.bounds, node_count=11)
        pi.register_base("test", funcs)
        # approx_func = pi.Function(np.cos, domain=dz.bounds, derivative_handles=[lambda z: -np.sin(z), lambda z: -np.cos(z)])
        approx_func = pi.Function(lambda z: np.sin(3*z), domain=dz.bounds,
                                  derivative_handles=[lambda z: 3*np.cos(3*z), lambda z: -9*np.sin(3*z)])
        weights = approx_func(nodes)

        # weights = np.ones((len(nodes),))
        cls = pi.visualization.create_colormap(len(funcs))

        pw = pg.plot(title="{}-Test".format(func_type.__name__))
        pw.addLegend()
        pw.showGrid(x=True, y=True, alpha=0.5)

        [pw.addItem(pg.PlotDataItem(np.array(dz),
                                    weights[idx]*func.derive(der_order)(dz),
                                    pen=pg.mkPen(color=cls[idx]),
                                    name="{}.{}".format(func_type.__name__, idx)))
         for idx, func in enumerate(funcs)]

        # plot hull curve
        hull = pi.evaluate_approximation("test", np.atleast_2d(weights),
                                         temp_domain=dt, spat_domain=dz, spat_order=der_order)
        pw.addItem(pg.PlotDataItem(np.array(hull.input_data[1]), hull.output_data[0, :],
                                   pen=pg.mkPen(width=2), name="hull-curve"))
        pw.addItem(pg.PlotDataItem(np.array(dz), approx_func.derive(der_order)(dz),
                                   pen=pg.mkPen(color="m", width=2, style=pg.QtCore.Qt.DashLine), name="original"))

        pg.QtCore.QCoreApplication.instance().exec_()


class CureTestCase(unittest.TestCase):
    def setUp(self):
        self.node_cnt = 3
        self.nodes = np.linspace(0, 2, self.node_cnt, endpoint=True)
        self.dz = (2 - 0) / (self.node_cnt - 1)  # =1 for the fast ones ...
        self.test_functions = np.array([pyinduct.shapefunctions.LagrangeFirstOrder(0, 0, 1),
                                        pyinduct.shapefunctions.LagrangeFirstOrder(0, 1, 2),
                                        pyinduct.shapefunctions.LagrangeFirstOrder(1, 2, 2)])

    def test_init(self):
        self.assertRaises(TypeError, pyinduct.shapefunctions.cure_interval, np.sin, [2, 3])
        self.assertRaises(TypeError, pyinduct.shapefunctions.cure_interval, np.sin, (2, 3))
        self.assertRaises(ValueError, pyinduct.shapefunctions.cure_interval, pyinduct.shapefunctions.LagrangeFirstOrder,
                          (0, 2))
        self.assertRaises(ValueError, pyinduct.shapefunctions.cure_interval, pyinduct.shapefunctions.LagrangeFirstOrder,
                          (0, 2), 2, 1)

    def test_rest(self):
        nodes1, funcs1 = pyinduct.shapefunctions.cure_interval(pyinduct.shapefunctions.LagrangeFirstOrder, (0, 2),
                                                               node_count=self.node_cnt)
        self.assertTrue(np.allclose(nodes1, self.nodes))
        nodes2, funcs2 = pyinduct.shapefunctions.cure_interval(pyinduct.shapefunctions.LagrangeFirstOrder, (0, 2),
                                                               node_distance=self.dz)
        self.assertTrue(np.allclose(nodes2, self.nodes))

        for i in range(self.test_functions.shape[0]):
            self.assertEqual(self.test_functions[i].nonzero, funcs1[i].nonzero)
            self.assertEqual(self.test_functions[i].nonzero, funcs2[i].nonzero)

    def test_lagrange_2nd_order(self):
        nodes, funcs = pyinduct.shapefunctions.cure_interval(pyinduct.shapefunctions.LagrangeSecondOrder, (0, 1),
                                                             node_count=2)
        self.assertTrue(np.allclose(np.diag(np.ones(len(funcs))),
                                    np.array([funcs[i](nodes) for i in range(len(funcs))])))
        if show_plots:
            fig = plt.figure(figsize=(14, 6), facecolor='white')
            mpl.rcParams.update({'font.size': 50})
            plt.xticks(nodes)
            plt.yticks([0, 1])
            z = np.linspace(0, 1, 1000)
            [plt.plot(z, fun.derive(0)(z)) for fun in funcs]
            plt.grid(True)
            plt.show()
