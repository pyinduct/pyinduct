import sys
import unittest

import numpy as np
import pyinduct as pi
import pyinduct.shapefunctions as sh
import sympy as sp
import tests.test_data.test_shapefunctions_data as shape_data
from pyinduct.visualization import create_colormap

if any([arg in {'discover', 'setup.py', 'test'} for arg in sys.argv]):
    show_plots = False
else:
    # show_plots = True
    show_plots = False

if show_plots:
    import pyqtgraph as pg

    app = pg.QtGui.QApplication([])


class CureTestCase(unittest.TestCase):
    def test_init(self):
        self.assertRaises(TypeError, sh.cure_interval, np.sin, [2, 3], 2)
        self.assertRaises(ValueError, sh.cure_interval, sh.LagrangeFirstOrder,
                          (0, 2), 2, 1)

    def test_smoothness(self):
        func_classes = [pi.LagrangeFirstOrder, pi.LagrangeSecondOrder]
        derivatives = {pi.LagrangeFirstOrder: range(0, 2),
                       pi.LagrangeSecondOrder: range(0, 3)}
        tolerances = {pi.LagrangeFirstOrder: [5e0, 1.5e2],
                      pi.LagrangeSecondOrder: [1e0, 1e2, 9e2]}

        for func_cls in func_classes:
            for order in derivatives[func_cls]:
                self.assertGreater(tolerances[func_cls][order], self.shape_generator(func_cls, order))

    def shape_generator(self, cls, der_order):
        """
        verify the correct connection with visual feedback
        """

        dz = pi.Domain((0, 1), step=.001)
        dt = pi.Domain((0, 0), num=1)

        nodes, funcs = pi.cure_interval(cls, dz.bounds, node_count=11)
        pi.register_base("test", funcs, overwrite=True)

        # approx_func = pi.Function(np.cos, domain=dz.bounds,
        #                           derivative_handles=[lambda z: -np.sin(z), lambda z: -np.cos(z)])
        approx_func = pi.Function(lambda z: np.sin(3 * z), domain=dz.bounds,
                                  derivative_handles=[lambda z: 3 * np.cos(3 * z), lambda z: -9 * np.sin(3 * z)])

        weights = approx_func(nodes)

        hull = pi.evaluate_approximation("test", np.atleast_2d(weights),
                                         temp_domain=dt, spat_domain=dz, spat_order=der_order)

        if show_plots:
            # plot shapefunctions
            c_map = create_colormap(len(funcs))
            pw = pg.plot(title="{}-Test".format(cls.__name__))
            pw.addLegend()
            pw.showGrid(x=True, y=True, alpha=0.5)

            [pw.addItem(pg.PlotDataItem(np.array(dz),
                                        weights[idx] * func.derive(der_order)(dz),
                                        pen=pg.mkPen(color=c_map[idx]),
                                        name="{}.{}".format(cls.__name__, idx)))
             for idx, func in enumerate(funcs)]

            # plot hull curve
            pw.addItem(pg.PlotDataItem(np.array(hull.input_data[1]), hull.output_data[0, :],
                                       pen=pg.mkPen(width=2), name="hull-curve"))
            # plot original function
            pw.addItem(pg.PlotDataItem(np.array(dz), approx_func.derive(der_order)(dz),
                                       pen=pg.mkPen(color="m", width=2, style=pg.QtCore.Qt.DashLine), name="original"))
            pg.QtCore.QCoreApplication.instance().exec_()

        return np.sum(np.abs(hull.output_data[0, :] - approx_func.derive(der_order)(dz)))


class NthOrderCureTestCase(unittest.TestCase):
    def test_element(self):
        nodes = np.array([1, 2])
        self.assertRaises(ValueError, sh.LagrangeNthOrder, 0, nodes)
        self.assertRaises(ValueError, sh.LagrangeNthOrder, 1, np.array([2, 1]))
        self.assertRaises(TypeError, sh.LagrangeNthOrder, 1, nodes, left=1)
        self.assertRaises(TypeError, sh.LagrangeNthOrder, 1, nodes, right=1)
        self.assertRaises(ValueError, sh.LagrangeNthOrder, 3, nodes, mid_num=3)
        self.assertRaises(ValueError, sh.LagrangeNthOrder, 3, nodes)

    def test_smoothness(self):
        self.tolerances = shape_data.tolerances
        for conf in range(5):
            orders = range(1, 5)
            self.shape_benchmark(orders, conf)

    def shape_benchmark(self, orders, conf):
        derivatives = dict([(order, range(0, order + 1)) for order in orders])

        # approximation function
        z = sp.symbols("z")
        sin_func = [sp.sin(3 * z)]
        [sin_func.append(sin_func[-1].diff()) for i in range(orders[-1])]
        lam_sin_func = [sp.lambdify(z, func) for func in sin_func]
        approx_func = pi.Function(lam_sin_func[0], domain=(0, 1), derivative_handles=lam_sin_func[1:])

        dz = pi.Domain((0, 1), step=.001)
        dt = pi.Domain((0, 0), num=1)

        for order in orders:
            num_nodes = 1 + (1 + conf) * order
            nodes, funcs = pi.cure_interval(sh.LagrangeNthOrder, (0, 1), node_count=num_nodes, order=order)
            pi.register_base("test", funcs, overwrite=True)
            weights = approx_func(nodes)

            for der_order in derivatives[order]:
                hull_test = pi.evaluate_approximation("test", np.atleast_2d(weights),
                                                      temp_domain=dt, spat_domain=nodes, spat_order=der_order)
                self.assertAlmostEqual(self.tolerances[(order, num_nodes, der_order)], np.sum(np.abs(
                    hull_test.output_data[0, :] - approx_func.derive(der_order)(nodes))) / len(nodes))

                if show_plots:
                    hull_show = pi.evaluate_approximation("test", np.atleast_2d(weights),
                                                          temp_domain=dt, spat_domain=dz, spat_order=der_order)
                    # plot shapefunctions
                    c_map = create_colormap(len(funcs))
                    win = pg.GraphicsWindow(title="Debug window")
                    win.resize(1500, 600)
                    pw1 = win.addPlot()
                    pw1.addLegend()
                    pw1.showGrid(x=True, y=True, alpha=0.5)
                    pw2 = win.addPlot(
                        title="{} lagrange shapefunctions of order {}, derivative {}".format(num_nodes, order,
                                                                                             der_order))
                    pw2.showGrid(x=True, y=True, alpha=0.5)

                    for idx, func in enumerate(funcs):
                        pw1.addItem(pg.PlotDataItem(np.array(dz),
                                                    weights[idx] * func.derive(der_order)(dz),
                                                    pen=pg.mkPen(color=c_map[idx]),
                                                    name="{}.{}".format(order, idx)))
                        pw2.addItem(pg.PlotDataItem(np.array(dz),
                                                    func.derive(der_order)(dz),
                                                    pen=pg.mkPen(color=c_map[idx])))

                    # plot hull curve
                    pw1.addItem(pg.PlotDataItem(np.array(hull_show.input_data[1]), hull_show.output_data[0, :],
                                                pen=pg.mkPen(color="b", width=3), name="hull-curve"))
                    # plot original function
                    pw1.addItem(pg.PlotDataItem(np.array(dz), approx_func.derive(der_order)(dz),
                                                pen=pg.mkPen(color="m", width=2, style=pg.QtCore.Qt.DashLine),
                                                name="original"))
                    pg.QtCore.QCoreApplication.instance().exec_()
