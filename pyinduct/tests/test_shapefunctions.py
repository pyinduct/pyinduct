import unittest

import numpy as np
import sympy as sp
import pyinduct as pi
import pyinduct.tests.test_data.test_shapefunctions_data as shape_data
from pyinduct.tests import show_plots
from pyinduct.core import integrate_function

if show_plots:
    import pyqtgraph as pg

    app = pg.QtGui.QApplication([])


class CureTestCase(unittest.TestCase):
    def test_init(self):
        self.assertRaises(TypeError, pi.cure_interval, np.sin, [2, 3], 2)
        self.assertRaises(ValueError, pi.cure_interval, pi.LagrangeFirstOrder,
                          (0, 2), 2, 1)

    def test_smoothness(self):
        func_classes = [pi.LagrangeFirstOrder, pi.LagrangeSecondOrder]
        derivatives = {pi.LagrangeFirstOrder: range(0, 2),
                       pi.LagrangeSecondOrder: range(0, 3)}
        tolerances = {pi.LagrangeFirstOrder: [5e0, 1.5e2],
                      pi.LagrangeSecondOrder: [1e0, 1e2, 9e2]}

        for func_cls in func_classes:
            for order in derivatives[func_cls]:
                self.assertGreater(tolerances[func_cls][order],
                                   self.shape_generator(func_cls, order))

    def shape_generator(self, cls, der_order):
        """
        verify the correct connection with visual feedback
        """

        dz = pi.Domain((0, 1), step=.001)
        dt = pi.Domain((0, 0), num=1)

        nodes, base = pi.cure_interval(cls, dz.bounds, node_count=11)
        pi.register_base("test", base)

        approx_func = pi.Function(lambda z: np.sin(3 * z), domain=dz.bounds,
                                  derivative_handles=[
                                      lambda z: 3 * np.cos(3 * z),
                                      lambda z: -9 * np.sin(3 * z)])

        weights = pi.project_on_base(approx_func, base)
        shape_vals = np.array([func.derive(der_order)(dz) for func in base])
        hull = pi.EvalData(dz, weights@shape_vals)

        if show_plots:
            # plot shapefunctions
            c_map = pi.create_colormap(len(base.fractions))
            pw = pg.plot(title="{}-Test".format(cls.__name__))
            pw.addLegend()
            pw.showGrid(x=True, y=True, alpha=0.5)

            [pw.addItem(pg.PlotDataItem(np.array(dz),
                                        weights[idx] * func.derive(der_order)(dz),
                                        pen=pg.mkPen(color=c_map[idx]),
                                        name="{}.{}".format(cls.__name__, idx)))
             for idx, func in enumerate(base.fractions)]

            # plot hull curve
            pw.addItem(pg.PlotDataItem(x=hull.input_data[0].points,
                                       y=hull.output_data,
                                       pen=pg.mkPen(width=2),
                                       name="hull-curve"))
            # plot original function
            pw.addItem(pg.PlotDataItem(x=dz.points,
                                       y=approx_func.derive(der_order)(dz),
                                       pen=pg.mkPen(color="m", width=2,
                                                    style=pg.QtCore.Qt.DashLine),
                                       name="original"))
            pi.show(show_mpl=False)

        pi.deregister_base("test")
        return np.sum(np.abs(hull.output_data
                             - approx_func.derive(der_order)(dz)))


class NthOrderCureTestCase(unittest.TestCase):
    def test_element(self):
        nodes = np.array([1, 2])
        self.assertRaises(ValueError, pi.LagrangeNthOrder, 0, nodes)
        self.assertRaises(ValueError, pi.LagrangeNthOrder, 1, np.array([2, 1]))
        self.assertRaises(TypeError, pi.LagrangeNthOrder, 1, nodes, left=1)
        self.assertRaises(TypeError, pi.LagrangeNthOrder, 1, nodes, right=1)
        self.assertRaises(ValueError, pi.LagrangeNthOrder, 3, nodes, mid_num=3)
        self.assertRaises(ValueError, pi.LagrangeNthOrder, 3, nodes)

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
        approx_func = pi.Function(lam_sin_func[0],
                                  domain=(0, 1),
                                  derivative_handles=lam_sin_func[1:])

        dz = pi.Domain((0, 1), step=.001)
        dt = pi.Domain((0, 0), num=1)

        for order in orders:
            num_nodes = 1 + (1 + conf) * order
            nodes, base = pi.cure_interval(pi.LagrangeNthOrder,
                                           (0, 1),
                                           node_count=num_nodes,
                                           order=order)
            pi.register_base("test", base)

            weights = pi.project_on_base(approx_func, base)

            for der_order in derivatives[order]:
                shape_vals_test = np.array([func.derive(der_order)(nodes)
                                       for func in base])
                hull_test = pi.EvalData(nodes,  weights @ shape_vals_test)

                def squared_error_function(z):
                    return (np.sum(np.array([w * f(z)
                                             for w, f in
                                             zip(weights, base.fractions)]),
                                   axis=0)
                            - approx_func.derive(der_order)(z)) ** 2

                self.assertAlmostEqual(
                    self.tolerances[(order, num_nodes, der_order)],
                    integrate_function(squared_error_function, [(0, 1)])[0])

                if show_plots:
                    shape_vals_show = np.array([func.derive(der_order)(dz)
                                           for func in base])
                    hull_show = pi.EvalData(dz,  weights @ shape_vals_show)

                    # plot shapefunctions
                    c_map = pi.create_colormap(len(base.fractions))
                    win = pg.GraphicsWindow(title="Debug window")
                    win.resize(1500, 600)
                    pw1 = win.addPlot()
                    pw1.addLegend()
                    pw1.showGrid(x=True, y=True, alpha=0.5)
                    pw2 = win.addPlot(title="{} lagrange shapefunctions of "
                                            "order {}, derivative {}".format(
                        num_nodes, order, der_order))
                    pw2.showGrid(x=True, y=True, alpha=0.5)

                    for idx, func in enumerate(base.fractions):
                        pw1.addItem(pg.PlotDataItem(np.array(dz),
                                                    weights[idx] * func.derive(der_order)(dz),
                                                    pen=pg.mkPen(color=c_map[idx]),
                                                    name="{}.{}".format(order, idx)))
                        pw2.addItem(pg.PlotDataItem(np.array(dz),
                                                    func.derive(der_order)(dz),
                                                    pen=pg.mkPen(color=c_map[idx])))

                    # plot hull curve
                    pw1.addItem(pg.PlotDataItem(x=hull_show.input_data[0].points,
                                                y=hull_show.output_data,
                                                pen=pg.mkPen(color="b", width=3),
                                                name="hull-curve"))
                    # plot original function
                    pw1.addItem(pg.PlotDataItem(x=dz.points,
                                                y=approx_func.derive(der_order)(dz),
                                                pen=pg.mkPen(color="m",
                                                             width=2,
                                                             style=pg.QtCore.Qt.DashLine),
                                                name="original"))
                    pi.show(show_mpl=False)

            pi.deregister_base("test")
