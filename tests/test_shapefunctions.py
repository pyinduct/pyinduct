import sys
import unittest

import numpy as np

import pyinduct as pi
import pyinduct.shapefunctions
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
        self.assertRaises(TypeError, pyinduct.shapefunctions.cure_interval, np.sin, [2, 3])
        self.assertRaises(TypeError, pyinduct.shapefunctions.cure_interval, np.sin, (2, 3))
        self.assertRaises(ValueError, pyinduct.shapefunctions.cure_interval, pyinduct.shapefunctions.LagrangeFirstOrder,
                          (0, 2))
        self.assertRaises(ValueError, pyinduct.shapefunctions.cure_interval, pyinduct.shapefunctions.LagrangeFirstOrder,
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
