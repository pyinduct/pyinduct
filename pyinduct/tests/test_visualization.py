import os
import unittest
import copy
import numpy as np

import pyinduct as pi
import pyinduct.visualization as vis
from pyinduct.tests import show_plots
from pyinduct.tests.test_simulation import StringMassTest


if show_plots:
    import pyqtgraph as pg

    app = pg.mkQApp()
else:
    app = None


def create_test_data():
    """
    create a test data set
    """
    swm = StringMassTest()
    swm.setUp()
    swm.test_fem()
    test_data = copy.copy(swm.example_data)
    swm.tearDown()
    return test_data


class PlotTestCase(unittest.TestCase):

    test_data = create_test_data()

    def setUp(self):
        lim = 50
        self.short_data = pi.EvalData(
            [self.test_data[0].input_data[0][0:lim],
             self.test_data[0].input_data[1][0:lim]],
            self.test_data[0].output_data[0:lim, 0:lim],
            name="short set")

    def test_slice_plot(self):
        pt = vis.PgSlicePlot(self.test_data[0])
        if show_plots:
            pi.show(show_mpl=False)

    def test_3d_line_plot(self):
        pt = vis.PgLinePlot3d(self.test_data)
        if show_plots:
            pi.show(show_mpl=False)

    def test_animated_plot_unequal(self):
        # test plotting of data sets with unequal length and spatial
        # discretization
        pt = vis.PgAnimatedPlot(self.test_data + [self.short_data],
                                title="Test Plot")
        if show_plots:
            pi.show(show_mpl=False)

    @unittest.skip("PyQtgraph raises an error here")
    def test_animated_plot_export(self):
        # test export
        pt = vis.PgAnimatedPlot(self.test_data + [self.short_data],
                                title="Test Plot",
                                save_pics=True)
        if show_plots:
            pi.show(show_mpl=False)

        self.assertTrue(os.path.isdir(os.sep.join([os.getcwd(), pt._res_path])))

    def test_surface_plot(self):
        pt = vis.PgSurfacePlot(self.test_data,
                               scales=(.1, 1, .1)
                               )
        if show_plots:
            pi.show(show_mpl=False)

    def test_animated_surface_plot(self):

        def data_func(x, y, t):
            d = (x**2 + y**2)/5
            return np.exp(-d/10) * np.sin(d - t)

        x_arr = np.linspace(-7, 7, 100)
        y_arr = np.linspace(-5, 5, 200)
        t_arr = np.linspace(0, 10*np.pi, 300)

        xx, yy, tt = np.meshgrid(y_arr, x_arr, t_arr)
        data = data_func(xx, yy, tt)

        data_set_0 = pi.EvalData([t_arr, x_arr, y_arr],
                                 np.rollaxis(data, 2))
        data_set_1 = pi.EvalData([x_arr, t_arr, y_arr],
                                 np.swapaxes(data, 2, 1) - 1000)
        data_set_2 = pi.EvalData([x_arr, y_arr, t_arr],
                                 data)

        # animation axis has to be provided for 3d data
        self.assertRaises(ValueError, pi.PgSurfacePlot, data_set_0)

        pt0 = pi.PgSurfacePlot(data_set_0, animation_axis=0)
        pt1 = pi.PgSurfacePlot(data_set_1, animation_axis=1)
        pt2 = pi.PgSurfacePlot(data_set_2, animation_axis=2)

        if show_plots:
            pi.show(show_mpl=False)

    def test_mpl_surface_plot(self):
        vis.MplSurfacePlot(self.test_data[1], keep_aspect=False)
        if show_plots:
            pi.show(show_pg=False)

    def test_mpl_slice_plot(self):
        vis.MplSlicePlot(self.test_data + self.test_data + self.test_data,
                         spatial_point=0.5,
                         ylabel='$x(0,t)$',
                         legend_label=['1', '2', '3', '4', '5', '6'])
        if show_plots:
            pi.show(show_pg=False)
