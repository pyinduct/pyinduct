import os
import unittest
import numpy as np

import pyinduct as pi
import pyinduct.widgets.pgwidgets as vis
from pyinduct.tests import show_plots
from pyinduct.tests.test_simulation import StringMassTest


class PlotTestCase(unittest.TestCase):
    swm = StringMassTest()

    def setUp(self):
        lim = 50
        self.test_data = self.swm.create_test_data()
        self.short_data = pi.EvalData(
            [self.test_data[0].input_data[0][0:lim],
             self.test_data[0].input_data[1][0:lim]],
            self.test_data[0].output_data[0:lim, 0:lim],
            name="short set")
        self.test_2d_data = pi.EvalData(
            [self.test_data[0].input_data[0][0:lim]],
            self.test_data[0].output_data[0:lim, 1],
            name="2d set")

    @unittest.skip
    def test_2d_plot(self):
        pt = vis.Pg2DPlot(data=self.test_2d_data)
        if show_plots:
            pi.show(show_mpl=False)

    def test_2d_animated_plot(self):
        pt = vis.Pg2DPlot(data=self.test_data, animationAxis=True)
        if show_plots:
            pi.show(show_mpl=False)

    @unittest.skip
    def test_surface_plot(self):
        pt = vis.PgSurfacePlot(data=self.test_data)
        if show_plots:
            pi.show(show_mpl=False)

    # @unittest.skip
    # def test_animated_surface_plot(self):
    #
    #     def data_func(x, y, t):
    #         d = (x**2 + y**2)/5
    #         return np.exp(-d/10) * np.sin(d - t)
    #
    #     x_arr = np.linspace(-7, 7, 100)
    #     y_arr = np.linspace(-5, 5, 200)
    #     t_arr = np.linspace(0, 10*np.pi, 300)
    #
    #     xx, yy, tt = np.meshgrid(y_arr, x_arr, t_arr)
    #     data = data_func(xx, yy, tt)
    #
    #     data_set_0 = pi.EvalData([t_arr, x_arr, y_arr],
    #                              np.rollaxis(data, 2))
    #     data_set_1 = pi.EvalData([x_arr, t_arr, y_arr],
    #                              np.swapaxes(data, 2, 1) - 1000)
    #     data_set_2 = pi.EvalData([x_arr, y_arr, t_arr],
    #                              data)
    #
    #     # animation axis has to be provided for 3d data
    #     self.assertRaises(ValueError, pi.PgSurfacePlot, data_set_0)
    #
    #     pt0 = pi.PgSurfacePlot(data_set_0, animation_axis=0)
    #     pt1 = pi.PgSurfacePlot(data_set_1, animation_axis=1)
    #     pt2 = pi.PgSurfacePlot(data_set_2, animation_axis=2)
    #
    #     if show_plots:
    #         pi.show(show_mpl=False)
