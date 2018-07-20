import os
import unittest
import numpy as np

import pyinduct as pi
import pyinduct.visualization as vis
from pyinduct.tests import show_plots
from pyinduct.tests.test_simulation import StringMassTest
from pyinduct.tests.test_core import FindRootsTestCase


class VisualizeRootsTestCase(unittest.TestCase):
    funcs = FindRootsTestCase()
    funcs.setUp()

    def test_real_function(self):
        # lets have a first look without knowing any roots
        p1 = pi.visualize_roots(None,
                                [np.linspace(np.pi/20, 3*np.pi/2, num=1000)],
                                self.funcs.frequent_eq,
                                return_window=True)

        # lets check some roots we guessed
        p2 = pi.visualize_roots(np.array(range(10)),
                                [np.linspace(np.pi/20, 3*np.pi/2, num=1000)],
                                self.funcs.frequent_eq,
                                return_window=True)
        if show_plots:
            pi.show(show_mpl=False)

    def test_complex_function(self):
        grid = [np.linspace(-2, 2), np.linspace(-2, 2)]

        # lets have a first look without knowing any roots
        p1 = pi.visualize_roots(None,
                                grid,
                                self.funcs.complex_eq,
                                cmplx=True,
                                return_window=True)

        # lets check some roots we guessed
        p2 = pi.visualize_roots(np.array(range(5))
                                + 1j * np.array(range(5, 0, -1)),
                                grid,
                                self.funcs.complex_eq,
                                cmplx=True,
                                return_window=True)
        if show_plots:
            pi.show(show_mpl=False)


class VisualizeFunctionsTestCase(unittest.TestCase):

    def setUp(self):
        self.cos_func = pi.Function(np.cos, domain=(0, 1))
        self.sin_func = pi.Function(np.sin, domain=(0, 2))
        self.complex_func = pi.Function(lambda x: (x + 2j)**2, domain=(-5, 5))
        self.vectorial_funcs = [self.cos_func, self.sin_func]

        self.tan_func = pi.Function(lambda x: np.tan(x),
                                    domain={(0, np.pi/2-1e-2),
                                            (np.pi/2+1e-2, np.pi)})

    def test_cont_dom(self):
        pi.visualize_functions(self.cos_func, return_window=True)
        pi.visualize_functions(self.sin_func, return_window=True)
        pi.visualize_functions(self.complex_func, return_window=True)
        pi.visualize_functions(self.vectorial_funcs, return_window=True)

    def test_disc_dom(self):
        pi.visualize_functions(self.tan_func, return_window=True)


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
                                title="Test Plot",
                                labels={"left": "string deflection",
                                        "bottom": "time"})
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

    def test_mpl_surface_plot_nan(self):
        nan_data = self.test_data[1].output_data.copy()
        nan_data[100:200, :] = np.nan
        tricky_data = pi.EvalData(self.test_data[1].input_data, nan_data)
        vis.MplSurfacePlot(tricky_data, keep_aspect=False)
        if show_plots:
            pi.show(show_pg=False)

    def test_mpl_slice_plot(self):
        vis.MplSlicePlot(self.test_data + self.test_data + self.test_data,
                         spatial_point=0.5,
                         ylabel='$x(0,t)$',
                         legend_label=['1', '2', '3', '4', '5', '6'])
        if show_plots:
            pi.show(show_pg=False)
