import os
import unittest
import copy

import matplotlib.pyplot as plt
import pyinduct as pi
import pyinduct.visualization as vis
from tests import show_plots
from tests.test_simulation import StringMassTest


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
            app.exec_()

    def test_3d_line_plot(self):
        pt = vis.PgLinePlot3d(self.test_data)
        if show_plots:
            app.exec_()

    def test_animated_plot_unequal(self):
        # test plotting of data sets with unequal length and spatial
        # discretization
        pt = vis.PgAnimatedPlot(self.test_data + [self.short_data],
                                title="Test Plot")
        if show_plots:
            app.exec_()

    @unittest.skip("PyQtgraph raises an error here")
    def test_animated_plot_export(self):
        # test export
        pt = vis.PgAnimatedPlot(self.test_data + [self.short_data],
                                title="Test Plot",
                                save_pics=True)
        if show_plots:
            app.exec_()

        self.assertTrue(os.path.isdir(os.sep.join([os.getcwd(), pt._res_path])))

    def test_surface_plot(self):
        pt = vis.PgSurfacePlot(self.test_data[0])
        if show_plots:
            app.exec_()

    def test_mpl_surface_plot(self):
        vis.MplSurfacePlot(self.test_data[1], keep_aspect=False)
        if show_plots:
            plt.show()

    def test_mpl_slice_plot(self):
        vis.MplSlicePlot(self.test_data + self.test_data + self.test_data,
                         spatial_point=0.5,
                         ylabel='$x(0,t)$',
                         legend_label=['1', '2', '3', '4', '5', '6'])
        if show_plots:
            plt.show()
