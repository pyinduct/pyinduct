import pyinduct.visualization as vis
import os
import sys
import unittest
from pickle import loads

import matplotlib.pyplot as plt
import pyqtgraph as pg


# TODO: __init__ global variable show_plots
if any([arg in {'discover', 'setup.py', 'test'} for arg in sys.argv]):
    show_plots = False
else:
    show_plots = True
    # show_plots = False
app = pg.QtGui.QApplication([])


class PlotTestCase(unittest.TestCase):
    def setUp(self):
        try:
            root_dir = os.getcwd()
            if root_dir.split(os.sep)[-1] == "tests":
                res_dir = os.sep.join([os.getcwd(), "resources"])
            else:
                res_dir = os.sep.join([os.getcwd(), "tests", "resources"])

            file_path = os.sep.join([res_dir, "test_data.res"])
            with open(file_path, "r+b") as f:
                self.test_data = loads(f.read())
        except:
            raise ValueError("run 'test_simulation' first!")

        lim = 50
        self.short_data = vis.EvalData([
            self.test_data[0].input_data[0][0:lim],
            self.test_data[0].input_data[1][0:lim]],
            self.test_data[0].output_data[0:lim, 0:lim], name="short set")

    def test_slice_plot(self):
        pt = vis.PgSlicePlot(self.test_data[0])
        if show_plots:
            app.exec_()

    def test_3d_line_plot(self):
        pt = vis.PgLinePlot3d(self.test_data)
        if show_plots:
            app.exec_()

    def test_animated_plot_unequal(self):
        # test plotting of data sets with unequal length and spatial discretization
        pt = vis.PgAnimatedPlot(self.test_data + [self.short_data], title="Test Plot")
        if show_plots:
            app.exec_()

    def test_animated_plot_export(self):
        # test export
        pt = vis.PgAnimatedPlot(self.test_data + [self.short_data], title="Test Plot", save_pics=True)
        if show_plots:
            app.exec_()

        self.assertTrue(os.path.isdir(os.sep.join([os.getcwd(), pt._res_path])))

    def test_surface_plot(self):
        pt = vis.PgSurfacePlot(self.test_data[0], grid_height=10)
        if show_plots:
            app.exec_()

    def test_mpl_surface_plot(self):
        vis.MplSurfacePlot(self.test_data[1], keep_aspect=False)
        if show_plots:
            plt.show()

    def test_mpl_slice_plot(self):
        vis.MplSlicePlot(self.test_data + self.test_data + self.test_data, spatial_point=0.5, ylabel='$x(0,t)$',
                         legend_label=['1', '2', '3', '4', '5', '6'])
        if show_plots:
            plt.show()
