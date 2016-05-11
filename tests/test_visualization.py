import os
import sys
import unittest
from pickle import loads

import matplotlib.pyplot as plt

import pyinduct.visualization as vis

# TODO: __init__ global variable show_plots
if any([arg in {'discover', 'setup.py', 'test'} for arg in sys.argv]):
    show_plots = False
else:
    # show_plots = True
    show_plots = False

if show_plots:
    import pyqtgraph as pg
    app = pg.QtGui.QApplication([])


class PlotTestCase(unittest.TestCase):
    def setUp(self):
        if show_plots:
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

    def test_slice_plot(self):
        if show_plots:
            pt = vis.PgSlicePlot(self.test_data[0])
            app.exec_()

    def test_3d_line_plot(self):
        if show_plots:
            pt = vis.PgLinePlot3d(self.test_data)
            app.exec_()

    def test_animated_plot(self):
        if show_plots:
            lim = 50
            short_data = vis.EvalData([
                self.test_data[0].input_data[0][0:lim],
                self.test_data[0].input_data[1][0:lim]],
                self.test_data[0].output_data[0:lim, 0:lim], name="short set")
            pt = vis.PgAnimatedPlot(self.test_data + [short_data], title="Test Plot")
            app.exec_()

    def test_surface_plot(self):
        if show_plots:
            pt = vis.PgSurfacePlot(self.test_data[0], grid_height=10)
            app.exec_()

    def test_mpl_surface_plot(self):
        if show_plots:
            vis.MplSurfacePlot(self.test_data[1], keep_aspect=False)
            plt.show()

    def test_mpl_slice_plot(self):
        if show_plots:
            vis.MplSlicePlot(self.test_data+self.test_data+self.test_data, spatial_point=0.5, ylabel='$x(0,t)$',
                             legend_label=['1', '2', '3', '4', '5', '6'])
            plt.show()
