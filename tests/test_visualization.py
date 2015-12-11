from __future__ import division
import os
import unittest
import numpy as np
from cPickle import loads

import pyinduct.visualization as vis
import pyqtgraph as pg
import matplotlib.pyplot as plt
import sys

__author__ = 'stefan'

# TODO: __init__ global variable show_plots
if any([arg == 'discover' for arg in sys.argv]):
    show_plots = False
else:
    app = pg.QtGui.QApplication([])
    show_plots = True


class PlotTestCase(unittest.TestCase):
    def setUp(self):
        if show_plots:
            try:
                with open(os.sep.join(["resources", "test_data.res"])) as f:
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
            pt = vis.PgAnimatedPlot(self.test_data)
            app.exec_()

    def test_surface_plot(self):
        if show_plots:
            pt = vis.PgSurfacePlot(self.test_data[0], grid_heigth=10)
            app.exec_()

    def test_mpl_surface_plot(self):
        if show_plots:
            vis.MplSurfacePlot(self.test_data[1], keep_aspect=False)
            plt.show()

    def test_mpl_slice_plot(self):
        if show_plots:
            vis.MplSlicePlot(self.test_data+self.test_data+self.test_data, spatial_point=0.5, ylabel=u'$x(0,t)$',
                               legend_label=['1', '2', '3', '4', '5', '6'])
            plt.show()
