from __future__ import division
import os
import unittest
import numpy as np
from cPickle import loads

from pyinduct.visualization import SlicePlot, LinePLot3d
import pyqtgraph as pg
import sys

__author__ = 'stefan'

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
            pt = SlicePlot(self.test_data[0])
            app.exec_()

    def test_3d_line_plot(self):
        if show_plots:
            pt = LinePLot3d(self.test_data)
            app.exec_()



