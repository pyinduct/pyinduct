from __future__ import division
import os
import unittest
import numpy as np
from cPickle import loads

from pyinduct.visualization import SlicePlot, LinePLot3d
import pyqtgraph as pg

__author__ = 'stefan'

# show_plots = True
show_plots = False


class PlotTestCase(unittest.TestCase):
    def setUp(self):
        self.app = pg.QtGui.QApplication([])

        try:
            with open(os.sep.join(["resources", "test_data.res"])) as f:
                self.test_data = loads(f.read())
        except:
            raise ValueError("run 'test_simulation' first!")

    def test_slice_plot(self):
        pt = SlicePlot(self.test_data[0])
        self.app.exec_()

    def test_3d_line_plot(self):
        pt = LinePLot3d(self.test_data)
        self.app.exec_()

    def tearDown(self):
        del self.app



