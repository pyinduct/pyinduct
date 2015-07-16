from __future__ import division

__author__ = 'stefan'

import unittest
import numpy as np

from pyinduct import core, utils, trajectory as tr, visualization as vis
import pyqtgraph as pg


class SmoothTransitionTestCase(unittest.TestCase):
    def setUp(self):
        self.z_start = 0
        self.z_end = 1
        self.t_start = 0
        self.t_end = 5
        self.z_step = 0.01
        self.t_step = 0.01
        self.t_values = np.arange(self.t_start, self.t_end + self.t_step, self.t_step)
        self.z_values = np.arange(self.z_start, self.z_end + self.z_step, self.z_step)
        self.y0 = 0
        self.y1 = 10

        self.app = pg.QtGui.QApplication([])

    def test_trajectory(self):
        # build flatness based trajectory generator
        fs = tr.FlatString(y0=self.y0, y1=self.y1, z0=self.z_start, z1=self.z_end, t0=0, dt=2, m=1, v=1)
        zz, tt = np.meshgrid(self.z_values, self.t_values)
        x_values = fs.system_state(zz, tt)
        u_values = fs.control_input(self.t_values)
        eval_data_x = vis.EvalData([self.t_values, self.z_values], x_values)

        # plot stuff
        pw = pg.plot(title="control_input")
        pw.plot(self.t_values, u_values)
        ap = vis.AnimatedPlot(eval_data_x)

        self.app.exec_()

    def tearDown(self):
        del self.app
