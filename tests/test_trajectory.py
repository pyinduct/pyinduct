from __future__ import division

__author__ = 'stefan'

import unittest
import numpy as np

from pyinduct import core, utils, trajectory as tr, visualization as vis
import pyinduct.eigenfunctions as ef
import pyinduct.utils as ut
import pyqtgraph as pg
import sys

if any([arg == 'discover' for arg in sys.argv]):
    show_plots = False
else:
    show_plots = True
    app = pg.QtGui.QApplication([])

class SmoothTransitionTestCase(unittest.TestCase):
    def setUp(self):
        self.z_start = 0
        self.z_end = 1
        self.t_start = 0
        self.t_end = 10
        self.z_step = 0.01
        self.t_step = 0.01
        self.t_values = np.arange(self.t_start, self.t_end + self.t_step, self.t_step)
        self.z_values = np.arange(self.z_start, self.z_end + self.z_step, self.z_step)
        self.y0 = -5
        self.y1 = 10

    def test_trajectory(self):
        # build flatness based trajectory generator
        fs = tr.FlatString(y0=self.y0, y1=self.y1, z0=self.z_start, z1=self.z_end, t0=self.t_start, dt=2, sigma=1,
                           v=.5)
        zz, tt = np.meshgrid(self.z_values, self.t_values)
        x_values = fs.system_state(zz, tt)
        u_values = fs.control_input(self.t_values)
        eval_data_x = vis.EvalData([self.t_values, self.z_values], x_values)

        if show_plots:
            # plot stuff
            pw = pg.plot(title="control_input")
            pw.plot(self.t_values, u_values)
            ap = vis.PgAnimatedPlot(eval_data_x)
            app.exec_()

class FormalPowerSeriesTest(unittest.TestCase):

    def setUp(self):

        self.l=1; self.T=1
        a2 = 1; a1 = 0; a0 = 6; self.alpha = 0.5; self.beta = 0.5
        self.param = [a2, a1, a0, self.alpha, self.beta]
        self.n_y = 80
        self.y, self.t = tr.gevrey_tanh(self.T, self.n_y, 1.1, 1)

    def test_temporal_derive(self):

        b_desired = 0.4
        k = 5 # = k1 + k2
        k1, k2, b = ut.split_domain(k, b_desired, self.l, mode='coprime')[0:3]
        # q
        E = tr.coefficient_recursion(self.y, self.beta*self.y, self.param)
        q = tr.temporal_derived_power_series(self.l-b, E, int(self.n_y/2)-1, self.n_y)
        # u
        B = tr.coefficient_recursion(self.y, self.alpha*self.y, self.param)
        xq = tr.temporal_derived_power_series(self.l, B, int(self.n_y/2)-1, self.n_y, spatial_der_order=0)
        d_xq = tr.temporal_derived_power_series(self.l, B, int(self.n_y/2)-1, self.n_y, spatial_der_order=1)
        u = d_xq + self.beta*xq
        # x(0,t)
        C = tr.coefficient_recursion(q, self.beta*q, self.param)
        D = tr.coefficient_recursion(np.zeros(u.shape), u, self.param)
        x_0t = tr.power_series(0, self.t, C)
        if show_plots:
            pw = pg.plot(title="control_input")
            pw.plot(self.t, x_0t)
            app.exec_()

    def test_recursion_vs_explicit(self):

        # recursion
        B = tr.coefficient_recursion(self.y, self.alpha*self.y, self.param)
        x_l = tr.power_series(self.l, self.t, B)
        d_x_l = tr.power_series(self.l, self.t, B, spatial_der_order=1)
        u_c = d_x_l + self.beta*x_l
        u_a = tr.InterpTrajectory(self.t, u_c, show_plot=show_plots)
        u_a_t = u_a(time=self.t)
        # explicit
        u_b = tr.RadTrajectory(self.l, self.T, self.param, "robin", "robin", n=self.n_y, show_plot=show_plots)
        u_b_t = u_b(time=self.t)
        self.assertTrue(all(np.isclose(u_b_t, u_a_t, atol=0.005)))
        if show_plots:
            pw = pg.plot(title="control_input")
            pw.plot(self.t, u_a_t)
            pw.plot(self.t, u_b_t)
            app.exec_()

