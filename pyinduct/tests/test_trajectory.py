import unittest
import warnings

import numpy as np
import scipy.signal as sig

import pyinduct as pi
import pyinduct.hyperbolic.feedforward as hff
import pyinduct.parabolic as parabolic
from pyinduct.tests import show_plots
import pyqtgraph as pg


class ConstantTrajectoryTestCase(unittest.TestCase):
    def setUp(self):
        self.spat_domain = pi.Domain(bounds=(0, 1), num=2)
        self.temp_domain = pi.Domain(bounds=(0, 1), num=10)
        self.const_traj = pi.ConstantTrajectory(1)

    def test_const_traj(self):
        self.assertAlmostEqual(self.const_traj(time=1), 1)
        np.testing.assert_array_almost_equal(self.const_traj(time=np.arange(10)), np.ones((10,)))
        with self.assertRaises(NotImplementedError):
            self.const_traj(time=(1,))


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

        self.params = pi.Parameters
        self.params.m = 1
        self.params.sigma = 1
        self.params.tau = 2

    def test_trajectory(self):
        # build flatness based trajectory generator
        fs = hff.FlatString(y0=self.y0, y1=self.y1, z0=self.z_start, z1=self.z_end, t0=self.t_start, dt=2,
                                               params=self.params)
        zz, tt = np.meshgrid(self.z_values, self.t_values)
        x_values = fs.system_state(zz, tt)
        u_values = fs.control_input(self.t_values)
        eval_data_x = pi.EvalData([self.t_values, self.z_values], x_values)

        if show_plots:
            # plot stuff
            pw = pg.plot(title="control_input")
            pw.plot(self.t_values, u_values)
            ap = pi.PgAnimatedPlot(eval_data_x)
            pi.show(show_mpl=False)


class FormalPowerSeriesTest(unittest.TestCase):
    def setUp(self):

        self.l = 1;
        self.T = 1
        a2 = 1;
        a1 = 0;
        a0 = 6;
        self.alpha = 0.5;
        self.beta = 0.5
        self.param = [a2, a1, a0, self.alpha, self.beta]
        self.n_y = 80
        self.y, self.t = pi.gevrey_tanh(self.T, self.n_y, 1.1, 2)

    def test_temporal_derive(self):

        b_desired = 0.4
        k = 5  # = k1 + k2
        k1, k2, b = parabolic.split_domain(k, b_desired, self.l, mode='coprime')[0:3]
        # q
        E = pi.coefficient_recursion(self.y, self.beta * self.y, self.param)
        q = pi.temporal_derived_power_series(self.l - b, E, int(self.n_y / 2) - 1, self.n_y)
        # u
        B = pi.coefficient_recursion(self.y, self.alpha * self.y, self.param)
        xq = pi.temporal_derived_power_series(self.l, B, int(self.n_y / 2) - 1, self.n_y, spatial_der_order=0)
        d_xq = pi.temporal_derived_power_series(self.l, B, int(self.n_y / 2) - 1, self.n_y, spatial_der_order=1)
        u = d_xq + self.beta * xq
        # x(0,t)
        C = pi.coefficient_recursion(q, self.beta * q, self.param)
        D = pi.coefficient_recursion(np.zeros(u.shape), u, self.param)
        x_0t = pi.power_series(0, self.t, C)
        if show_plots:
            pw = pg.plot(title="control_input")
            pw.plot(self.t, x_0t)
            pi.show(show_mpl=False)

    def test_recursion_vs_explicit(self):

        # recursion
        B = pi.coefficient_recursion(self.y, self.alpha * self.y, self.param)
        x_l = pi.power_series(self.l, self.t, B)
        d_x_l = pi.power_series(self.l, self.t, B, spatial_der_order=1)
        u_c = d_x_l + self.beta * x_l
        u_a = pi.InterpolationTrajectory(self.t, u_c, show_plot=show_plots)
        u_a_t = u_a(time=self.t)
        # explicit
        u_b = parabolic.RadFeedForward(self.l, self.T, self.param, "robin", "robin", n=self.n_y,
                                                   show_plot=show_plots)
        u_b_t = u_b(time=self.t)
        self.assertTrue(all(np.isclose(u_b_t, u_a_t, atol=0.005)))
        if show_plots:
            pw = pg.plot(title="control_input")
            pw.plot(self.t, u_a_t)
            pw.plot(self.t, u_b_t)
            pi.show(show_mpl=False)


class InterpSignalGeneratorTest(unittest.TestCase):
    def setUp(self):
        if not any([sig_form in sig.waveforms.__all__ for sig_form in
                    ['sawtooth', 'square', 'gausspulse', 'chirp', 'sweep_poly']]):
            warnings.warn("New scipy.signal module interface!"
                          "Rewrite this TestCase (and have a look at pyinduct.trajectory.SignalGenerator!")

        self.t = pi.Domain(bounds=(0, 1), num=500)
        self.t_interp = pi.Domain(bounds=(0, 1), num=50)
        self.t1 = 1
        self.f0 = 50
        self.f1 = 500
        self.width = 100
        self.poly = np.array([1, 1, 1])
        self.no_plot = False

    def test_sawtooth(self):
        self.sig_gen = pi.SignalGenerator('sawtooth', self.t, offset=0.5, scale=0.5, frequency=5)
        np.testing.assert_array_almost_equal(np.array([0, 1, 1, 1]),
                                             self.sig_gen.__call__(time=np.array([0, .2, .4, .6]) - 2e-3), decimal=2)
        np.testing.assert_array_almost_equal(np.array([0, .5, .5, .5]),
                                             self.sig_gen.__call__(time=np.array([0, .1, .3, .5]) - 2e-3), decimal=2)

    def test_square(self):
        self.sig_gen = pi.SignalGenerator('square', self.t, offset=0.5, scale=0.5, frequency=5)
        np.testing.assert_array_almost_equal(np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0]),
                                             self.sig_gen.__call__(
                                                 time=np.array(
                                                     [0, .04, .06, .14, .16, .24, .26, .34, .36, .94, .96, ])),
                                             decimal=5)

    @unittest.skip("gausspulse gives an underflow error")
    def test_gausspulse(self):
        self.sig_gen = pi.SignalGenerator('gausspulse', self.t, phase_shift=0.5)
        np.testing.assert_array_almost_equal(np.array([0, 0, 0, 0, 0, .4, 0, 0, 0, 0]),
                                             self.sig_gen.__call__(time=np.arange(0, 1, 0.1)), dezimal=2)

    def test_kwarg(self):
        self.no_plot = True
        with self.assertWarns(UserWarning):
            pi.SignalGenerator('square', self.t, offset=0.5, scale=0.5)
        with self.assertRaises(NotImplementedError):
            pi.SignalGenerator('gausspulse', self.t, frequency=5)

    def tearDown(self):
        if hasattr(self, "sig_gen"):
            if show_plots and not self.no_plot:
                pw = pg.plot(title="control_input")
                pw.plot(self.t.points, self.sig_gen.__call__(time=self.t), pen='c')
                pw.plot(self.t_interp.points, self.sig_gen.__call__(time=self.t_interp), pen='g')
                pi.show(show_mpl=False)

