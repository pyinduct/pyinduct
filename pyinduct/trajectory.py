from __future__ import division
import sympy as sp
import numpy as np

from simulation import SimulationInput


__author__ = 'stefan ecklebe'


class SmoothTransition(object):
    """
    trajectory generator for a smooth transition between to states with derivatives of arbitrary height.
    """
    def __init__(self, states, interval, differential_order):
        """
        :param states: tuple of states in beginning and end of interval
        :param interval: time interval (tuple)
        :param differential_order: grade of differential flatness :math:`\\gamma`
        """
        self.yd = states
        self.t0 = interval[0]
        self.t1 = interval[1]
        self.dt = interval[1] - interval[0]
        gamma = differential_order + 1

        # setup symbolic expressions
        tau, k = sp.symbols('tau, k')

        alpha = sp.factorial(2 * gamma + 1)

        f = sp.binomial(gamma, k) * (-1) ** k * tau ** (gamma + k + 1) / (gamma + k + 1)
        phi = alpha / sp.factorial(gamma) ** 2 * sp.summation(f, (k, 0, gamma))

        # differentiate phi(tau), index in list corresponds to order
        dphi_sym = [phi]  # init with phi(tau)
        for order in range(differential_order):
            dphi_sym.append(dphi_sym[-1].diff(tau))

        # lambdify
        self.dphi_num = []
        for der in dphi_sym:
            self.dphi_num.append(sp.lambdify(tau, der, 'numpy'))

        self.desired_values = np.vectorize(self._desired_values)

    def __call__(self, time):
        return self.desired_values(time)

    def _desired_values(self, t):
        """
        Calculates desired trajectory and all derivatives for moment t

        :param t: time value for which trajectory and derivatives are needed
        :returns np.ndarray
        :math:`\\boldsymbol{y}_d = \\left(y_d, \\dot{y}_d, \\ddot{y}_d, \\dotsc, \\y_d^{(\\gamma)}\\right)`
        """
        y = np.zeros((len(self.dphi_num), 1))
        if t < self.t0:
            y[0] = self.yd[0]
        elif t > self.t1:
            y[0] = self.yd[1]
        else:
            for order, dphi in enumerate(self.dphi_num):
                if order == 0:
                    ya = self.yd[0]
                else:
                    ya = 0

                y[order] = ya + (self.yd[1] - self.yd[0])*dphi((t - self.t0)/self.dt)*1/self.dt**order

        return y


class FlatString(SimulationInput):
    """
    class that implements a flatness based control approach
    for the "string with mass" model
    """

    def __init__(self, y0=0, y1=1, t0=0, dt=1, m=1.0, v=1.0, z0=0, z1=1, sigma=1.0):
        # construct trajectory generator for yd
        self.trajectory_gen = SmoothTransition((y0, y1), (t0, t0 + dt), 2)

        # store params
        self._tA = t0
        self._dt = dt
        self._dz = z1 - z0
        self._m = m             # []=kg mass at z=0
        self._v = v             # []=m/s speed of wave translation in string
        self._sigma = sigma     # []=kgm/s**2 pretension of string

        # create vectorized functions
        self.control_input = np.vectorize(self._control_input)
        self.system_state = np.vectorize(self._system_sate)

    def _trans_arg(self, t):
        """
        translate desired trajectory on time axis by moving its argument since values below t=0 are required otherwise
        :param t: time
        :return:translated time
        """
        return t - self._dz / self._v - self._tA

    def _control_input(self, t):
        """
        control input for system gained through flatness based approach that will
        satisfy the target trajectory for y

        :param t: time
        :return: input force f
        """
        yd1 = self.trajectory_gen(self._trans_arg(t - self._dz / self._v))
        yd2 = self.trajectory_gen(self._trans_arg(t + self._dz / self._v))

        return 0.5*self._m*(yd2[2] + yd1[2]) + self._sigma/(2*self._v)*(yd2[1] - yd1[1])

    def _system_sate(self, z, t):
        """
        x(z, t) of string-mass system for given flat output y
        :param z: location
        :param t: time
        :return: state (deflection of string)
        """
        yd1 = self.trajectory_gen(self._trans_arg(t - z / self._v))
        yd2 = self.trajectory_gen(self._trans_arg(t + z / self._v))

        return (self._v*self._m)/(2*self._sigma)*(yd2[1] - yd1[1]) + .5*(yd1[0] + yd2[0])

    def __call__(self, t, q=None, **kwargs):
        """
        use time to calculate system input and return force
        :param t:
        :param q:
        :param kwargs:
        :return:
        """
        return self._control_input(t)
