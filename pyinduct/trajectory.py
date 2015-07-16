from __future__ import division
import sympy as sp
import numpy as np


__author__ = 'stefan ecklebe'

class FlatString:
    """
    class that implements a flatness based control approach
    for the "string with mass" model
    """

    def __init__(self, y0=0, y1=1, t0=0, dt=0, m=1.0, v=1.0, z0=0, z1=1, sigma=1.0):
        # store params
        self._yA = y0
        self._yB = y1
        self._tA = t0
        self._dt = dt
        self._dz = z1 - z0

        # physical parameters
        self._m = m  # []=kg mass at z=0
        self._v = v  # []=m/s speed of wave translation in string
        self._sigma = sigma  # []=kgm/s**2 pretension of string

        # create trajectory for flat output and its derivatives
        t = sp.symbols("t")

        gamma = 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3
        gamma_d = gamma.diff(t)
        gamma_dd = gamma_d.diff(t)

        self._gamma_func = sp.lambdify(t, gamma)
        self._gamma_d_func = sp.lambdify(t, gamma_d)
        self._gamma_dd_func = sp.lambdify(t, gamma_dd)
        del t

        self.control_input = np.vectorize(self._control_input)
        self.system_state = np.vectorize(self._system_sate)

    def _trans_arg(self, t):
        """
        translate desired trajectory on time axis by moving its argument since values below t=0 are required otherwise
        :param t: time
        :return:translated time
        """
        return t - self._dz / self._v - self._tA

    # trajectory for flat output and derivatives
    def _y(self, t):
        """
        calculate desired trajectory
        :param t: time
        :return: y
        """
        t = self._trans_arg(t)
        # if t < self._tA:
        if t < 0:
            return self._yA
        # elif t < self._tA + self._dt:
        elif t < self._dt:
            return self._yA + (self._yB - self._yA) * self._gamma_func(t / self._dt)
        else:
            return self._yB

    def _yd(self, t):
        """
        calculate first derivative of desired trajectory
        :param t: time
        :return: yd
        """
        t = self._trans_arg(t)
        # if t < self._tA:
        if t < 0:
            return 0
        # elif t < self._tA + self._dt:
        elif t < self._dt:
            return (self._yB - self._yA) * self._gamma_d_func(t / self._dt) / self._dt
        else:
            return 0

    def _ydd(self, t):
        """
        calculate second derivative of desired trajectory
        :param t: time
        :return: ydd
        """
        t = self._trans_arg(t)
        # if t < self._tA:
        if t < 0:
            return 0
        # elif t < self._tA + self._dt:
        elif t < self._dt:
            return (self._yB - self._yA) * self._gamma_dd_func(t / self._dt) / self._dt ** 2
        else:
            return 0

    def _control_input(self, t):
        """
        control input for system gained through flatness based approach that will
        satisfy the target trajectory for y
        :param t: time
        :return: input force f
        """
        # if t - self._c * self._zB < 0:
        #     raise ValueError("Planned trajectory required control before t=0")

        return 0.5 * self._m * (self._ydd(t + self._dz / self._v) + self._ydd(t - self._dz / self._v)) \
               + self._sigma/(2*self._v) * (self._yd(t + self._dz / self._v) - self._yd(t - self._dz / self._v))

    def _system_sate(self, z, t):
        """
        x(z, t) of string-mass system for given flat output y
        :param z: location
        :param t: time
        :return: state (deflection of string)
        """
        return (self._v * self._m) / (2 * self._sigma) * (self._yd(t + z / self._v) - self._yd(t - z / self._v)) \
               + .5 * (self._y(t - z / self._v) + self._y(t + z / self._v))

