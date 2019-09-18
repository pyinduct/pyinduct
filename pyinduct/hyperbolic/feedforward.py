import numpy as np

from ..simulation import SimulationInput
from ..trajectory import SmoothTransition

__all__ = ["FlatString"]


class FlatString(SimulationInput):
    """
    Flatness based feedforward for the "string with mass" model.

    The flat output :math:`y` of this system is given by the mass position
    at :math:`z = z_0` . This output will be transferred from *y0* to *y1*
    starting at *t0*, lasting *dt* seconds.

    Args:
        y0 (float): Initial value for the flat output.
        y1 (float): Final value for the flat output.
        z0 (float): Position of the flat output (left side of the string).
        z1 (float): Position of the actuation (right side of the string).
        t0 (float): Time to start the transfer.
        dt (float): Duration of the transfer.
        params (bunch): Structure containing the physical parameters:
            * m: the mass
            * tau: the
            * sigma: the strings tension
    """
    def __init__(self, y0, y1, z0, z1, t0, dt, params):
        SimulationInput.__init__(self)

        # store params
        self._tA = t0
        self._dt = dt
        self._dz = z1 - z0
        self._m = params.m  # []=kg mass at z=0
        self._tau = params.tau  # []=m/s speed of wave translation in string
        self._sigma = params.sigma  # []=kgm/s**2 pretension of string

        # construct trajectory generator for yd
        ts = max(t0, self._dz * self._tau)  # never too early
        self.trajectory_gen = SmoothTransition((y0, y1), (ts, ts + dt),
                                               method="poly",
                                               differential_order=2)

        # create vectorized functions
        self.control_input = np.vectorize(self._control_input,
                                          otypes=[np.float])
        self.system_state = np.vectorize(self._system_sate,
                                         otypes=[np.float])

    def _control_input(self, t):
        """
        Calculate the control input for system, by using flatness based
        approach that will satisfy the target trajectory for y.

        Args:
            t (float): Time stamp to evaluate at.

        Return:
            float: Input force f.
        """
        yd1 = self.trajectory_gen(t - self._dz * self._tau)
        yd2 = self.trajectory_gen(t + self._dz * self._tau)

        return (0.5 * self._m * (yd2[2] + yd1[2])
                + self._sigma * self._tau / 2 * (yd2[1] - yd1[1]))

    def _system_sate(self, z, t):
        """
        Calculate the systems state :math:`x(z, t)` from a given flat output y.

        Args:
            z: Location for evaluation.
            t: Time stamp for evaluation.

        Return:
            float: String deflection at given coordinates.
        """
        yd1 = self.trajectory_gen(t - z * self._tau)
        yd2 = self.trajectory_gen(t + z * self._tau)

        return (self._m / (2 * self._sigma * self._tau) * (yd2[1] - yd1[1])
                + .5 * (yd1[0] + yd2[0]))

    def _calc_output(self, **kwargs):
        """
        Use time to calculate system input and return force.

        Keyword Args:
            time:

        Return:
            dict: Result is the value of key "output".
        """
        return dict(output=self._control_input(kwargs["time"]))


