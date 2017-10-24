import numpy as np

from .general import eliminate_advection_term
from ..trajectory import (
    InterpolationTrajectory, gevrey_tanh, SecondOrderOperator,
    power_series_flat_out)

__all__ = ["RadFeedForward"]


class RadFeedForward(InterpolationTrajectory):
    r"""
    Class that implements a flatness based control approach
    for the reaction-advection-diffusion equation

    .. math:: \dot x(z,t) = a_2 x''(z,t) + a_1 x'(z,t) + a_0 x(z,t)

    with the boundary condition

        - :code:`bound_cond_type == "dirichlet"`: :math:`x(0,t)=0`

            - A transition from :math:`x'(0,0)=y0` to  :math:`x'(0,T)=y1` is
              considered.
            - With :math:`x'(0,t) = y(t)` where :math:`y(t)` is the flat output.

        - :code:`bound_cond_type == "robin"`: :math:`x'(0,t) = \alpha x(0,t)`

            - A transition from :math:`x(0,0)=y0` to  :math:`x(0,T)=y1` is
              considered.
            - With :math:`x(0,t) = y(t)` where :math:`y(t)` is the flat output.

    and the actuation

        - :code:`actuation_type == "dirichlet"`: :math:`x(l,t)=u(t)`

        - :code:`actuation_type == "robin"`:
          :math:`x'(l,t) = -\beta x(l,t) + u(t)`.

    The flat output trajectory :math:`y(t)` will be calculated with
    :py:func:`.gevrey_tanh`.

    Args:
        l (float): Domain length.
        t_end (float): Transition time.
        param_original (tuple): Tuple holding the coefficients of the pde and
            boundary conditions.
        bound_cond_type (string): Boundary condition type. Can be `dirichlet` or
            `robin`,  see above.
        actuation_type (string): Actuation condition type. Can be `dirichlet` or
            `robin`,  see above.
        n (int): Derivative order to provide (defaults to 80).
        sigma (float): Sigma value for :py:func:`.gevrey_tanh`.
        k (float): k value for :py:func:`.gevrey_tanh`.
        y0 (float): Initial value for the flat output.
        y1 (float): Desired value for the flat output after transition time.
        **kwargs: see below. All arguments that are not specified below
            are passed to :py:class:`.InterpolationTrajectory` .

    """

    def __init__(self, l, T, param_original, bound_cond_type, actuation_type,
                 n=80, sigma=None, k=None, y_start=0, y_end=1,
                 **kwargs):

        cases = {"dirichlet", "robin"}
        if bound_cond_type not in cases:
            raise TypeError(
                "Type of boundary condition by z=0 is not understood.")
        if actuation_type not in cases:
            raise TypeError("Type of actuation_type is not understood.")

        self._l = l
        self._T = T
        self._param = eliminate_advection_term(param_original, l)
        self._bound_cond_type = bound_cond_type
        self._actuation_type = actuation_type
        self._n = n
        self._z = np.array([self._l])

        kwargs = dict()
        kwargs.update(sigma=sigma) if sigma is not None else None
        kwargs.update(K=k) if k is not None else None
        delta, t = gevrey_tanh(self._T, self._n + 2, **kwargs)
        y = delta * (y_end - y_start)
        y[0, :] += y_start
        x, d_x = power_series_flat_out(self._z,
                                       t,
                                       self._n,
                                       self._param,
                                       y,
                                       bound_cond_type)

        if isinstance(self._param, SecondOrderOperator):
            self._a1_original = param_original.a1
            a2 = self._param.a2
            beta = self._param.beta0
        else:
            a2, a1, a0, alpha, beta = self._param
            self._a1_original = param_original[1]

        if self._actuation_type is 'dirichlet':
            u = x[:, -1]
        elif self._actuation_type is 'robin':
            u = d_x[:, -1] + beta * x[:, -1]
        else:
            raise NotImplementedError

        """
        actually the algorithm only considers the pde
        d/dt x(z,t) = a_2 x''(z,t) + a_0 x(z,t)
        but with the following back transformation also
        pdes with advection term a_1 x'(z,t) can be considered
        """
        u *= np.exp(-self._a1_original / 2. / a2 * l)

        InterpolationTrajectory.__init__(self, t, u, **kwargs)
