import numpy as np

from ..trajectory import sigma_tanh, K_tanh, gevrey_tanh, _power_series_flat_out, InterpolationTrajectory

__all__ = ["RadTrajectory"]


class RadTrajectory(InterpolationTrajectory):
    """
    Class that implements a flatness based control approach
    for the reaction-advection-diffusion equation

    .. math:: \\dot x(z,t) = a_2 x''(z,t) + a_1 x'(z,t) + a_0 x(z,t)

    with the boundary condition

        - :code:`bound_cond_type == "dirichlet"`: :math:`x(0,t)=0`

            - A transition from :math:`x'(0,0)=0` to  :math:`x'(0,T)=1` is considered.
            - With :math:`x'(0,t) = y(t)` where :math:`y(t)` is the flat output.

        - :code:`bound_cond_type == "robin"`: :math:`x'(0,t) = \\alpha x(0,t)`

            - A transition from :math:`x(0,0)=0` to  :math:`x(0,T)=1` is considered.
            - With :math:`x(0,t) = y(t)` where :math:`y(t)` is the flat output.

    and the actuation

        - :code:`actuation_type == "dirichlet"`: :math:`x(l,t)=u(t)`

        - :code:`actuation_type == "robin"`: :math:`x'(l,t) = -\\beta x(l,t) + u(t)`.

    The flat output trajectory :math:`y(t)` will be calculated with :py:func:`gevrey_tanh`.
    """

    # TODO: kwarg: t_step
    def __init__(self, l, T, param_original, bound_cond_type, actuation_type, n=80, sigma=sigma_tanh, K=K_tanh,
                 show_plot=False):

        cases = {'dirichlet', 'robin'}
        if bound_cond_type not in cases:
            raise TypeError('Type of boundary condition by z=0 is not understood.')
        if actuation_type not in cases:
            raise TypeError('Type of actuation_type is not understood.')

        self._l = l
        self._T = T
        self._a1_original = param_original[1]
        self._param = parabolic.general.eliminate_advection_term(param_original)
        self._bound_cond_type = bound_cond_type
        self._actuation_type = actuation_type
        self._n = n
        self._sigma = sigma
        self._K = K

        self._z = np.array([self._l])
        y, t = gevrey_tanh(self._T, self._n + 2, self._sigma, self._K)
        x, d_x = _power_series_flat_out(self._z, t, self._n, self._param, y, bound_cond_type)

        a2, a1, a0, alpha, beta = self._param
        if self._actuation_type is 'dirichlet':
            u = x[:, -1]
        elif self._actuation_type is 'robin':
            u = d_x[:, -1] + beta * x[:, -1]
        else:
            raise NotImplementedError

        # actually the algorithm consider the pde
        # d/dt x(z,t) = a_2 x''(z,t) + a_0 x(z,t)
        # with the following back transformation are also
        # pde's with advection term a_1 x'(z,t) considered
        u *= np.exp(-self._a1_original / 2. / a2 * l)

        InterpolationTrajectory.__init__(self, t, u, show_plot=show_plot)
