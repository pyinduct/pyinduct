import numpy as np
import scipy.special as ss

from .general import eliminate_advection_term
from ..eigenfunctions import SecondOrderOperator
from ..trajectory import InterpolationTrajectory, gevrey_tanh

__all__ = ["RadFeedForward", "power_series_flat_out"]


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
        sigma (number.Number): `sigma` value for :py:func:`.gevrey_tanh`.
        k (number.Number): `K` value for :py:func:`.gevrey_tanh`.
        length_t (int): `length_t` value for :py:func:`.gevrey_tanh`.
        y0 (float): Initial value for the flat output.
        y1 (float): Desired value for the flat output after transition time.
        **kwargs: see below. All arguments that are not specified below
            are passed to :py:class:`.InterpolationTrajectory` .

    """

    def __init__(self, l, T, param_original, bound_cond_type, actuation_type,
                 n=80, sigma=None, k=None, length_t=None, y_start=0, y_end=1,
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

        gt_kwargs = dict()
        gt_kwargs.update(sigma=sigma) if sigma is not None else None
        gt_kwargs.update(K=k) if k is not None else None
        gt_kwargs.update(length_t=length_t) if k is not None else None
        delta, t = gevrey_tanh(self._T, self._n + 2, **gt_kwargs)
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

        # actually the algorithm only considers the pde
        # d/dt x(z,t) = a_2 x''(z,t) + a_0 x(z,t)
        # but with the following back transformation also
        # pdes with advection term a_1 x'(z,t) can be considered
        u *= np.exp(-self._a1_original / 2. / a2 * l)

        InterpolationTrajectory.__init__(self, t, u, **kwargs)


def power_series_flat_out(z, t, n, param, y, bound_cond_type):
    """
    Provide the solution x(z,t) and x'(z,t) of the pde

    .. math:: \dot x(z,t) = a_2 x''(z,t) + a_1 x'(z,t) + a_0 x(z,t)

    with

        - the boundary condition :code:`bound_cond_type == "dirichlet"` /
            :math:`x(0,t)=0` and the flat output :math:`y(t) = x'(0,t)`

        - the boundary condition :code:`bound_cond_type == "robin"`: /
            :math:`x'(0,t) = \alpha x(0,t)` and the flat output
            :math:`y(t) = x(0,t)`

    as power series approximation.

    Args:
        z (numpy.ndarray): [0, ..., l]
        t (numpy.ndarray): [0, ... , t_end]
        n (int): Series termination index.
        param (array_like): [a2, a1, a0, alpha, beta]
        y (array_like): Flat output and derivatives
            np.array([[y],...,[y^(n/2)]]).

    Return:
        Field variable x(z,t) and spatial derivative x'(z,t).
    """

    if isinstance(param, SecondOrderOperator):
        a2 = param.a2
        a1 = param.a1
        a0 = param.a0
        alpha = -param.alpha0
        beta = param.beta0

    else:
        a2, a1, a0, alpha, beta = param

    shape = (len(t), len(z))
    x = np.nan * np.ones(shape)
    d_x = np.nan * np.ones(shape)

    # Actually power_series() is designed for robin boundary condition by z=0.
    # With the following modification it can also used for dirichlet boundary
    # condition by z=0.
    if bound_cond_type is 'robin':
        is_robin = 1.
    elif bound_cond_type is 'dirichlet':
        alpha = 1.
        is_robin = 0.
    else:
        raise ValueError(
            "Selected boundary condition {0} not supported! "
            "Use 'robin' or 'dirichlet'".format(bound_cond_type))

    # TODO: flip iteration order: z <--> t,
    #   result: one or two instead len(t) call's
    for i in range(len(t)):
        sum_x = np.zeros(len(z))
        for j in range(n):
            sum_b = np.zeros(len(z))
            for k in range(j + 1):
                sum_b += ss.comb(j, k) * (-a0) ** (j - k) * y[k, i]
            sum_x += ((is_robin + alpha * z / (2. * j + 1.)) *
                      z ** (2 * j) / ss.factorial(2 * j) / a2 ** j * sum_b)
        x[i, :] = sum_x

    for i in range(len(t)):
        sum_x = np.zeros(len(z))
        for j in range(n):
            sum_b = np.zeros(len(z))
            for k in range(j + 2):
                sum_b += ss.comb(j + 1, k) * (-a0) ** (j - k + 1) * y[k, i]
            if j == 0:
                sum_x += alpha * y[0, i]
            sum_x += ((is_robin + alpha * z / (2. * (j + 1))) *
                      z ** (2 * j + 1) / ss.factorial(2 * j + 1) /
                      a2 ** (j + 1) * sum_b)
        d_x[i, :] = sum_x

    return x, d_x
