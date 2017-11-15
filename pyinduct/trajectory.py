"""
In the module :py:mod:`pyinduct.trajectory` are some trajectory generators defined.
Besides you can find here a trivial (constant) input signal generator as
well as input signal generator for equilibrium to equilibrium transitions for
hyperbolic and parabolic systems.
"""
import warnings
from numbers import Number

import numpy as np
import pyqtgraph as pg
import scipy.misc as sm
import scipy.signal as sig
import sympy as sp

from .core import Domain
from .simulation import SimulationInput
from .eigenfunctions import SecondOrderOperator

__all__ = ["ConstantTrajectory", "InterpolationTrajectory",
           "SmoothTransition",  "SignalGenerator",
           "gevrey_tanh", "power_series", "temporal_derived_power_series",
           "coefficient_recursion"]


class ConstantTrajectory(SimulationInput):
    """
    Trivial trajectory generator for a constant value as simulation input signal.

    Args:
        const (numbers.Number): Desired constant value of the output.
    """

    def __init__(self, const=0, name=""):
        super().__init__(name)
        self._const = const

    def _calc_output(self, **kwargs):
        if isinstance(kwargs["time"], (list, np.ndarray)):
            return dict(output=np.ones(len(np.atleast_1d(kwargs["time"]))) * self._const)
        elif isinstance(kwargs["time"], Number):
            return dict(output=self._const)
        else:
            raise NotImplementedError


class SmoothTransition:
    r"""
    A smooth transition between two given steady-states *states* on an
    *interval* using either:

        - polynomial method

        - trigonometric method

    To create smooth transitions.

    Args:
        states (tuple): States at beginning and end of interval.
        interval (tuple): Time interval.
        method (str): Method to use (``poly`` or ``tanh``).
        differential_order (int): Grade of differential flatness :math:`\gamma`.
    """

    def __init__(self, states, interval, method, differential_order=0):
        self.yd = states
        self.t0 = interval[0]
        self.t1 = interval[1]
        self.dt = interval[1] - interval[0]

        # setup symbolic expressions
        if method == "tanh":
            tau, sigma = sp.symbols('tau, sigma')
            # use a gevrey-order of alpha = 1 + 1/sigma
            sigma = 1.1
            phi = .5 * (1 + sp.tanh(2 * (2 * tau - 1) / ((4 * tau * (1 - tau)) ** sigma)))

        elif method == "poly":
            gamma = differential_order  # + 1 # TODO check this against notes
            tau, k = sp.symbols('tau, k')

            alpha = sp.factorial(2 * gamma + 1)

            f = sp.binomial(gamma, k) * (-1) ** k * tau ** (gamma + k + 1) / (gamma + k + 1)
            phi = alpha / sp.factorial(gamma) ** 2 * sp.summation(f, (k, 0, gamma))
        else:
            raise NotImplementedError("method {} not implemented!".format(method))

        # differentiate phi(tau), index in list corresponds to order
        dphi_sym = [phi]  # init with phi(tau)
        for order in range(differential_order):
            dphi_sym.append(dphi_sym[-1].diff(tau))

        # lambdify
        self.dphi_num = []
        for der in dphi_sym:
            self.dphi_num.append(sp.lambdify(tau, der, 'numpy'))

    def __call__(self, *args, **kwargs):
        return self._desired_values(args[0])

    def _desired_values(self, t):
        r"""
        Calculates the desired trajectory and its derivatives for time-step `t`.

        Args:
            t (array_like): Time-step for which trajectory and derivatives are
                needed.

        Return:
            numpy.ndarray:

                .. math::

                    \boldsymbol{y}_d = \left(y_d, \dot{y}_d, \ddot{y}_d,
                    \dotsc, y_d^{(\gamma)}\right)
        """
        y = np.zeros((len(self.dphi_num)))
        if t <= self.t0:
            y[0] = self.yd[0]
        elif t >= self.t1:
            y[0] = self.yd[1]
        else:
            for order, dphi in enumerate(self.dphi_num):
                if order == 0:
                    ya = self.yd[0]
                else:
                    ya = 0

                y[order] = ya + (self.yd[1] - self.yd[0]) * dphi((t - self.t0) / self.dt) * 1 / self.dt ** order

        return y


def gevrey_tanh(T, n, sigma=1.1, K=2, length_t=None):
    r"""
    Provide Gevrey function

    .. math::
        \eta(t) = \left\{ \begin{array}{lcl}
            0 & \forall & t<0 \\
            \frac{1}{2} + \frac{1}{2}\tanh \left(K \frac{2(2t-1)}
            {(4(t^2-t))^\sigma} \right) & \forall & 0\le t \le T \\
            1 & \forall & t>T  \end{array} \right.

    with the Gevrey-order :math:`\rho=1+\frac{1}{\sigma}` and the derivatives
    up to order n.

    Note:

        For details of the recursive calculation of the derivatives see:

            Rudolph, J., J. Winkler und F. Woittennek: Flatness Based Control of
            Distributed Parameter Systems: Examples and Computer Exercises from
            Various Technological Domains (Berichte aus der Steuerungs- und
            Regelungstechnik). Shaker Verlag GmbH, Germany, 2003.

    Args:
        T (numbers.Number): End of the time domain=[0, T].
        n (int): The derivatives will calculated up to order n.
        sigma (numbers.Number): Constant :math:`\sigma` to adjust the Gevrey
            order :math:`\rho=1+\frac{1}{\sigma}` of :math:`\varphi(t)`.
        K (numbers.Number): Constant to adjust the slope of :math:`\varphi(t)`.
        length_t (int): Ammount of sample points to use.
            Default: :code:`50 * T`

    Return:
        tuple:
        - numpy.array([[:math:`\varphi(t)`], ... , [:math:`\varphi^{(n)}(t)`]])
        - t: numpy.array([0,...,T])
    """

    if length_t is None:
        length_t = 50 * T

    t_init = t = np.linspace(0., T, length_t)

    # pop
    t = np.delete(t, 0, 0)
    t = np.delete(t, -1, 0)

    # main
    tau = t / T

    a = dict()
    a[0] = K * (4 * tau * (1 - tau)) ** (1 - sigma) / (2 * (sigma - 1))
    a[1] = (2 * tau - 1) * (sigma - 1) / (tau * (1 - tau)) * a[0]
    for k in range(2, n + 2):
        a[k] = (tau * (1 - tau)) ** -1 * (
            (sigma - 2 + k) * (2 * tau - 1) * a[k - 1] + (k - 1) * (2 * sigma - 4 + k) * a[k - 2])

    yy = dict()
    yy[0] = np.tanh(a[1])
    if n > 0:
        yy[1] = a[2] * (1 - yy[0] ** 2)
    z = dict()
    z[0] = (1 - yy[0] ** 2)
    for i in range(2, n + 1):
        sum_yy = np.zeros(len(t))
        for k in range(i):
            if k == 0:
                sum_z = np.zeros(len(t))
                for j in range(i):
                    sum_z += -sm.comb(i - 1, j) * yy[j] * yy[i - 1 - j]
                z[i - 1] = sum_z
            sum_yy += sm.comb(i - 1, k) * a[k + 2] * z[i - 1 - k]
        yy[i] = sum_yy

    # push
    phi = np.nan * np.zeros((n + 1, len(t) + 2))
    for i in range(n + 1):
        phi_temp = 0.5 * yy[i]
        if i == 0:
            phi_temp += 0.5
            phi_temp = np.insert(phi_temp, 0, [0.], axis=0)
            phi[i, :] = np.append(phi_temp, [1.])
        else:
            phi_temp = np.insert(phi_temp, 0, [0.], axis=0)
            # attention divide by T^i
            phi[i, :] = np.append(phi_temp, [0.]) / T ** i

    return phi, t_init


def power_series_flat_out(z, t, n, param, y, bound_cond_type):
    """
    Provide the power series approximation for x(z,t) and x'(z,t).

    Args:
        z (numpy.ndarray): [0, ..., l]
        t (numpy.ndarray): [0, ... , t_end]
        n (int): Series termination index.
        param (array_like): [a2, a1, a0, alpha, beta]
        y (array_like): Flat output and derivatives np.array([[y],...,[y^(n/2)]]).

    Return:
        Field variable x(z,t) and spatial derivative x'(z,t).
    """
    # TODO: documentation
    # TODO this is more a feedforward than a trajectory -> move

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
    # With the following modification it can also used for dirichlet boundary condition by z=0.
    if bound_cond_type is 'robin':
        is_robin = 1.
    elif bound_cond_type is 'dirichlet':
        alpha = 1.
        is_robin = 0.
    else:
        raise ValueError(
            "Selected Boundary condition {0} not supported! Use 'robin' or 'dirichlet'".format(bound_cond_type))

    # TODO: flip iteration order: z <--> t, result: one or two instead len(t) call's
    for i in range(len(t)):
        sum_x = np.zeros(len(z))
        for j in range(n):
            sum_b = np.zeros(len(z))
            for k in range(j + 1):
                sum_b += sm.comb(j, k) * (-a0) ** (j - k) * y[k, i]
            sum_x += (is_robin + alpha * z / (2. * j + 1.)) * z ** (2 * j) / sm.factorial(2 * j) / a2 ** j * sum_b
        x[i, :] = sum_x

    for i in range(len(t)):
        sum_x = np.zeros(len(z))
        for j in range(n):
            sum_b = np.zeros(len(z))
            for k in range(j + 2):
                sum_b += sm.comb(j + 1, k) * (-a0) ** (j - k + 1) * y[k, i]
            if j == 0:
                sum_x += alpha * y[0, i]
            sum_x += (is_robin + alpha * z / (2. * (j + 1))) * z ** (2 * j + 1) / sm.factorial(2 * j + 1) / a2 ** (
                j + 1) * sum_b
        d_x[i, :] = sum_x

    return x, d_x


def coefficient_recursion(c0, c1, param):
    r"""
    Return to the recursion

    .. math:: c_k(t) = \frac{ \dot c_{k-2}(t) - a_1 c_{k-1}(t)
        - a_0 c_{k-2}(t) }{ a_2 }

    with initial values

    .. math::
        c_0 = numpy.array([c_0^{(0)}, ... , c_0^{(N)}]) \\
        c_1 = numpy.array([c_1^{(0)}, ... , c_1^{(N)}])

    as much as computable subsequent coefficients

    .. math::

        c_2 = numpy.array&([c_2^{(0)}, ... , c_2^{(N-1)}])   \\
        c_3 = numpy.array&([c_3^{(0)}, ... , c_3^{(N-1)}])   \\
        &\vdots                                          \\
        c_{2N-1} = numpy.array&([c_{2N-1}^{(0)}])                \\
        c_{2N} = numpy.array&([c_{2N}^{(0)}])

    Args:
        c0 (array_like): :math:`c_0`
        c1 (array_like): :math:`c_1`
        param (array_like): (a_2, a_1, a_0, None, None)

    Return:
        dict: :math:`C = \{0: c_0, 1: c_1, ..., 2N-1: c_{2N-1}, 2N: c_{2N}\}`
    """
    # TODO: documentation: only constant coefficients
    if c0.shape != c1.shape:
        raise ValueError

    a2, a1, a0, _, _ = param
    N = c0.shape[0]
    C = dict()

    C[0] = c0
    C[1] = c1

    for i in range(2, 2 * N):
        reduced_derivative_order = int(i / 2.)
        C[i] = np.nan * np.zeros((N - reduced_derivative_order, c0.shape[1]))
        for j in range(N - reduced_derivative_order):
            C[i][j, :] = (C[i - 2][j + 1, :] - a1 * C[i - 1][j, :] - a0 * C[i - 2][j, :]) / a2

    return C


def temporal_derived_power_series(z, C, up_to_order, series_termination_index, spatial_der_order=0):
    r"""
    Compute the temporal derivatives

    .. math:: q^{(j,i)}(z=z^*,t) = \sum_{k=0}^{N}
        \underbrace{c_{k+j}^{(i)}}_{\text{C[k+j][i,:]}}
        \frac{{z^*}^k}{k!}, \qquad i=0,...,n.

    Args:
        z (numbers.Number): Evaluation point :math:`z^*`.
        C (dict): Coefficient dictionary whose keys correspond to the
            coefficient index. The values are 2D numpy.arrays. For example C[1]
            should provide a 2d-array with the coefficient :math:`c_1(t)` and at
            least :math:`n` temporal derivatives

            .. math:: \text{np.array}([c_1^{(0)}(t), ... , c_1^{(i)}(t)]) .

        up_to_order (int): Maximum temporal derivative order :math:`n` to
            compute.
        series_termination_index (int): Series termination index :math:`N` .
        spatial_der_order (int): Spatial derivative order :math:`j` .

    Return:
        numpy.ndarray: array holding the elements
        :math:`q^{(j,0)}, \dotsc, q^{(j,n)}`
    """

    if not isinstance(z, Number):
        raise TypeError
    if any([C[i].shape[0] - 1 < up_to_order for i in range(series_termination_index + 1)]):
        raise ValueError

    len_t = C[0].shape[1]
    Q = np.nan * np.zeros((up_to_order + 1, len_t))

    for i in range(up_to_order + 1):
        sum_Q = np.zeros(len_t)
        for j in range(series_termination_index + 1 - spatial_der_order):
            sum_Q += C[j + spatial_der_order][i, :] * z ** (j) / sm.factorial(j)
        Q[i, :] = sum_Q

    return Q


def power_series(z, t, C, spatial_der_order=0, temporal_der_order=0):
    r"""
    Compute the function values

    .. math:: x^{(j,i)}(z,t)=\sum_{k=0}^{N} c_{k+j}^{(i)}(t) \frac{z^k}{k!}.

    Args:
        z (array_like): Spatial steps to compute.
        t (array like): Temporal steps to compute.
        C (dict):
            Coeffient dictionary which keys correspond to the coefficient index.
            The values are 2D numpy.array's. For example C[1] should provide a
            2d-array with the coefficient :math:`c_1(t)` and at least :math:`i`
            temporal derivatives

            .. math:: \text{np.array}([c_1^{(0)}(t), ... , c_1^{(i)}(t)]).

        spatial_der_order (int): Spatial derivative order :math:`j`.
        temporal_der_order (int): Temporal derivative order :math:`i`.

    Return:
        numpy.array: Array of shape (len(t), len(z)).
    """
    if not all([isinstance(item, (Number, np.ndarray)) for item in [z, t]]):
        raise TypeError
    z = np.atleast_1d(z)
    t = np.atleast_1d(t)
    if not all([len(item.shape) == 1 for item in [z, t]]):
        raise ValueError

    x = np.nan * np.zeros((len(t), len(z)))
    for i in range(len(z)):
        sum_x = np.zeros(t.shape[0])
        for j in range(len(C) - spatial_der_order):
            sum_x += C[j + spatial_der_order][temporal_der_order, :] * z[i] ** j / sm.factorial(j)
        x[:, i] = sum_x

    if any([dim == 1 for dim in x.shape]):
        x = x.flatten()

    return x


class InterpolationTrajectory(SimulationInput):
    """
    Provides a system input through one-dimensional linear interpolation in
    the given vector :math:`u` .

    Args:
        t (array_like): Vector :math:`t` with time steps.
        u (array_like): Vector :math:`u` with function values, evaluated at :math:`t`.
        **kwargs: see below

    Keyword Args:
        show_plot (bool): to open a plot window, showing u(t).
        scale (float): factor to scale the output.

    """

    def __init__(self, t, u, **kwargs):
        SimulationInput.__init__(self)

        self._t = t
        self._T = t[-1]
        self._u = u * kwargs.get("scale", 1)

        if kwargs.get("show_plot", False):
            self.get_plot()

    def _calc_output(self, **kwargs):
        return dict(output=np.interp(kwargs["time"], self._t, self._u))

    def scale(self, scale):
        self._u *= scale

    def get_plot(self):
        """
        Create a plot of the interpolated trajectory.

        Todo:
            the function name does not really tell that a QtEvent loop will be executed in here

        Returns:
            (pg.PlotWindow): the PlotWindow widget.
        """
        pw = pg.plot(title="InterpTrajectory", labels=dict(left='u(t)', bottom='t'), pen='b')
        pw.plot(self._t, self.__call__(time=self._t), pen='b')
        pg.QtGui.QApplication.instance().exec_()
        return pw


class SignalGenerator(InterpolationTrajectory):
    """
    Signal generator that combines :py:mod:`scipy.signal.waveforms` and
    :py:class:`.InterpTrajectory`.

    Args:
        waveform (str): A waveform which is provided from
            :py:mod:`scipy.signal.waveforms`.
        t (array_like): Array with time steps or :py:class:`.Domain` instance.
        scale (numbers.Number): Scale factor:
            output = waveform_output * scale + offset.
        offset (numbers.Number): Offset value:
            output = waveform_output * scale + offset.
        kwargs: The corresponding keyword arguments to the desired
            :py:mod:`scipy.signal` waveform. In addition to the kwargs of the
            desired waveform function from scipy.signal (which will simply
            forwarded) the keyword arguments :py:obj:`frequency`
            (for waveforms: 'sawtooth' and 'square') and :py:obj:`phase_shift`
            (for all waveforms) provided.
    """

    def __init__(self, waveform, t, scale=1, offset=0, **kwargs):
        if waveform not in sig.waveforms.__all__:
            raise ValueError('Desired waveform is not provided by scipy.signal module.')
        if isinstance(t, Domain):
            t = t.points
        if not any([isinstance(value, Number) for value in [scale, offset]]):
            raise ValueError('scale and offset must be a Number')
        self._signal = getattr(sig, waveform)

        if waveform in {'sawtooth', 'square'}:
            # pop not scipy.signal.waveform.__all__ kwarg
            try:
                frequency = kwargs.pop('frequency')
            except KeyError:
                warnings.warn('If keyword argument frequency is not provided, it is set to 1.')
                frequency = 1
            t_gen_sig = 2 * np.pi * frequency * t
        else:
            if 'frequency' in kwargs.keys():
                raise NotImplementedError
            t_gen_sig = t

        # pop non scipy.signal.waveform.__all__ kwargs
        try:
            phase_shift = kwargs.pop('phase_shift')
        except KeyError as e:
            phase_shift = 0
        u = self._signal(t_gen_sig - phase_shift, **kwargs) * scale + offset
        InterpolationTrajectory.__init__(self, t, u)
