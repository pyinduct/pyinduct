from __future__ import division
import sympy as sp
import numpy as np
import pyqtgraph as pg

from simulation import SimulationInput
from numbers import Number
import utils as ut
import eigenfunctions as ef
import scipy.misc as sm

__author__ = 'stefan ecklebe'


class ConstantTrajectory(SimulationInput):
    """
    trivial trajectory generator for a constant value as simulation input signal
    """
    def __init__(self, const=0):
        SimulationInput.__init__(self)
        self._const = const

    def _calc_output(self, **kwargs):
        if isinstance(kwargs["time"], (list, np.ndarray)):
            return np.ones(len(kwargs["time"]))*self._const
        elif isinstance(kwargs["time"], Number):
            return self._const
        else:
            raise NotImplementedError


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
        gamma = differential_order  # + 1 # TODO check this against notes

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

    def __call__(self, time):
        return self._desired_values(time)

    def _desired_values(self, t):
        """
        Calculates desired trajectory and all derivatives for moment t

        :param t: time value for which trajectory and derivatives are needed
        :returns np.ndarray
        :math:`\\boldsymbol{y}_d = \\left(y_d, \\dot{y}_d, \\ddot{y}_d, \\dotsc, \\y_d^{(\\gamma)}\\right)`
        """
        y = np.zeros((len(self.dphi_num)))
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
        SimulationInput.__init__(self)

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
        self.control_input = np.vectorize(self._control_input, otypes=[np.float])
        self.system_state = np.vectorize(self._system_sate, otypes=[np.float])

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

    def _calc_output(self, **kwargs):
        """
        use time to calculate system input and return force
        :param t:
        :param q:
        :param kwargs:
        :return:
        """
        return self._control_input(kwargs["time"])

#TODO: kwarg: t_step
def gevrey_tanh(T, n, sigma, K):
    """
    Provide the flat output y(t) = phi(t), with the gevrey-order
    1+1/sigma, and the derivatives up to order n.
    :param t: [0, ... , t_end]  (numpy array)
    :param n: (integer)
    :param sigma: (float)
    :param K: (float)
    :return: np.array([[phi], ... ,[phi^(n)]])
    """

    t_init = t = np.linspace(0., T, int(0.5*10**(2+np.log10(T))))

    # pop
    t = np.delete(t, 0, 0)
    t = np.delete(t, -1, 0)

    # main
    tau = t/T

    a = dict()
    a[0] = K*(4*tau*(1-tau))**(1-sigma)/(2*(sigma-1))
    a[1] = (2*tau - 1)*(sigma-1)/(tau*(1-tau))*a[0]
    for k in xrange(2, n+2):
        a[k] = (tau*(1-tau))**-1 * ((sigma-2+k)*(2*tau-1)*a[k-1]+(k-1)*(2*sigma-4+k)*a[k-2])

    yy = dict()
    yy[0] = np.tanh(a[1])
    if n > 0:
        yy[1] = a[2]*(1-yy[0]**2)
    z = dict()
    z[0] = (1-yy[0]**2)
    for i in xrange(2, n+1):
        sum_yy = np.zeros(len(t))
        for k in xrange(i):
            if k == 0:
                sum_z = np.zeros(len(t))
                for j in xrange(i):
                    sum_z += -sm.comb(i-1, j)*yy[j]*yy[i-1-j]
                z[i-1] = sum_z
            sum_yy += sm.comb(i-1, k)*a[k+2]*z[i-1-k]
        yy[i] = sum_yy

    # push
    phi = np.nan*np.zeros((n+1, len(t)+2))
    for i in xrange(n+1):
        phi_temp = 0.5*yy[i]
        if i == 0:
            phi_temp += 0.5
            phi_temp = np.insert(phi_temp, 0, [0.], axis=0)
            phi[i, :] = np.append(phi_temp, [1.])
        else:
            phi_temp = np.insert(phi_temp, 0, [0.], axis=0)
            # attention divide by T^i
            phi[i, :] = np.append(phi_temp, [0.])/T**i

    return phi, t_init

def _power_series_flat_out(z, t, n, param, y, boundary_condition):
    """ Provide the power series approximation for x(z,t) and x'(z,t).
    :param z: [0, ..., l] (numpy array)
    :param t: [0, ... , t_end] (numpy array)
    :param n: series termination index (integer)
    :param param: [a2, a1, a0, alpha, beta] (list)
    :param y: flat output with derivation np.array([[y],...,[y^(n/2)]])
    :return: field variable x(z,t) and spatial derivation x'(z,t)
    """
    a2, a1, a0, alpha, beta = param
    shape = (len(t), len(z))
    x = np.nan*np.ones(shape)
    d_x = np.nan*np.ones(shape)

    # Actually power_series() is designed for robin boundary condition by z=0.
    # With the following modification it can also used for dirichlet boundary condition by z=0.
    if boundary_condition is 'robin':
        is_robin = 1.
    elif boundary_condition is 'dirichlet':
        alpha = 1.
        is_robin = 0.
    else:
        raise ValueError("Selected Boundary condition {0} not supported! Use 'robin' or 'dirichlet'".format(
            boundary_condition))

    #TODO: flip iteration order: z <--> t, result: one or two instead len(t) call's
    for i in xrange(len(t)):
        sum_x = np.zeros(len(z))
        for j in xrange(n):
            sum_b = np.zeros(len(z))
            for k in xrange(j+1):
                sum_b += sm.comb(j, k)*(-a0)**(j-k)*y[k, i]
            sum_x += (is_robin+alpha*z/(2.*j+1.))*z**(2*j)/sm.factorial(2*j)/a2**j*sum_b
        x[i, :] = sum_x

    for i in xrange(len(t)):
        sum_x = np.zeros(len(z))
        for j in xrange(n):
            sum_b = np.zeros(len(z))
            for k in xrange(j+2):
                sum_b += sm.comb(j+1, k)*(-a0)**(j-k+1)*y[k, i]
            if j == 0:
                sum_x += alpha*y[0, i]
            sum_x += (is_robin+alpha*z/(2.*(j+1)))*z**(2*j+1)/sm.factorial(2*j+1)/a2**(j+1)*sum_b
        d_x[i, :] = sum_x

    return x, d_x

def coefficient_recursion(c0, c1, param):
    """
    return to the recursion
    c_k = (c_{k-2}^{(1)} - a_1*c_{k-1} - a_0*c_{k-2}) / a_2
    with initial values
    c0 = np.array([c_0^{(0)}, ... , c_0^{(N)}])
    c1 = np.array([c_1^{(0)}, ... , c_1^{(N)}])
    as much as computable subsequent coefficients
    c2 = np.array([c_2^{(0)}, ... , c_2^{(N-1)}])
    c3 = np.array([c_3^{(0)}, ... , c_3^{(N-1)}])
    ....
    c_{2N-1} = np.array([c_{2N-1}^{(0)}])
    c_{2N} = np.array([c_{2N}^{(0)}])
    :param c0:
    :param c1:
    :param param:
    :return: C = {0: c0, 1: c1, ..., 2N-1: c_{2N-1}, 2N: c_{2N}}
    """
    if c0.shape != c1.shape:
        raise ValueError

    a2, a1, a0, _, _ = param
    N = c0.shape[0]
    C = dict()

    C[0] = c0
    C[1] = c1

    for i in range(2, 2*N):
        reduced_derivative_order = int(i/2.)
        C[i] = np.nan*np.zeros((N-reduced_derivative_order, c0.shape[1]))
        for j in range(N-reduced_derivative_order):
            C[i][j, :] = (C[i-2][j+1, :] - a1*C[i-1][j, :] - a0*C[i-2][j, :])/a2

    return C

def temporal_derived_power_series(z, C, up_to_order, series_termination_index, spatial_der_order=0):
    """
    compute the temporal derivatives
    q^{(n)}(z) = \sum_{k=0}^{series_termination_index} C[k][n,:] z^k / k!
    from n=0 to n=up_to_order
    :param z: scalar
    :param C:
    :param up_to_order:
    :param series_termination_index:
    :return: Q = np.array( [q^{(0)}, ... , q^{(up_to_order)}] )
    """

    if not isinstance(z, Number):
        raise TypeError
    if any([C[i].shape[0] - 1 < up_to_order for i in range(series_termination_index+1)]):
        raise ValueError

    len_t = C[0].shape[1]
    Q = np.nan*np.zeros((up_to_order+1, len_t))

    for i in range(up_to_order+1):
        sum_Q = np.zeros(len_t)
        for j in range(series_termination_index+1-spatial_der_order):
            sum_Q += C[j+spatial_der_order][i, :]*z**(j)/sm.factorial(j)
        Q[i, :] = sum_Q

    return Q

def power_series(z, t, C, spatial_der_order=0):

    if not all([isinstance(item, (Number, np.ndarray)) for item in [z, t]]):
        raise TypeError
    z = np.atleast_1d(z)
    t = np.atleast_1d(t)
    if not all([len(item.shape) == 1 for item in [z, t]]):
        raise ValueError

    x = np.nan*np.zeros((len(t), len(z)))
    for i in range(len(z)):
        sum_x = np.zeros(t.shape[0])
        for j in range(len(C)-spatial_der_order):
            sum_x += C[j+spatial_der_order][0, :]*z[i]**j/sm.factorial(j)
        x[:, i] = sum_x

    if any([dim == 1 for dim in x.shape]):
        x = x.flatten()

    return x

class InterpTrajectory(SimulationInput):

    def __init__(self, t, u, show_plot=False):
        SimulationInput.__init__(self)

        self._t = t
        self._T = t[-1]
        self._u = u
        self.scale = 1

        if show_plot:
            pw = pg.plot(title="InterpTrajectory")
            pw.plot(self._t, self.__call__(time=self._t))
            pw.plot([0, self._T], self.__call__(time=[0, self._T]), pen=None, symbolPen=pg.mkPen("g"))
            pg.QtGui.QApplication.instance().exec_()

    def _calc_output(self, **kwargs):
        """
        use time to calculate system input and return force
        :param kwargs:
        :return:
        """
        return np.interp(kwargs["time"], self._t, self._u)*self.scale


class RadTrajectory(InterpTrajectory):
    """
    Class that implements a flatness based control approach
    for the reaction-advection-diffusion equation
    d/dt x(z,t) = a2 x''(z,t) + a1 x'(z,t) + a0 x(z,t)

    with the
    boundary condition
    case 1: x(0,t)=0  (Dirichlet)
    --> a transition from x'(0,0)=0 to  x'(0,t_end)=1 is considered
    --> with x'(0,t) = y(t) where y(t) is the flat output
    case 2: x'(0,t) = alpha x(0,t)  (Robin)
    --> a transition from x(0,0)=0 to  x(0,t_end)=1 is considered
    --> with x(0,t) = y(t) where y(t) is the flat output

    and the
    actuation
    case 1: x(l,t)=u(t)  (Dirichlet)
    case 2: x'(l,t) = -beta x(l,t) + u(t)  (Robin).
    """
    #TODO: kwarg: t_step
    def __init__(self, l, T, param_original, boundary_condition, actuation, n=80, sigma=1.1, K=2., show_plot=False):

        cases = {'dirichlet', 'robin'}
        if boundary_condition not in cases:
            raise TypeError('Type of boundary condition by z=0 is not understood.')
        if actuation not in cases:
            raise TypeError('Type of actuation is not understood.')

        self._l = l
        self._T = T
        self._a1_original = param_original[1]
        self._param = ef.transform2intermediate(param_original)
        self._boundary_condition = boundary_condition
        self._actuation = actuation
        self._n = n
        self._sigma = sigma
        self._K = K

        self._z = np.array([self._l])
        y, t = gevrey_tanh(self._T, self._n+2, self._sigma, self._K)
        x, d_x = _power_series_flat_out(self._z, t, self._n, self._param, y, boundary_condition)

        a2, a1, a0, alpha, beta = self._param
        if self._actuation is 'dirichlet':
            u = x[:, -1]
        elif self._actuation is 'robin':
            u = d_x[:, -1] + beta*x[:, -1]
        # actually the algorithm consider the pde
        # d/dt x(z,t) = a_2 x''(z,t) + a_0 x(z,t)
        # with the following back transformation are also
        # pde's with advection term a_1 x'(z,t) considered
        u *= np.exp(-self._a1_original/2./a2*l)

        InterpTrajectory.__init__(self, t, u, show_plot=show_plot)

