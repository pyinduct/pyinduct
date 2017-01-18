import core
import hyperbolic.feedforward
import parabolic.control
import pyinduct.trajectory as tr
import pyinduct.core as cr
import pyinduct.shapefunctions as sh
import pyinduct.simulation as sim
import pyinduct.visualization as vis
import pyinduct.placeholder as ph
import pyinduct.utils as ut
import pyinduct.control as ct
from pyinduct import register_base
import numpy as np
from pyqtgraph.Qt import QtGui


def build_weak_form(approx_label, spatial_interval, u, params):
    x = ph.FieldVariable(approx_label)
    psi = ph.TestFunction(approx_label)
    term1 = ph.IntegralTerm(ph.Product(x.derive(temp_order=2), psi), limits=spatial_interval)
    term2 = ph.IntegralTerm(ph.Product(x.derive(spat_order=1), psi.derive(1)), limits=spatial_interval)
    term3 = ph.ScalarTerm(ph.Product(x(0).derive(temp_order=2), psi(0)), scale=params.m)
    term4 = ph.ScalarTerm(ph.Product(ph.Input(u), psi(1)), scale=-1)

    return sim.WeakFormulation([term1, term2, term3, term4], name="swm_system")


def build_control_law(approx_label, params):
    x = ph.FieldVariable(approx_label)
    dz_x1 = x.derive(spat_order=1)
    x2 = x.derive(temp_order=1)
    xi1 = x(0)
    xi2 = x(0).derive(temp_order=1)

    scalar_scale_funcs = [cr.Function(lambda theta: params.m * (1 - np.exp(-theta / params.m))),
                          cr.Function(lambda theta: params.m * (-1 + np.exp(theta / params.m))),
                          cr.Function(lambda theta: np.exp(-theta / params.m)),
                          cr.Function(lambda theta: np.exp(theta / params.m))]

    register_base("int_scale1", cr.Function(lambda tau: 1 - np.exp(-(1 - tau) / params.m)))
    register_base("int_scale2", cr.Function(lambda tau: -1 + np.exp((-1 + tau) / params.m)))
    register_base("int_scale3", cr.Function(lambda tau: np.exp(-(1 - tau) / params.m) / params.m))
    register_base("int_scale4", cr.Function(lambda tau: np.exp((-1 + tau) / params.m) / params.m))

    limits = (0, 1)
    y_bar_plus1 = [ph.ScalarTerm(xi1),
                   ph.ScalarTerm(xi2, scale=scalar_scale_funcs[0](1)),
                   ph.IntegralTerm(ph.Product(ph.ScalarFunction("int_scale1"), dz_x1), limits=limits),
                   ph.IntegralTerm(ph.Product(ph.ScalarFunction("int_scale1"), x2), limits=limits)]
    y_bar_minus1 = [ph.ScalarTerm(xi1),
                    ph.ScalarTerm(xi2, scale=scalar_scale_funcs[1](-1)),
                    ph.IntegralTerm(ph.Product(ph.ScalarFunction("int_scale2"), dz_x1), limits=limits, scale=-1),
                    ph.IntegralTerm(ph.Product(ph.ScalarFunction("int_scale2"), x2), limits=limits)]
    dz_y_bar_plus1 = [ph.ScalarTerm(xi2, scale=scalar_scale_funcs[2](1)),
                      ph.IntegralTerm(ph.Product(ph.ScalarFunction("int_scale3"), dz_x1), limits=limits),
                      ph.IntegralTerm(ph.Product(ph.ScalarFunction("int_scale3"), x2), limits=limits)]
    dz_y_bar_minus1 = [ph.ScalarTerm(xi2, scale=scalar_scale_funcs[3](-1)),
                       ph.IntegralTerm(ph.Product(ph.ScalarFunction("int_scale4"), dz_x1), limits=limits, scale=-1),
                       ph.IntegralTerm(ph.Product(ph.ScalarFunction("int_scale4"), x2), limits=limits)]

    return ct.ControlLaw(parabolic.control.scale_equation_term_list(
        [ph.ScalarTerm(x2(1), scale=-(1 - params.alpha))] +
        parabolic.control.scale_equation_term_list(dz_y_bar_plus1, factor=(1 - params.m * params.k1)) +
        parabolic.control.scale_equation_term_list(dz_y_bar_minus1, factor=-params.alpha * (1 + params.m * params.k1)) +
        parabolic.control.scale_equation_term_list(y_bar_plus1, factor=-params.m * params.k0) +
        parabolic.control.scale_equation_term_list(y_bar_minus1, factor=-params.alpha * params.m * params.k0),
        factor=(1 + params.alpha) ** -1
    ))


def build_observer():
    pass


class SecondOrderFeedForward(sim.SimulationInput):
    def __init__(self, desired_handle, params):
        sim.SimulationInput.__init__(self)
        self._y = desired_handle
        self._params = params

    def _calc_output(self, **kwargs):
        y_p = self._y(kwargs["time"] + 1)
        y_m = self._y(kwargs["time"] - 1)
        f = + self._params.k0 * (y_p[0] + self._params.alpha * y_m[0]) \
            + self._params.k1 * (y_p[1] + self._params.alpha * y_m[1]) \
            + y_p[2] + self._params.alpha * y_m[2]
        return dict(output=self._params.m / (1 + self._params.alpha) * f)


class Parameters:
    def __init__(self):
        pass


app = QtGui.QApplication([])

# temporal and spatial domain specification
t_start = 0
t_end = 10
t_step = .01
temp_domain = core.Domain(bounds=(t_start, t_end), step=t_step)
z_start = 0
z_end = 1
z_step = .01
spat_domain = core.Domain(bounds=(z_start, z_end), step=z_step)

# system/simulation parameters
params = Parameters
params.node_count = 10
params.m = 1.0
params.tau = 1.0  # hard written to 1 in this example script
params.sigma = 1.0  # hard written to 1 in this example script

# controller parameters
params.k0 = 10
params.k1 = 10
params.alpha = 0.7

# inital conditions
x_zt = lambda z, t: 0
dt_x_zt = lambda z, t: 0
ic = np.array([
    cr.Function(lambda z: x_zt(z, 0)),
    cr.Function(lambda z: dt_x_zt(z, 0)),
])

# initial function
sys_nodes, sys_funcs = sh.cure_interval(sh.LagrangeFirstOrder, spat_domain.bounds, node_count=10)
ctrl_nodes, ctrl_funcs = sh.cure_interval(sh.LagrangeFirstOrder, spat_domain.bounds, node_count=20)
obs_nodes, obs_funcs = sh.cure_interval(sh.LagrangeSecondOrder, spat_domain.bounds, node_count=7)
register_base("sim", sys_funcs)
register_base("ctrl", ctrl_funcs)
register_base("obs", obs_funcs)

# system input
if 1:
    # trajectory for the new input (closed_loop_traj)
    smooth_transition = tr.SmoothTransition((0, 1), (1, 3), method="poly", differential_order=2)
    closed_loop_traj = SecondOrderFeedForward(smooth_transition, params)
    # controller
    ctrl = ct.Controller(build_control_law("ctrl", params))
    u = sim.SimulationInputSum([closed_loop_traj, ctrl])
else:
    # trajectory for the original input (open_loop_traj)
    open_loop_traj = hyperbolic.feedforward.FlatString(y0=x_zt(0, 0), y1=1, z0=z_start, z1=z_end, t0=1, dt=3, params=params)
    u = sim.SimulationInputSum([open_loop_traj])

# weak formulation
sys = build_weak_form("sim", spat_domain.bounds, u, params)

# simulation
results = sim.simulate_system(sys, ic, temp_domain, spat_domain)

# animation
plot1 = vis.PgAnimatedPlot(results)
plot2 = vis.PgSurfacePlot(results)
app.exec_()
