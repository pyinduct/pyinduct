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


def build_weak_form(label, spatial_interval, u, params):
    p1 = ph.Product(ph.TemporalDerivedFieldVariable(label, order=2),
                    ph.TestFunction(label))
    t1 = ph.IntegralTerm(p1, limits=spatial_interval)
    p2 = ph.Product(ph.SpatialDerivedFieldVariable(label, 1),
                    ph.TestFunction(label, order=1))
    t2 = ph.IntegralTerm(p2, limits=spatial_interval)
    p3 = ph.Product(ph.TemporalDerivedFieldVariable(label, order=2, location=0),
                    ph.TestFunction(label, location=0))
    t3 = ph.ScalarTerm(p3, scale=params.m)
    p4 = ph.Product(ph.Input(u), ph.TestFunction(label, location=1))
    t4 = ph.ScalarTerm(p4, scale=-1)

    return sim.WeakFormulation([t1, t2, t3, t4], name="swm_fem")


def build_control_law(sys_label, params):
    dz_x1 = ph.SpatialDerivedFieldVariable(sys_label, order=1)
    x2 = ph.TemporalDerivedFieldVariable(sys_label, order=1)
    xi1 = ph.FieldVariable(sys_label, location=0)
    xi2 = ph.TemporalDerivedFieldVariable(sys_label, order=1, location=0)

    x2_at1 = ph.TemporalDerivedFieldVariable(sys_label, order=1, location=1)

    scalar_scale_funcs = [cr.Function(lambda theta: params.m * (1 - np.exp(-theta / params.m))),
                          cr.Function(lambda theta: params.m * (-1 + np.exp(theta / params.m))),
                          cr.Function(lambda theta: np.exp(-theta / params.m)),
                          cr.Function(lambda theta: np.exp(theta / params.m))]

    register_base("int_scale_func1", cr.Function(lambda tau: 1 - np.exp(-(1 - tau) / params.m)))
    register_base("int_scale_func2", cr.Function(lambda tau: -1 + np.exp((-1 + tau) / params.m)))
    register_base("int_scale_func3", cr.Function(lambda tau: np.exp(-(1 - tau) / params.m) / params.m))
    register_base("int_scale_func4", cr.Function(lambda tau: np.exp((-1 + tau) / params.m) / params.m))

    limits = (0, 1)
    y_bar_plus1 = [ph.ScalarTerm(xi1),
                   ph.ScalarTerm(xi2, scale=scalar_scale_funcs[0](1)),
                   ph.IntegralTerm(ph.Product(ph.ScalarFunction("int_scale_func1"), dz_x1), limits=limits),
                   ph.IntegralTerm(ph.Product(ph.ScalarFunction("int_scale_func1"), x2), limits=limits)]
    y_bar_minus1 = [ph.ScalarTerm(xi1),
                    ph.ScalarTerm(xi2, scale=scalar_scale_funcs[1](-1)),
                    ph.IntegralTerm(ph.Product(ph.ScalarFunction("int_scale_func2"), dz_x1), limits=limits, scale=-1),
                    ph.IntegralTerm(ph.Product(ph.ScalarFunction("int_scale_func2"), x2), limits=limits)]
    dz_y_bar_plus1 = [ph.ScalarTerm(xi2, scale=scalar_scale_funcs[2](1)),
                      ph.IntegralTerm(ph.Product(ph.ScalarFunction("int_scale_func3"), dz_x1), limits=limits),
                      ph.IntegralTerm(ph.Product(ph.ScalarFunction("int_scale_func3"), x2), limits=limits)]
    dz_y_bar_minus1 = [ph.ScalarTerm(xi2, scale=scalar_scale_funcs[3](-1)),
                       ph.IntegralTerm(ph.Product(ph.ScalarFunction("int_scale_func4"), dz_x1), limits=limits, scale=-1),
                       ph.IntegralTerm(ph.Product(ph.ScalarFunction("int_scale_func4"), x2), limits=limits)]

    return ct.ControlLaw(
        [ph.ScalarTerm(x2_at1, scale=-(1 - params.alpha) / (1 + params.alpha))] +
        ut.scale_equation_term_list(dz_y_bar_plus1,
                                    factor=(1 - params.m * params.k1) / (1 + params.alpha)) +
        ut.scale_equation_term_list(dz_y_bar_minus1,
                                    factor=-params.alpha * (1 + params.m * params.k1) / (1 + params.alpha)) +
        ut.scale_equation_term_list(y_bar_plus1,
                                    factor=-params.m * params.k0 / (1 + params.alpha)) +
        ut.scale_equation_term_list(y_bar_minus1,
                                    factor=-params.alpha * params.m * params.k0 / (1 + params.alpha))
    )


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


def build_observer():
    pass


class Parameters:
    def __init__(self):
        pass


app = QtGui.QApplication([])

# temporal and spatial domain specification
t_start = 0
t_end = 10
t_step = .01
temp_domain = sim.Domain(bounds=(t_start, t_end), step=t_step)
z_start = 0
z_end = 1
z_step = .01
spat_domain = sim.Domain(bounds=(z_start, z_end), step=z_step)

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

nodes, ini_funcs = sh.cure_interval(sh.LagrangeFirstOrder,
                                    spat_domain.bounds,
                                    node_count=params.node_count)
register_base("sim", ini_funcs)

# system input
if 1:
    # trajectory for the new input (closed_loop_traj)
    smooth_transition = tr.SmoothTransition((0, 1), (1, 3), method="poly", differential_order=2)
    closed_loop_traj = SecondOrderFeedForward(smooth_transition, params)
    # controller
    ctrl = ct.Controller(build_control_law("sim", params))
    u = sim.SimulationInputSum([closed_loop_traj, ctrl])
else:
    # trajectory for the original input (open_loop_traj)
    open_loop_traj = tr.FlatString(y0=x_zt(0, 0), y1=1, z0=z_start, z1=z_end, t0=1, dt=3, params=params)
    u = sim.SimulationInputSum([open_loop_traj])

# weak formulation
sys = build_weak_form("sim", spat_domain.bounds, u, params)

# simulation
results = sim.simulate_system(sys, ic, temp_domain, spat_domain)

# animation
plot1 = vis.PgAnimatedPlot(results)
plot2 = vis.PgSurfacePlot(results)
app.exec_()
