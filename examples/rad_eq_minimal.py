from pyinduct import registry as re
from pyinduct import core as cr
from pyinduct import control as ct
from pyinduct import visualization as vis
from pyinduct import placeholder as ph
from pyinduct import trajectory as tr
from pyinduct import eigenfunctions as ef
from pyinduct import simulation as sim
import numpy as np
import pyqtgraph as pg
import matplotlib.pyplot as plt

# PARAMETERS TO VARY
# number of eigenfunctions, used for control law approximation
n_modal = 10
# control law parameter, stabilizing: param_a0_t < 0, destabilizing: param_a0_t > 0
param_a0 = 7
# initial profile x(z,0) (desired x(z,0)=0)
init_profile = 1

# original system parameters
a2 = 1
a1 = 3
a0 = param_a0
param = [a2, a1, a0, None, None]

# target system parameters (controller parameters)
a1_t = -3
a0_t = -6
param_t = [a2, a1_t, a0_t, None, None]

# system/simulation parameters
actuation_type = 'dirichlet'
bound_cond_type = 'dirichlet'
l = 1
T = 1
spatial_domain = sim.Domain(bounds=(0, l), num=30)
temporal_domain = sim.Domain(bounds=(0, T), num=1e2)
n = n_modal

# eigenvalues /-functions original system
scale = np.ones(n) * np.sqrt(2)
eig_values, eig_funcs = ef.SecondOrderDirichletEigenfunction.solve_evp_hint(param, l, n=n, scale=scale)
re.register_base("eig_funcs", eig_funcs)

# eigenfunctions target system
eig_freq = ef.SecondOrderDirichletEigenfunction.eigval_tf_eigfreq(param, eig_val=eig_values)
eig_freq_t = ef.SecondOrderDirichletEigenfunction.eigval_tf_eigfreq(param_t, eig_val=eig_values)
scale_t = scale * eig_freq / eig_freq_t
_, eig_funcs_t = ef.SecondOrderDirichletEigenfunction.solve_evp_hint(param_t, l, eig_freq=eig_freq_t, scale=scale_t)
re.register_base("eig_funcs_t", eig_funcs_t)

# init controller
x_at_1 = ph.FieldVariable("eig_funcs", location=1)
xt_at_1 = ph.FieldVariable("eig_funcs_t", weight_label="eig_funcs", location=1)
controller = ct.Controller(ct.ControlLaw([ph.ScalarTerm(x_at_1, 1), ph.ScalarTerm(xt_at_1, -1)]))

# derive initial field variable x(z,0) and weights
start_state = cr.Function(lambda z: init_profile)
initial_weights = cr.project_on_base(start_state, eig_funcs)

# init trajectory
traj = tr.RadTrajectory(l, T, param_t, bound_cond_type, actuation_type)

# input with feedback
control_law = sim.SimulationInputSum([traj, controller])

# determine (A,B) with modal-transfomation
A = np.diag(eig_values)
B = -a2 * np.array([eig_funcs[i].derive()(l) for i in range(n)])
ss = sim.StateSpace("eig_funcs", A, B, input_handle=control_law)

# evaluate desired output data
z_d = np.linspace(0, l, len(spatial_domain))
y_d, t_d = tr.gevrey_tanh(T, 80)
C = tr.coefficient_recursion(np.zeros(y_d.shape), y_d, param)
x_l = tr.power_series(z_d, t_d, C)
evald_traj = vis.EvalData([t_d, z_d], x_l, name="x(z,t) desired")

# simulate
t, q = sim.simulate_state_space(ss, initial_weights, temporal_domain)

# pyqtgraph visualization
evald_x = sim.evaluate_approximation("eig_funcs", q, t, spatial_domain, name="x(z,t) with x(z,0)=" + str(init_profile))
win1 = vis.PgAnimatedPlot([evald_x, evald_traj], title="animation")
win2 = vis.PgSurfacePlot([evald_x], title=evald_x.name, grid_height=1)
win3 = vis.PgSurfacePlot([evald_traj], title=evald_traj.name, grid_height=1)
pg.QtGui.QApplication.instance().exec_()

# visualization
vis.MplSlicePlot([evald_x, evald_traj], time_point=1, legend_label=["$x(z,1)$", "$x_d(z,1)$"], legend_location=2)
plt.show()
