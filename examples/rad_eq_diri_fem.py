import pyinduct.trajectory as tr
import pyinduct.core as cr
import pyinduct.shapefunctions as sh
import pyinduct.simulation as sim
import pyinduct.visualization as vis
import pyinduct.placeholder as ph
from pyinduct import register_base, get_base

import numpy as np
import pyqtgraph as pg

n_fem = 17
T = 1
l = 1
param = [1, 0, 0, None, None]  # or try this: param = [1, -0.5, -8, None, None]     :)))
a2, a1, a0, _, _ = param

temp_domain = sim.Domain(bounds=(0, T), num=1e2)
spat_domain = sim.Domain(bounds=(0, l), num=n_fem * 11)

# initial and test functions
nodes, fem_funcs = sh.cure_interval(sh.LagrangeFirstOrder, spat_domain.bounds, node_count=n_fem)
act_fem_func = fem_funcs[-1]
not_act_fem_funcs = fem_funcs[1:-1]
vis_fems_funcs = fem_funcs[1:]
register_base("sim", not_act_fem_funcs)
register_base("vis", vis_fems_funcs)

# trajectory
u = tr.RadTrajectory(l, T, param, "dirichlet", "dirichlet")

# weak form of the homogeneous system
not_acuated_weak_form = sim.WeakFormulation([
    ph.IntegralTerm(ph.Product(ph.TemporalDerivedFieldVariable("sim", order=1),
                               ph.TestFunction("sim", order=0)),
                    limits=spat_domain.bounds),
    ph.IntegralTerm(ph.Product(ph.SpatialDerivedFieldVariable("sim", order=1),
                               ph.TestFunction("sim", order=1)),
                    limits=spat_domain.bounds,
                    scale=a2),
    ph.IntegralTerm(ph.Product(ph.SpatialDerivedFieldVariable("sim", order=1),
                               ph.TestFunction("sim", order=0)),
                    limits=spat_domain.bounds,
                    scale=-a1),
    ph.IntegralTerm(ph.Product(ph.SpatialDerivedFieldVariable("sim", order=0),
                               ph.TestFunction("sim", order=0)),
                    limits=spat_domain.bounds,
                    scale=-a0),
])

# inhomogeneous part of the system
cf = sim.parse_weak_formulation(not_acuated_weak_form)
E1 = cf._matrices["E"][1][1]
A = cf.convert_to_state_space().A[1]
A_tilde = sim.calculate_scalar_product_matrix(sim.dot_product_l2, not_act_fem_funcs, not_act_fem_funcs)
b0_sum = - a2 * np.array(
    [[sim.dot_product_l2(act_fem_func.derive(1), fem_func.derive(1))] for fem_func in not_act_fem_funcs]) \
         + a1 * np.array(
    [[sim.dot_product_l2(act_fem_func.derive(1), fem_func.derive(0))] for fem_func in not_act_fem_funcs]) \
         + a0 * np.array(
    [[sim.dot_product_l2(act_fem_func.derive(0), fem_func.derive(0))] for fem_func in not_act_fem_funcs])
b0 = np.dot(np.linalg.inv(E1), b0_sum)
b1_sum = - np.array([[sim.dot_product_l2(act_fem_func, fem_func)] for fem_func in not_act_fem_funcs])
b1 = np.dot(np.linalg.inv(E1), b1_sum)

# transformation
A_bar = np.dot(np.dot(A_tilde, A), np.linalg.inv(A_tilde))
b_bar = np.dot(A_tilde, np.dot(A, b1) + b0)

# simulation
start_func = cr.Function(lambda z: 0)
start_state = np.array([sim.project_on_base(start_func, get_base(cf.weights, 0))]).flatten()
transf_start_state = np.dot(A_tilde, start_state) - (b1 * u(time=0)).flatten()
ss = sim.StateSpace("transf_sim", A_bar, b_bar, input_handle=u)
sim_temp_domain, sim_transf_weights = sim.simulate_state_space(ss, transf_start_state, temp_domain)

# back-transformation
u_vec = np.matrix(u.get_results(sim_temp_domain)).T
sim_weights = np.nan * np.zeros((sim_transf_weights.shape[0], len(not_act_fem_funcs)))
for i in range(sim_transf_weights.shape[0]):
    sim_weights[i, :] = np.dot(np.linalg.inv(A_tilde), sim_transf_weights[i, :]) + (b1 * u_vec[i]).flatten()

# visualisation
save_pics = False
vis_weights = np.hstack((np.matrix(sim_weights), u_vec))
eval_d = sim.evaluate_approximation("vis", vis_weights, sim_temp_domain, spat_domain, spat_order=0)
der_eval_d = sim.evaluate_approximation("vis", vis_weights, sim_temp_domain, spat_domain, spat_order=1)
win1 = vis.PgAnimatedPlot(eval_d, labels=dict(left='x(z,t)', bottom='z'), save_pics=save_pics)
win2 = vis.PgAnimatedPlot(der_eval_d, labels=dict(left='x\'(z,t)', bottom='z'), save_pics=save_pics)
win3 = vis.PgSurfacePlot(eval_d, title="x(z,t)")
win4 = vis.PgSurfacePlot(der_eval_d, title="x'(z,t)")

# save pics
if save_pics:
    path = vis.save_2d_pg_plot(u.get_plot(), 'rad_dirichlet_traj')[1]
    win3.gl_widget.grabFrameBuffer().save(path + 'rad_dirichlet_3d_x.png')
    win4.gl_widget.grabFrameBuffer().save(path + 'rad_dirichlet_3d_dx.png')
pg.QtGui.QApplication.instance().exec_()
