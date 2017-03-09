import numpy as np
import pyinduct as pi
import pyinduct.parabolic as parabolic


n_fem = 17
T = 1
l = 1
param = [1, 0, 0, None, None]
# or try these:
# param = [1, -0.5, -8, None, None]     :)))
a2, a1, a0, _, _ = param

temp_domain = pi.Domain(bounds=(0, T), num=100)
spat_domain = pi.Domain(bounds=(0, l), num=n_fem * 11)

# initial and test functions
nodes, fem_base = pi.cure_interval(pi.LagrangeFirstOrder,
                                   spat_domain.bounds,
                                   node_count=n_fem)
act_fem_base = pi.Base(fem_base[-1])
not_act_fem_base = pi.Base(fem_base[1:-1])
vis_fems_base = pi.Base(fem_base[1:])

pi.register_base("act_base", act_fem_base)
pi.register_base("sim_base", not_act_fem_base)
pi.register_base("vis_base", vis_fems_base)

# trajectory
u = parabolic.trajectory.RadTrajectory(l, T, param, "dirichlet", "dirichlet")

# weak form ...
x = pi.FieldVariable("sim_base")
phi = pi.TestFunction("sim_base")
act_phi = pi.ScalarFunction("act_base")

unmotivated_weak_form = pi.WeakFormulation([
    # ... of the homogeneous part of the system
    pi.IntegralTerm(pi.Product(x.derive(temp_order=1), phi),
                    limits=spat_domain.bounds),
    pi.IntegralTerm(pi.Product(x.derive(spat_order=1), phi.derive(1)),
                    limits=spat_domain.bounds,
                    scale=a2),
    pi.IntegralTerm(pi.Product(x.derive(spat_order=1), phi),
                    limits=spat_domain.bounds,
                    scale=-a1),
    pi.IntegralTerm(pi.Product(x, phi), limits=spat_domain.bounds, scale=-a0),

    # ... of the inhomogeneous part of the system
    pi.IntegralTerm(pi.Product(pi.Product(act_phi, phi), pi.Input(u, order=1)),
                    limits=spat_domain.bounds),
    pi.IntegralTerm(pi.Product(pi.Product(act_phi.derive(1), phi.derive(1)),
                               pi.Input(u)), limits=spat_domain.bounds,
                    scale=a2),
    pi.IntegralTerm(pi.Product(pi.Product(act_phi.derive(1), phi),
                               pi.Input(u)),
                    limits=spat_domain.bounds,
                    scale=-a1),
    pi.IntegralTerm(pi.Product(pi.Product(act_phi, phi), pi.Input(u)),
                    limits=spat_domain.bounds,
                    scale=-a0)],
    name="main_system")

# system matrices \dot x = A x + b0 u + b1 \dot u
cf = pi.parse_weak_formulation(unmotivated_weak_form)
ss = pi.create_state_space(cf)

a_mat = ss.A[1]
b0 = ss.B[0][1]
b1 = ss.B[1][1]

# transformation into \dot \bar x = \bar A \bar x + \bar b u
a_tilde = np.diag(np.ones(a_mat.shape[0]), 0)
a_tilde_inv = np.linalg.inv(a_tilde)

a_bar = (a_tilde @ a_mat) @ a_tilde_inv
b_bar = a_tilde @ (a_mat @ b1) + b0

# simulation
start_func = pi.Function.from_constant(0)
start_state = np.array([pi.project_on_base(start_func,
                                           pi.get_base("sim_base")
                                           )]).flatten()

start_state_bar = a_tilde @ start_state - (b1 * u(time=0)).flatten()
ss = pi.StateSpace(a_bar, b_bar, base_lbl="sim", input_handle=u)
sim_temp_domain, sim_weights_bar = pi.simulate_state_space(ss,
                                                           start_state_bar,
                                                           temp_domain)

# back-transformation
u_vec = np.reshape(u.get_results(sim_temp_domain), (len(temp_domain), 1))
sim_weights = sim_weights_bar @ a_tilde_inv + u_vec @ b1.T

# visualisation
plots = list()
save_pics = False
vis_weights = np.hstack((sim_weights, u_vec))

eval_d = pi.evaluate_approximation("vis_base",
                                   vis_weights,
                                   sim_temp_domain,
                                   spat_domain,
                                   spat_order=0)
der_eval_d = pi.evaluate_approximation("vis_base",
                                       vis_weights,
                                       sim_temp_domain,
                                       spat_domain,
                                       spat_order=1)
plots.append(pi.PgAnimatedPlot(eval_d,
                         labels=dict(left='x(z,t)', bottom='z'),
                         save_pics=save_pics))
plots.append(pi.PgAnimatedPlot(der_eval_d,
                         labels=dict(left='x\'(z,t)', bottom='z'),
                         save_pics=save_pics))
win1 = pi.PgSurfacePlot(eval_d, title="x(z,t)")
win2 = pi.PgSurfacePlot(der_eval_d, title="x'(z,t)")
# save pics
if save_pics:
    path = pi.save_2d_pg_plot(u.get_plot(), 'rad_dirichlet_traj')[1]
    win1.gl_widget.grabFrameBuffer().save(path + 'rad_dirichlet_3d_x.png')
    win2.gl_widget.grabFrameBuffer().save(path + 'rad_dirichlet_3d_dx.png')
pi.show()

pi.tear_down(("act_base", "sim_base", "vis_base"), plots + [win1, win2])
