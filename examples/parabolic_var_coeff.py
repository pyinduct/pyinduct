
import numpy as np
import pyqtgraph as pg
import scipy.integrate as si
import matplotlib.pyplot as plt

import simulation
from pyinduct import register_base
from pyinduct import core as cr
from pyinduct import placeholder as ph
from pyinduct import utils as ut
from pyinduct import shapefunctions as sh
from pyinduct import trajectory as tr
from pyinduct import eigenfunctions as ef
from pyinduct import simulation as sim
from pyinduct import visualization as vis

# system/simulation parameters
actuation_type = 'robin'
bound_cond_type = 'robin'
l = 1.
T = 1
spatial_domain = sim.Domain(bounds=(0, l), num=30)
temporal_domain = sim.Domain(bounds=(0, T), num=1e2)
n = 10

# original system parameters
a2 = .5
a1_z = cr.Function(lambda z: 0.1 * np.exp(4 * z),
                   derivative_handles=[lambda z: 0.4 * np.exp(4 * z)])
a0_z = lambda z: 1 + 10 * z + 2 * np.sin(4 * np.pi / l * z)
alpha = -1
beta = -1
param = [a2, a1_z, a0_z, alpha, beta]

# target system parameters (controller parameters)
a1_t = -0
a0_t = -6
alpha_t = 3
beta_t = 3
param_t = [a2, a1_t, a0_t, alpha_t, beta_t]
adjoint_param_t = ef.get_adjoint_rad_evp_param(param_t)

# original intermediate ("_i") and target intermediate ("_ti") system parameters
_, _, a0_i, alpha_i, beta_i = ef.transform2intermediate(param, d_end=l)
param_i = a2, 0, a0_i, alpha_i, beta_i
_, _, a0_ti, alpha_ti, beta_ti = ef.transform2intermediate(param_t)
param_ti = a2, 0, a0_ti, alpha_ti, beta_ti

# create (not normalized) target (_t) eigenfunctions
eig_freq_t, eig_val_t = ef.compute_rad_robin_eigenfrequencies(param_t, l, n)
init_eig_funcs_t = np.array([ef.SecondOrderRobinEigenfunction(om, param_t, spatial_domain.bounds) for om in eig_freq_t])
init_adjoint_eig_funcs_t = np.array(
    [ef.SecondOrderRobinEigenfunction(om, adjoint_param_t, spatial_domain.bounds) for om in eig_freq_t])

# normalize eigenfunctions and adjoint eigenfunctions
adjoint_and_eig_funcs_t = [cr.normalize_function(init_eig_funcs_t[i], init_adjoint_eig_funcs_t[i]) for i in range(n)]
eig_funcs_t = np.array([f_tuple[0] for f_tuple in adjoint_and_eig_funcs_t])
adjoint_eig_funcs_t = np.array([f_tuple[1] for f_tuple in adjoint_and_eig_funcs_t])

# transformed original eigenfunctions
eig_funcs = np.array([ef.TransformedSecondOrderEigenfunction(eig_val_t[i],
                                                             [eig_funcs_t[i](0), alpha * eig_funcs_t[i](0), 0, 0],
                                                             [a2, a1_z, a0_z],
                                                             np.linspace(0, l, 1e4))
                      for i in range(n)])

# create testfunctions
nodes, fem_funcs = sh.cure_interval(sh.LagrangeFirstOrder,
                                    spatial_domain.bounds,
                                    node_count=len(spatial_domain))

# register functions
register_base("eig_funcs_t", eig_funcs_t, overwrite=True)
register_base("adjoint_eig_funcs_t", adjoint_eig_funcs_t, overwrite=True)
register_base("eig_funcs", eig_funcs, overwrite=True)
register_base("fem_funcs", fem_funcs, overwrite=True)

# init trajectory
traj = tr.RadTrajectory(l, T, param_ti, bound_cond_type, actuation_type)

# original () and target (_t) field variable
fem_field_variable = ph.FieldVariable("fem_funcs", location=l)
field_variable_t = ph.FieldVariable("eig_funcs_t", weight_label="eig_funcs", location=l)
d_field_variable_t = ph.SpatialDerivedFieldVariable("eig_funcs_t", 1, weight_label="eig_funcs", location=l)
field_variable = ph.FieldVariable("eig_funcs", location=l)
d_field_variable = ph.SpatialDerivedFieldVariable("eig_funcs", 1, location=l)

# intermediate (_i) and target intermediate (_ti) transformations by z=l
transform_i_at_l = np.exp(si.quad(lambda z: a1_z(z) / 2 / a2, 0, l)[0])  # x_i  = x   * transform_i_at_l
inv_transform_i_at_l = np.exp(-si.quad(lambda z: a1_z(z) / 2 / a2, 0, l)[0])  # x  = x_i   * inv_transform_i_at_l
transform_ti_at_l = np.exp(a1_t / 2 / a2 * l)  # x_ti = x_t * transform_ti_at_l

# intermediate (_i) and target intermediate (_ti) field variable (list of scalar terms = sum of scalar terms)
x_fem_i_at_l = [ph.ScalarTerm(fem_field_variable, transform_i_at_l)]
x_i_at_l = [ph.ScalarTerm(field_variable, transform_i_at_l)]
xd_i_at_l = [ph.ScalarTerm(d_field_variable, transform_i_at_l),
             ph.ScalarTerm(field_variable, transform_i_at_l * a1_z(l) / 2 / a2)]
x_ti_at_l = [ph.ScalarTerm(field_variable_t, transform_ti_at_l)]
xd_ti_at_l = [ph.ScalarTerm(d_field_variable_t, transform_ti_at_l),
              ph.ScalarTerm(field_variable_t, transform_ti_at_l * a1_t / 2 / a2)]

# discontinuous operator (Kx)(t) = int_kernel_zz(l)*x(l,t)
int_kernel_zz = alpha_ti - alpha_i + si.quad(lambda z: (a0_i(z) - a0_ti) / 2 / a2, 0, l)[0]

# init controller
controller = ut.get_parabolic_robin_backstepping_controller(state=x_fem_i_at_l,
                                                            approx_state=x_i_at_l,
                                                            d_approx_state=xd_i_at_l,
                                                            approx_target_state=x_ti_at_l,
                                                            d_approx_target_state=xd_ti_at_l,
                                                            integral_kernel_zz=int_kernel_zz,
                                                            original_beta=beta_i,
                                                            target_beta=beta_ti,
                                                            trajectory=traj,
                                                            scale=inv_transform_i_at_l)

rad_pde = ut.get_parabolic_robin_weak_form("fem_funcs", "fem_funcs", controller, param, spatial_domain.bounds)
cf = sim.parse_weak_formulation(rad_pde)
ss_weak = cf.convert_to_state_space()

# simulate
t, q = sim.simulate_state_space(ss_weak, np.zeros((len(fem_funcs))), temporal_domain)

# pyqtgraph visualization
evald_x = simulation.evaluate_approximation("fem_funcs", q, t, spatial_domain, name="x(z,t)")
win1 = vis.PgAnimatedPlot([evald_x], title="animation", dt=T / temporal_domain.step*4)
win2 = vis.PgSurfacePlot(evald_x, title=evald_x.name, grid_height=1)
pg.QtGui.QApplication.instance().exec_()

# visualization
vis.MplSlicePlot([evald_x], time_point=1, legend_label=["$x(z,1)$"], legend_location=1)
vis.MplSurfacePlot(evald_x)
plt.show()
