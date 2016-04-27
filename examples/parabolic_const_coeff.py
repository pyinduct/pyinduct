import numpy as np
import pyqtgraph as pg
import matplotlib.pyplot as plt

from pyinduct import registry as re
from pyinduct import core as cr
from pyinduct import placeholder as ph
from pyinduct import utils as ut
from pyinduct import trajectory as tr
from pyinduct import eigenfunctions as ef
from pyinduct import simulation as sim
from pyinduct import visualization as vis
from pyinduct import shapefunctions as sh

# PARAMETERS TO VARY
# number of eigenfunctions, used for control law approximation
n_modal = 10
# number FEM test functions, used for system approximation/simulation
n_fem = 30
# control law parameter, stabilizing: param_a0_t < 0, destabilizing: param_a0_t > 0
param_a0_t = -6
# initial profile x(z,0) (desired x(z,0)=0)
init_profile = 0.2

# original system parameter
a2 = .5
a1 = 1
a0 = 6
alpha = -1
beta = -1
param = [a2, a1, a0, alpha, beta]
adjoint_param = ef.get_adjoint_rad_evp_param(param)

# target system parameters (controller parameters)
a1_t = 0
a0_t = param_a0_t
alpha_t = 3
beta_t = 3
# a1_t = a1 a0_t = a0 alpha_t = alpha beta_t = beta
param_t = [a2, a1_t, a0_t, alpha_t, beta_t]

# original intermediate ("_i") and target intermediate ("_ti") system parameters
_, _, a0_i, alpha_i, beta_i = ef.transform2intermediate(param)
param_i = a2, 0, a0_i, alpha_i, beta_i
_, _, a0_ti, alpha_ti, beta_ti = ef.transform2intermediate(param_t)
param_ti = a2, 0, a0_ti, alpha_ti, beta_ti

# system/simulation parameters
l = 1
T = 1
actuation_type = 'robin'
bound_cond_type = 'robin'
spatial_domain = sim.Domain(bounds=(0, l), num=n_fem)
temporal_domain = sim.Domain(bounds=(0, 1), num=1e2)
n = n_modal

# create (not normalized) eigenfunctions
eig_freq, eig_val = ef.compute_rad_robin_eigenfrequencies(param, l, n)
init_eig_funcs = np.array([ef.SecondOrderRobinEigenfunction(om, param, spatial_domain.bounds) for om in eig_freq])
init_adjoint_eig_funcs = np.array(
    [ef.SecondOrderRobinEigenfunction(om, adjoint_param, spatial_domain.bounds) for om in eig_freq])

# normalize eigenfunctions and adjoint eigenfunctions
adjoint_and_eig_funcs = [cr.normalize_function(init_eig_funcs[i], init_adjoint_eig_funcs[i]) for i in range(n)]
eig_funcs = np.array([f_tuple[0] for f_tuple in adjoint_and_eig_funcs])
adjoint_eig_funcs = np.array([f_tuple[1] for f_tuple in adjoint_and_eig_funcs])

# eigenfunctions from target system ("_t")
eig_freq_t = np.sqrt(-a1_t ** 2 / 4 / a2 ** 2 + (a0_t - eig_val) / a2)
eig_funcs_t = np.array(
    [ef.SecondOrderRobinEigenfunction(eig_freq_t[i], param_t, spatial_domain.bounds).scale(eig_funcs[i](0))
     for i in range(n)])

# create fem test functions
nodes, fem_funcs = sh.cure_interval(sh.LagrangeFirstOrder,
                                    spatial_domain.bounds,
                                    node_count=len(spatial_domain))

# register eigenfunctions
re.register_base("eig_funcs", eig_funcs, overwrite=True)
re.register_base("adjoint_eig_funcs", adjoint_eig_funcs, overwrite=True)
re.register_base("eig_funcs_t", eig_funcs_t, overwrite=True)
re.register_base("fem_funcs", fem_funcs, overwrite=True)

# original () and target (_t) field variable
fem_field_variable = ph.FieldVariable("fem_funcs", location=l)
field_variable = ph.FieldVariable("eig_funcs", location=l)
d_field_variable = ph.SpatialDerivedFieldVariable("eig_funcs", 1, location=l)
field_variable_t = ph.FieldVariable("eig_funcs_t", weight_label="eig_funcs", location=l)
d_field_variable_t = ph.SpatialDerivedFieldVariable("eig_funcs_t", 1, weight_label="eig_funcs", location=l)


def transform_i(z):
    """
    intermediate (_i) transformation at z=l
    """
    return np.exp(a1 / 2 / a2 * z)  # x_i  = x   * transform_i


def transform_ti(z):
    """
    target intermediate (_ti) transformation at z=l
    """
    return np.exp(a1_t / 2 / a2 * z)  # x_ti = x_t * transform_ti

# intermediate (_i) and target intermediate (_ti) field variable (list of scalar terms = sum of scalar terms)
x_fem_i_at_l = [ph.ScalarTerm(fem_field_variable, transform_i(l))]
x_i_at_l = [ph.ScalarTerm(field_variable, transform_i(l))]
xd_i_at_l = [ph.ScalarTerm(d_field_variable, transform_i(l)),
             ph.ScalarTerm(field_variable, transform_i(l) * a1 / 2 / a2)]
x_ti_at_l = [ph.ScalarTerm(field_variable_t, transform_ti(l))]
xd_ti_at_l = [ph.ScalarTerm(d_field_variable_t, transform_ti(l)),
              ph.ScalarTerm(field_variable_t, transform_ti(l) * a1_t / 2 / a2)]


# discontinuous operator (Kx)(t) = int_kernel_zz(l)*x(l,t)
def int_kernel_zz(z):
    return alpha_ti - alpha_i + (a0_i - a0_ti) / 2 / a2 * z

# init trajectory
traj = tr.RadTrajectory(l, T, param_ti, bound_cond_type, actuation_type)

# controller initialization
controller = ut.get_parabolic_robin_backstepping_controller(state=x_i_at_l,
                                                            approx_state=x_i_at_l,
                                                            d_approx_state=xd_i_at_l,
                                                            approx_target_state=x_ti_at_l,
                                                            d_approx_target_state=xd_ti_at_l,
                                                            integral_kernel_zz=int_kernel_zz(l),
                                                            original_beta=beta_i,
                                                            target_beta=beta_ti,
                                                            trajectory=traj,
                                                            scale=transform_i(-l))

# determine (A,B)
rad_pde = ut.get_parabolic_robin_weak_form("fem_funcs", "fem_funcs", controller, param, spatial_domain.bounds)
cf = sim.parse_weak_formulation(rad_pde)
ss_weak = cf.convert_to_state_space()
# simulate
t, q = sim.simulate_state_space(ss_weak, init_profile * np.ones(n_fem), temporal_domain)

# evaluate desired output data
y_d, t_d = tr.gevrey_tanh(T, 80)
C = tr.coefficient_recursion(y_d, alpha * y_d, param)
x_l = tr.power_series(np.array(spatial_domain), t_d, C)
evald_traj = vis.EvalData([t_d, spatial_domain], x_l, name="x(z,t) desired")

# pyqtgraph visualization
eval_d = sim.evaluate_approximation("fem_funcs", q, t, spatial_domain, name="x(z,t) with x(z,0)=" + str(init_profile))
win1 = vis.PgAnimatedPlot([eval_d, evald_traj], title="animation", dt=temporal_domain.step)
win2 = vis.PgSurfacePlot([eval_d], title=eval_d.name, grid_height=1)
win3 = vis.PgSurfacePlot([evald_traj], title=evald_traj.name, grid_height=1)
pg.QtGui.QApplication.instance().exec_()

# matplotlib visualization
vis.MplSlicePlot([evald_traj, eval_d], spatial_point=0, legend_label=["$x_d(0,t)$", "$x(0,t)$"])
plt.show()
