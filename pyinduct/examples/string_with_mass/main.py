from pyinduct.examples.string_with_mass.utils import *
from pyinduct.examples.string_with_mass.system import *
import pyinduct as pi


# domains
z_end = 1
spatial_discretization = 100
spatial_domain = pi.Domain((0, z_end), spatial_discretization)
t_end = 10
temporal_discretization = 300
temporal_domain = pi.Domain((0, t_end), temporal_discretization)

# controller
input_ = pi.ConstantTrajectory(0)

# system approximation
sys_lbl = "string_with_mass"
fem_funcs1_nodes = pi.Domain(spatial_domain.bounds, 4)
fem_funcs2_nodes = pi.Domain(spatial_domain.bounds, 3)
build_fem_bases(sys_lbl, fem_funcs1_nodes, fem_funcs2_nodes)
sys_wf = build_weak_formulation(sys_lbl, spatial_domain, input_, sys_lbl)

# simulation
init_cond = {sys_wf.name: [SwmBaseFraction(
    [pi.Function.from_constant(0), pi.Function.from_constant(0)], [0, 1])]}
spatial_domains = {sys_wf.name: spatial_domain}
ceq, ss, init_weights, weights, evald = pi.simulate_systems(
    [sys_wf], init_cond, temporal_domain, spatial_domains,
)

# check eigenvalues of the approximation
pprint()
pprint("Eigenvalues of the approximation:")
pprint(sort_eigenvalues(np.linalg.eigvals(ss.A[1])))

# visualization
eval_data1 = pi.get_sim_result(sys_lbl + "_1_visu", weights, temporal_domain, spatial_domain, 0, 0, name="x1(z,t)")
eval_data2 = pi.get_sim_result(sys_lbl + "_2_visu", weights, temporal_domain, spatial_domain, 0, 0, name="x2(z,t)")
eval_data3 = pi.get_sim_result(sys_lbl + "_3_visu", weights, temporal_domain, spatial_domain, 0, 0, name="xi1(t)")
eval_data4 = pi.get_sim_result(sys_lbl + "_4_visu", weights, temporal_domain, spatial_domain, 0, 0, name="xi2(t)")
plots = list()
plots.append(pi.PgAnimatedPlot(eval_data2 + eval_data4))
plots.append(pi.PgAnimatedPlot(eval_data1 + eval_data3))
pi.show()
