import sympy as sp
import numpy as np
import pyinduct as pi
import pyinduct.symbolic as sy

# system parameters
params = [1, 0, 5, 0, 0]
a2, a1, a0, alpha, beta = params

# spatial approximation order
N_fem = 30

# modal controller approximation order
N_modal = 10

# temporal domain
T = 5
temp_dom = pi.Domain((0, T), num=100)

# spatial domain
l = 1
spat_bounds = (0, l)
spat_dom = pi.Domain(spat_bounds, num=100)

# variables
var_pool = sy.VariablePool("transport system")
t = var_pool.new_symbol("t", "time")
z = var_pool.new_symbol("z", "location")

# define approximation base and symbols
nodes = pi.Domain(spat_bounds, num=N_fem)
fem_base = pi.LagrangeFirstOrder.cure_interval(nodes)
pi.register_base("fem_base", fem_base)
shape_funcs = var_pool.new_implemented_functions(
    ["phi{}".format(i) for i in range(N_fem)], [(z,)] * N_fem,
    fem_base.fractions, "shape functions")
test_funcs = var_pool.new_implemented_functions(
    ["psi{}".format(i) for i in range(N_fem)], [(z,)] * N_fem,
    fem_base.fractions, "test functions")

# build fem approximation
weights = sp.Matrix(var_pool.new_functions(
    ["c{}".format(i) for i in range(N_fem)], [(t,)] * N_fem, "approximation weights"))
x_approx = sum([c * phi for c, phi in zip(weights, shape_funcs)])
sy.pprint(x_approx, "approximation", N_fem)

# build modal approximation
eig_vals, modal_base = pi.SecondOrderRobinEigenfunction.cure_interval(
    spat_dom, param=params, n=N_modal)
sy.pprint(eig_vals, "eigen values")
pi.register_base("primal_base", modal_base)
weights_modal = sp.Matrix(var_pool.new_functions(
    ["c_m{}".format(i) for i in range(N_modal)], [(t,)] * N_modal, "modal approximation weights"))
shape_funcs_modal = var_pool.new_implemented_functions(
    ["phi_m{}".format(i) for i in range(N_modal)], [(z,)] * N_modal,
    modal_base.fractions, "modal shape functions")
x_approx_modal = sum([c * phi for c, phi in zip(weights_modal, shape_funcs_modal)])
sy.pprint(x_approx_modal, "modal approximation", N_fem)


# control law
params_target = [1, 0, -15, 0, 0]
_, modal_target_base = pi.SecondOrderRobinEigenfunction.cure_interval(
    spat_dom, param=params_target, eig_val=eig_vals)
shape_funcs_modal_target = var_pool.new_implemented_functions(
    ["phi_mt{}".format(i) for i in range(N_modal)], [(z,)] * N_modal,
    modal_target_base.fractions, "modal target shape functions")
a2_t, a1_t, a0_t, alpha_t, beta_t = params_target
assert a2 == a2_t == 1
assert a1 == a1_t == 0
assert alpha == alpha_t == 0
assert beta == beta_t == 0
assert l == 1
a0_t_m_a0 = a0_t - a0
x = var_pool.new_symbol("x", "symbols")
y = var_pool.new_symbol("y", "symbols")
sqrt_xy = sp.sqrt(-a0_t_m_a0 * (x ** 2 - y ** 2))
kernel = -a0_t_m_a0 * x * sp.besseli(1, sqrt_xy) / sqrt_xy
kernel_ll = sp.limit(kernel.subs(sqrt_xy, z).subs(x, l), z, 0)
control_kernel = sp.diff(kernel, x).subs(x, l)
control_kernel_imp = sy.implemented_function("control_kernel_imp", sp.lambdify(y, control_kernel))(z)
control_law = (
    -kernel_ll * x_approx.subs(z, l)
    - sp.Integral(control_kernel_imp * x_approx_modal, (z, 0, l)))
control_law = sy.evaluate_integrals(control_law)
control_law = sy.evaluate_implemented_functions(control_law)
sy.pprint(control_law, "control law explicit")
control_law = (
    -kernel_ll * x_approx.subs(z, l)
    + sum([-c * (sp.diff(phi_bar, z) - sp.diff(phi, z) - kernel_ll * phi).subs(z, l)
           for c, phi, phi_bar in zip(weights_modal, shape_funcs_modal, shape_funcs_modal_target)]))
control_law = sy.evaluate_implemented_functions(control_law)
sy.pprint(control_law, "control law")
control_law_info = {"fem_base": weights, "primal_base": weights_modal}
control_law_imp = sy.Feedback(sp.Matrix([control_law]), control_law_info)

# closed loop feed forward
from pyinduct.parabolic.feedforward import RadFeedForward
feed_forward_control = RadFeedForward(
    l, T, params_target, "robin", "robin", n=60)

# system input implementation
input_ = sy.SimulationInputWrapper(pi.SimulationInputSum([
    control_law_imp, feed_forward_control]))

# input variable which holds a pyinduct.SimulationInputWrapper
# as implemented function needs a unique  variable  from which
# they depend, since they are called with a bunch of
# arguments during simulation
input_arg = var_pool.new_symbol("input_arg", "simulation input argument")
u = var_pool.new_implemented_function("u", (input_arg,), input_, "input")
input_vector = sp.Matrix([u])

# project on test functions
projections = list()
limits = (z, spat_bounds[0], spat_bounds[1])
for psi_j in test_funcs:
    projections.append(
        -sp.Integral(sp.diff(x_approx, t) * psi_j, limits)
        - a2 * sp.Integral(sp.diff(x_approx, z) * sp.diff(psi_j, z), limits)
        + a1 * sp.Integral(sp.diff(x_approx, z) * psi_j, limits)
        + a0 * sp.Integral(x_approx * psi_j, limits)
        - a2 * beta * x_approx.subs(z, l) * psi_j.subs(z, l)
        + u * psi_j.subs(z, l)
        - a2 * alpha * x_approx.subs(z, 0) * psi_j.subs(z, 0)
    )
projections = sp.Matrix(projections)
sy.pprint(projections, "projections", N_fem)

# evaluate integrals
projections = sy.evaluate_integrals(projections)

# evaluate remaining implemented functions
projections = sy.evaluate_implemented_functions(projections)
sy.pprint(projections, "evaluated projections", N_fem)

# initial conditions
init_samples = np.ones(len(weights))

# derive rhs and simulate
rhs = sy.derive_first_order_representation(projections, weights, input_vector,
                                           mode="sympy.linear_eq_to_matrix")
sy.pprint(rhs, "right hand side of the discretization", N_fem)

# use numpy.dot to speed up the simulation (compare / run without this line)
rhs = sy.implement_as_linear_ode(rhs, weights, input_vector)

# simulate
_, q = sy.simulate_system(
    rhs, weights, init_samples, "fem_base", input_vector, t, temp_dom)

# evaluate desired output data
y_d, t_d = pi.gevrey_tanh(T, 40)
C = pi.coefficient_recursion(y_d, alpha * y_d, params)
x_l = pi.power_series(np.array(spat_dom), t_d, C)
data_d = pi.EvalData([t_d, np.array(spat_dom)], x_l, name="x(z,t) desired")

# visualization
data = pi.get_sim_result("fem_base", q, temp_dom, spat_dom, 0, 0)[0]
win = pi.PgAnimatedPlot([data, data_d])
pi.show()
