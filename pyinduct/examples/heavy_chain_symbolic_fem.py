import sympy as sp
import numpy as np
import pyinduct as pi
import pyinduct.symbolic as sy

# spatial approximation order
N = 2

# temporal domain
T = 5
temp_dom = pi.Domain((0, T), num=100)

# spatial domain
l = 1
spat_bounds = (0, l)
spat_dom = pi.Domain(spat_bounds, num=100)

# system input implementation
input_ = sy.SimulationInputWrapper(pi.SimulationInputSum([
    pi.SignalGenerator('sawtooth', np.array(temp_dom), frequency=0.2,
                       scale=1, offset=0, phase_shift=0)
]))

# variables
var_pool = sy.VariablePool("heavy chain")
t = var_pool.new_symbol("t", "time")
z = var_pool.new_symbol("z", "location")

# input variable which holds a pyinduct.SimulationInputWrapper
# as implemented function needs a unique  variable  from which
# they depend, since they are called with a bunch of
# arguments during simulation
input_var = var_pool.new_symbol("inputvar", "simulation input variable")
u = var_pool.new_implemented_function("u", (input_var,), input_, "input")
input_vector = sp.Matrix([u])

# system parameters
m_l = 1                 # [kg]
rho = 1# 1.78              # [kg/mm] -> line density
gravity = 9.81
l_hc = 1 # m
A_hc = 1 # 2.4*8*14.85 # m**2 -> cross sectional area of chain

# define approximation base and symbols
nodes = pi.Domain(spat_bounds, num=N)
fem_base = pi.LagrangeFirstOrder.cure_interval(nodes)
pi.register_base("complete_base", pi.Base(list(fem_base.fractions)*2))
pi.register_base("fem_base", fem_base)
init_funcs_w = var_pool.new_implemented_functions(
    ["phiw{}".format(i) for i in range(N)], [(z,)] * N,
    fem_base.fractions, "initial functions w")
init_funcs_v = var_pool.new_implemented_functions(
    ["phiv{}".format(i) for i in range(N)], [(z,)] * N,
    fem_base.fractions, "initial functions v")
test_funcs_half = sp.Matrix(var_pool.new_implemented_functions(
    ["psiw{}".format(i) for i in range(N)], [(z,)] * N,
    fem_base.fractions, "test functions w"))

# build approximation
weights_w = sp.Matrix(var_pool.new_functions(
    ["c_{w" + str(i) + "}" for i in range(N)], [(t,)] * N, "approximation weights for w"))
weights_v = sp.Matrix(var_pool.new_functions(
    ["c_{v" + str(i) + "}" for i in range(N)], [(t,)] * N, "approximation weights for v"))
w_approx = sum([c * phi for c, phi in zip(weights_w, init_funcs_w)])
v_approx = sum([c * phi for c, phi in zip(weights_v, init_funcs_v)])
sy.pprint(w_approx, "approximation of w", N)
sy.pprint(v_approx, "approximation of v", N)

# complete weights vector and set of test function
weights = sp.Matrix.vstack(weights_w, weights_v)
test_funcs_w = sp.Matrix.vstack(test_funcs_half, test_funcs_half * 0)
test_funcs_v = sp.Matrix.vstack(test_funcs_half * 0, test_funcs_half)
sy.pprint(test_funcs_w, "test functions", N)
sy.pprint(test_funcs_v, "test functions", N)

# project on test functions
projections = list()
limits = (z, spat_bounds[0], spat_bounds[1])
for psi_w, psi_v in zip(test_funcs_w, test_funcs_v):
    projections.append(
        sp.Integral(sp.diff(w_approx, t) * psi_w, limits)
        + sp.Integral(sp.diff(v_approx, t) * psi_v, limits)
        - sp.Integral(v_approx * psi_w, limits)
        + sp.Integral(sp.diff(w_approx, z) * psi_w, limits)
        - m_l * sp.diff(w_approx, z).subs(z, l_hc) * psi_w.subs(z, l_hc)
        + 1/(A_hc * rho) * u * psi_w.subs(z, 0)
        - sp.Integral(sp.diff(w_approx, z) * psi_w, limits)
        + (l_hc + m_l/(A_hc * rho)) * sp.Integral(sp.diff(w_approx, z) * sp.diff(psi_w, z), limits)
        - sp.Integral(sp.diff(w_approx, z) * sp.diff(psi_w, z) * z, limits)
    )
projections = sp.Matrix(projections)
sy.pprint(projections, "projections", N)

# evaluate integrals
projections = sy.evaluate_integrals(projections)

# evaluate remaining implemented functions
projections = sy.evaluate_implemented_functions(projections)
sy.pprint(projections, "evaluated projections", N)

# initial conditions
init_samples = np.zeros(len(weights))

# derive rhs and simulate
rhs = sy.derive_first_order_representation(projections, weights, input_vector,
                                           # mode="sympy.solve")
                                           mode="sympy.linear_eq_to_matrix")
sy.pprint(rhs, "right hand side of the discretization", N)

# use numpy.dot to speed up the simulation (compare / run without this line)
rhs = sy.implement_as_linear_ode(rhs, weights, input_vector)

# simulate
_, q = sy.simulate_system(
    rhs, weights, init_samples, "complete_base", input_vector, t, temp_dom)

# visualization
data = pi.get_sim_result("fem_base", q, temp_dom, spat_dom, 0, 0)
win = pi.PgAnimatedPlot(data)
pi.show()
