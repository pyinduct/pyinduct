import sympy as sp
# import symengine as sp
import numpy as np
import pyinduct as pi
import pyinduct.symbolic as ss
from matplotlib import pyplot as plt

# approximation order
N = 2

# temporal domain
T = 5
temp_dom = pi.Domain((0, T), num=100)

# spatial domain
l = 2
spat_bounds = (0, l)
spat_dom = pi.Domain(spat_bounds, num=100)

# define variables
var_pool = ss.VariablePool("transport system")
t = var_pool.new_symbol("t", "time")
z = var_pool.new_symbol("z", "location")
u = var_pool.new_function("u", (t,), "input")
x = var_pool.new_function("x", (z, t), "field variable")

# define system parameters
velocity = lambda t: 2 + np.sin(t)
v = var_pool.new_implemented_function("v", (t,), velocity, "system parameters")

# define approximation base and symbols
nodes = pi.Domain(spat_bounds, num=N)
fem_base = pi.LagrangeFirstOrder.cure_interval(nodes)
pi.register_base("fem", fem_base)
init_funcs = var_pool.new_implemented_functions(
    ["phi{}".format(i) for i in range(N)], [(z,)] * N,
    fem_base.fractions, "initial functions")
test_funcs = var_pool.new_implemented_functions(
    ["psi{}".format(i) for i in range(N)], [(z,)] * N,
    fem_base.fractions, "test functions")

# build approximation
weights = var_pool.new_functions(
    ["c{}".format(i) for i in range(N)], [(t,)] * N, "approximation weights")
x_approx = sum([c * phi for c, phi in zip(weights, init_funcs)])
ss.pprint(x_approx)

# project on test functions
projections = list()
limits = (z, spat_bounds[0], spat_bounds[1])
for psi_j in test_funcs:
    projections.append(
        sp.Integral(sp.diff(x_approx, t) * psi_j, limits)
        - v * sp.Integral(x_approx * sp.diff(psi_j, z), limits)
        + v * x_approx.subs(z, l) * psi_j.subs(z, l)
        - v * u * psi_j.subs(z, 0)
    )
ss.pprint(sp.Matrix(projections))

# evaluate projections
projections = evaluate_implemented_functions(projections)
projections = evaluate_integrals(projections)
state_space = build_linear_state_space(projections, weights, u)
sim_results = simulate_state_space(state_space)

