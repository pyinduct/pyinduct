import sympy as sp
import numpy as np
import pyinduct as pi
import mpmath
mpmath.mp.dps = 30
# order matters
import pyinduct.symbolic as sy

# spatial approximation order
N = 10

# temporal domain
T = 5
temp_dom = pi.Domain((0, T), num=100)

# spatial domain
l = 5
spat_bounds = (0, l)
spat_dom = pi.Domain(spat_bounds, num=100)

# system input implementation
input_ = sy.SimulationInputWrapper(pi.SimulationInputSum([
        pi.SignalGenerator('square', np.array(temp_dom), frequency=0.1,
                           scale=1, offset=1, phase_shift=1),
        pi.SignalGenerator('square', np.array(temp_dom), frequency=0.2,
                           scale=2, offset=2, phase_shift=2),
        pi.SignalGenerator('square', np.array(temp_dom), frequency=0.3,
                           scale=3, offset=3, phase_shift=3),
        pi.SignalGenerator('square', np.array(temp_dom), frequency=0.4,
                           scale=4, offset=4, phase_shift=4),
        pi.SignalGenerator('square', np.array(temp_dom), frequency=0.5,
                           scale=5, offset=5, phase_shift=5),
    ]))

# variables
var_pool = sy.VariablePool("transport system")
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
velocity = lambda t: 0 if np.sin(4 * t) > 0 else 10
v = var_pool.new_implemented_function("v", (t,), velocity, "system parameters")
v = 10

# define approximation base and symbols
nodes = pi.Domain(spat_bounds, num=N)
fem_base = pi.LagrangeFirstOrder.cure_interval(nodes)
pi.register_base("fem_base", fem_base)
init_funcs = var_pool.new_implemented_functions(
    ["phi{}".format(i) for i in range(N)], [(z,)] * N,
    fem_base.fractions, "initial functions")
test_funcs = var_pool.new_implemented_functions(
    ["psi{}".format(i) for i in range(N)], [(z,)] * N,
    fem_base.fractions, "test functions")

# build approximation
weights = sp.Matrix(var_pool.new_functions(
    ["c{}".format(i) for i in range(N)], [(t,)] * N, "approximation weights"))
x_approx = sum([c * phi for c, phi in zip(weights, init_funcs)])
sy.pprint(x_approx, "approximation", N)

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
projections = sp.Matrix(projections)
sy.pprint(projections, "projections", N)

# evaluate integrals
projections = sy.evaluate_integrals(projections)

# evaluate remaining implemented functions
projections = projections.n()
sy.pprint(projections, "evaluated projections", N)

# initial conditions
init_samples = np.zeros(len(weights))

# derive rhs and simulate
rhs = sy.derive_first_order_representation(projections, weights, input_vector,
                                           # mode="sympy.solve")
                                           mode="sympy.linear_eq_to_matrix")
sy.pprint(rhs, "right hand side of the discretization", N)
_, q = sy.simulate_system(
    rhs, weights, init_samples, "fem_base", input_vector, t, temp_dom)

# visualization
data = pi.get_sim_result("fem_base", q, temp_dom, spat_dom, 0, 0)
win = pi.PgAnimatedPlot(data)
pi.show()
