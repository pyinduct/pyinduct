import sympy as sp
import numpy as np
import pyinduct as pi
import pyinduct.symbolic as ss

# spatial approximation order
N = 40

# temporal domain
T = 15
temp_dom = pi.Domain((0, T), num=100)

# spatial domain
l = 10
spat_bounds = (0, l)
spat_dom = pi.Domain(spat_bounds, num=100)

# system input implementation
input_ = ss.SimulationInputWrapper(pi.InterpolationTrajectory(
    temp_dom.points, (np.heaviside(temp_dom.points - 1, .5) +
                      np.heaviside(temp_dom.points - 5, .5) * 5 +
                      np.heaviside(temp_dom.points - 10, .5) * (-8))
))

# variables
var_pool = ss.VariablePool("transport system")
t = var_pool.new_symbol("t", "time")
z = var_pool.new_symbol("z", "location")

# input variable which holds a pyinduct.SimulationInputWrapper
# as implemented function needs a unique  variable  from which
# they depend, since they are called with a bunch of
# arguments during simulation
input_var = var_pool.new_symbol("inputvar", "simulation input variable")
u = var_pool.new_implemented_function("u", (input_var,), input_, "input")

# system parameters
velocity = lambda t: 0 if np.sin(4 * t) > 0 else 5
v = var_pool.new_implemented_function("v", (t,), velocity, "system parameters")

# approximation base and symbols
nodes = pi.Domain(spat_bounds, num=N)
step_size = l / N
fdm_base = pi.LagrangeFirstOrder.cure_interval(nodes)
pi.register_base("fdm_base", fdm_base)
samples = sp.Matrix(var_pool.new_functions(
    ["x{}".format(i) for i in range(1, N)], [(t,)] * (N - 1), "sample points"))
ss.pprint(samples, "samples", N)

# initial conditions
init_samples = np.zeros(len(samples))

# upwind approximation scheme
discretisation = [sp.diff(samples[0], t) + v / step_size * (samples[0] - u)]
for i in range(1, N - 1):
    discretisation.append(
        sp.diff(samples[i], t) + v / step_size * (samples[i] - samples[i - 1])
    )
discretisation = sp.Matrix(discretisation)
ss.pprint(discretisation, "discretization", N)

# derive rhs and simulate
sol = sp.solve(discretisation, sp.diff(samples, t))
rhs = sp.Matrix([sol[sample] for sample in sp.diff(samples, t)])
ss.pprint(rhs, "right hand side of the discretization", N)
_, q = ss.simulate_system(rhs, samples, init_samples, "fdm_base", [u], t, temp_dom)

# visualization
u_data = np.reshape(input_._sim_input.get_results(temp_dom.points), (len(temp_dom), 1))
q_full = np.hstack((u_data, q))
data = pi.get_sim_result("fdm_base", q_full, temp_dom, spat_dom, 0, 0)
win = pi.PgAnimatedPlot(data)
pi.show()
