"""
Simulation of a diffusion system using multiple inputs
"""
import pyinduct as pi


def run(show_plots):
    # physical constants
    alpha = .1

    # define domains
    temp_dom = pi.Domain(bounds=(0, 10), num=100)
    spat_bounds = (0, 1)
    spat_dom = pi.Domain(bounds=spat_bounds, num=100)

    # create approximation basis
    nodes = pi.Domain(spat_bounds, num=11)
    if 0:
        # old interface
        _, fem_funcs = pi.LagrangeSecondOrder.cure_hint(nodes)
        fem_base = pi.Base(fem_funcs)
    else:
        # new interface
        fem_base = pi.LagrangeSecondOrder.cure_interval(nodes)
    pi.register_base("fem_base", fem_base)

    # create equation objects
    field_var = pi.FieldVariable("fem_base")
    field_var_dt = field_var.derive(temp_order=1)
    field_var_dz = field_var.derive(spat_order=1)

    psi = pi.TestFunction("fem_base")
    psi_dz = psi.derive(1)

    # define inputs
    input_traj = pi.SimulationInputVector([pi.ConstantTrajectory(10),
                                          pi.ConstantTrajectory(-10)])
    left_input = pi.Input(input_traj, index=0)
    right_input = pi.Input(input_traj, index=1)

    # system dynamics
    temp_int = pi.IntegralTerm(pi.Product(field_var_dt, psi),
                               limits=spat_bounds)
    spat_int = pi.IntegralTerm(pi.Product(field_var_dz, psi_dz),
                               limits=spat_bounds,
                               scale=alpha)
    input_term1 = pi.ScalarTerm(pi.Product(left_input, psi(spat_bounds[0])))
    input_term2 = pi.ScalarTerm(pi.Product(right_input, psi(spat_bounds[1])), scale=-1)

    # derive sate-space system
    pde = pi.WeakFormulation([temp_int, spat_int, input_term1, input_term2],
                             name="diff_equation")

    # define initial state
    t0 = pi.ConstantFunction(100, domain=spat_bounds)

    # simulate
    res = pi.simulate_system(pde, t0, temp_dom, spat_dom)
    pi.tear_down(["fem_base"])

    if show_plots:
        # display results
        win = pi.PgAnimatedPlot(res, title="fem approx and derivative")
        win2 = pi.surface_plot(res)
        pi.show()


if __name__ == "__main__":
    run(True)
