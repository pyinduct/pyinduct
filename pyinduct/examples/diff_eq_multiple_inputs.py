"""
Simulation of a simple heat diffusion equation given by:
"""
from pyinduct.tests import test_examples


def main():

    # physical constants
    alpha = .1

    # define domains
    temp_dom = pi.Domain(bounds=(0, 10), num=100)
    gamma = (0, 1)
    spat_dom = pi.Domain(bounds=gamma, num=100)

    # create approximation basis
    nodes = pi.Domain(gamma, num=11)
    if 1:
        # old interface
        _, fem_funcs = pi.LagrangeSecondOrder.cure_hint(nodes)
        fem_base = pi.Base(fem_funcs)
    else:
        # new interface
        fem_base = pi.LagrangeSecondOrder.cure_hint(nodes)
    pi.register_base("fem_base", fem_base)

    # create equation objects
    field_var = pi.FieldVariable("fem_base")
    field_var_dt = field_var.derive(temp_order=1)
    field_var_dz = field_var.derive(spat_order=1)

    psi = pi.TestFunction("fem_base")
    psi_dz = psi.derive(1)

    input1 = pi.Input(pi.ConstantTrajectory(10), index=0)
    input2 = pi.Input(pi.ConstantTrajectory(-10), index=1)

    # enter string with mass equations
    temp_int = pi.IntegralTerm(pi.Product(field_var_dt, psi),
                               limits=gamma)
    spat_int = pi.IntegralTerm(pi.Product(field_var_dz, psi_dz),
                               limits=gamma,
                               scale=alpha)
    input_term1 = pi.ScalarTerm(pi.Product(input1, psi(gamma[0])))
    input_term2 = pi.ScalarTerm(pi.Product(input2, psi(gamma[1])), scale=-1)

    # derive sate-space system
    pde = pi.WeakFormulation([temp_int, spat_int, input_term1, input_term2],
                             name="diff_equation")

    # define initial state
    t0 = pi.Function.from_constant(100)

    # simulate
    res = pi.simulate_system(pde, t0, temp_dom, spat_dom)

    # display results
    win = pi.PgAnimatedPlot(res, title="fem approx and derivative")
    win2 = pi.PgSurfacePlot(res)
    pi.show()


if __name__ == "__main__" or test_examples:
    import pyinduct as pi
    main()
