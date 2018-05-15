from pyinduct.tests import test_examples


def calc_eigen(order, l_value, EI, mu, der_order=4):
    import sympy as sp

    C, D, E, F = sp.symbols("C D E F")
    gamma, l = sp.symbols("gamma l")
    z = sp.symbols("z")

    eig_func = (C*sp.cos(gamma*z)
                + D*sp.sin(gamma*z)
                + E*sp.cosh(gamma*z)
                + F*sp.sinh(gamma*z))

    bcs = [eig_func.subs(z, 0),
           eig_func.diff(z, 1).subs(z, 0),
           eig_func.diff(z, 2).subs(z, l),
           eig_func.diff(z, 3).subs(z, l),
           ]
    e_sol = sp.solve(bcs[0], E)[0]
    f_sol = sp.solve(bcs[1], F)[0]
    new_bcs = [bc.subs([(E, e_sol), (F, f_sol)]) for bc in bcs[2:]]
    d_sol = sp.solve(new_bcs[0], D)[0]
    char_eq = new_bcs[1].subs([(D, d_sol), (l, l_value), (C, 1)])
    char_func = sp.lambdify(gamma, char_eq, modules="numpy")

    def char_wrapper(z):
        try:
            return char_func(z)
        except FloatingPointError:
            return 1

    grid = np.linspace(-1, 30, num=1000)
    roots = pi.find_roots(char_wrapper, grid, n_roots=order)
    if 0:
        pi.visualize_roots(roots, grid, char_func)

    # build eigenvectors
    eig_vec = eig_func.subs([(E, e_sol),
                             (F, f_sol),
                             (D, d_sol),
                             (l, l_value),
                             (C, 1)])
    # build derivatives
    eig_vec_derivatives = [eig_vec]
    for i in range(der_order):
        eig_vec_derivatives.append(eig_vec_derivatives[-1].diff(z, 1))

    # construct functions
    eig_fractions = []
    for root in roots:
        # localize and lambdify
        callbacks = [sp.lambdify(z, vec.subs(gamma, root), modules="numpy")
                     for vec in eig_vec_derivatives]

        frac = pi.Function(domain=(0, l_value),
                           eval_handle=callbacks[0],
                           derivative_handles=callbacks[1:])
        frac.eigenvalue = - root**4 * EI / mu
        eig_fractions.append(frac)

    eig_base = pi.Base(eig_fractions)
    normed_eig_base = pi.normalize_base(eig_base)

    if 0:
        pi.visualize_functions(eig_base.fractions)
        pi.visualize_functions(normed_eig_base.fractions)

    return normed_eig_base


if __name__ == "__main__" or test_examples:
    import pyinduct as pi
    import numpy as np

    sys_name = 'euler bernoulli beam'

    # domains
    spat_domain = pi.Domain(bounds=(0, 1), num=49)
    temp_domain = pi.Domain(bounds=(0, 10), num=1000)

    # physical properties
    height = .1  # [m]
    width = .1  # [m]
    e_module = 210e9  # [Pa]
    if 0:
        EI = 210e9 * (width * height**3)/12
        mu = 1e6  # [kg/m]
    else:
        EI = 1e0
        mu = 1e0

    # define approximation bases
    nodes, init_funcs = pi.cure_interval(pi.LagrangeNthOrder,
                                         spat_domain.bounds,
                                         order=3,
                                         node_count=len(spat_domain))
    pi.register_base("complete_base", init_funcs)

    red_nodes = pi.Domain(points=nodes.points[1:])
    red_init_funcs = pi.Base(init_funcs.fractions[1:])
    pi.register_base("reduced_base", red_init_funcs)

    eig_base = calc_eigen(7, 1, EI, mu)
    pi.register_base("eig_base", eig_base)

    approx_lbl = "eig_base"

    class ImpulseExcitation(pi.SimulationInput):

        def _calc_output(self, **kwargs):
            t = kwargs["time"]
            if t < 1:
                value = 0
            elif t < 1.1:
                value = 1000
            else:
                value = 0
            # value = 0
            return dict(output=value)

    u = ImpulseExcitation("Hammer")
    x = pi.FieldVariable(approx_lbl)
    phi = pi.TestFunction(approx_lbl)
    phi_u = pi.TestFunction(input_lbl)

    weak_form = pi.WeakFormulation([
        pi.ScalarTerm(pi.Product(pi.Input(u),
                                 phi(1)), scale=EI),
        pi.ScalarTerm(pi.Product(x.derive(spat_order=3)(0),
                                 phi(0)), scale=-EI),

        pi.ScalarTerm(pi.Product(x.derive(spat_order=2)(0),
                                 phi.derive(1)(0)), scale=EI),

        pi.ScalarTerm(pi.Product(x.derive(spat_order=1)(1),
                                 phi.derive(2)(1)), scale=EI),

        pi.IntegralTerm(pi.Product(x.derive(spat_order=1),
                                   phi.derive(3)),
                        spat_domain.bounds,
                        scale=-EI),
        pi.IntegralTerm(pi.Product(x.derive(temp_order=2), phi),
                        spat_domain.bounds,
                        scale=mu),
    ], name=sys_name)

    # init_form = eig_base.fractions[0]
    init_form = pi.Function.from_constant(0)
    init_form_dt = pi.Function.from_constant(0)
    initial_conditions = [init_form, init_form_dt]

    eval_data = pi.simulate_system(weak_form,
                                   initial_conditions,
                                   temp_domain,
                                   spat_domain,
                                   settings=dict(name="vode",
                                                 method="bdf",
                                                 order=5,
                                                 nsteps=1e4,
                                                 max_step=temp_domain.step))

    # pyqtgraph visualization
    # win0 = pg.plot(np.array(eval_data[0].input_data[0]).flatten(),
    #                u.get_results(eval_data[0].input_data[0]).flatten(),
    #                labels=dict(left='u(t)', bottom='t'), pen='b')
    # win0.showGrid(x=False, y=True, alpha=0.5)
    # vis.save_2d_pg_plot(win0, 'transport_system')

    win1 = pi.PgAnimatedPlot(eval_data, labels=dict(left='x(z,t)', bottom='z'))
    pi.show()

    # pi.tear_down((func_label,),
    #              (win1))

