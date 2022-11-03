r"""
This example considers a transmission line

    - :math:`x_1(z,t)` ~ line voltage

    - :math:`x_2(z,t)` ~ line current

    - :math:`u(t)` ~ system input

    - :math:`H(t)` ~ heaviside step function

    - :math:`v` ~ fluid velocity

    - :math:`L` ~ line length inductance per length

    - :math:`C` ~ line cross capacity per length

    - :math:`R` ~ line length resistance per length

    - :math:`R` ~ line cross conductance per length


by the following equations:

.. math::
    :nowrap:

    \begin{align*}
        \partial_z u(z,t) + L \partial_t i(z,t) + R i(z,t) &= 0
        \partial_z i(z,t) + C \partial_t u(z,t) + G u(z,t) &= 0
        u(0,t) &= U_e d(t) \\
        u(l,t) - Z i(l,t) &= 0
    \end{align*}

"""
import numpy as np
import pyinduct as pi

# parameters
l = 6  # [] = m
R = 0  # [] = ohm
G = 0  # [] = 1/ohm
L = 241e-9  # [] = H
C = 100e-12  # [] = F
Z = 10
Ue = 12

t_end = 1e-6


def run(show_plots):
    spat_bounds = (0, l)
    spat_domain = pi.Domain(bounds=spat_bounds, num=101)
    temp_domain = pi.Domain(bounds=(0, t_end), num=1000)

    # voltage
    # base_u = pi.LagrangeFirstOrder.cure_interval(spat_domain)
    base_u = pi.LagrangeSecondOrder.cure_interval(spat_domain)
    base_u0 = pi.Base(base_u[0])
    base_ub = pi.Base(base_u[1:])
    pi.register_base("base_ub", base_ub)
    pi.register_base("base_u0", base_u0)

    # current
    base_i = pi.LagrangeFirstOrder.cure_interval(spat_domain)
    base_ib = pi.Base(base_i[:-1])
    base_il = pi.Base(base_i[-1])
    pi.register_base("base_ib", base_ib)
    pi.register_base("base_il", base_il)

    d = pi.ConstantTrajectory(Ue)

    ub = pi.FieldVariable("base_ub")
    psi_u0 = pi.ScalarFunction("base_u0")
    psi_ub = pi.TestFunction("base_ub")
    ib = pi.FieldVariable("base_ib")
    psi_ib = pi.TestFunction("base_ib")
    psi_il = pi.ScalarFunction("base_il")

    inp = pi.Input(d)

    weak_form_u = pi.WeakFormulation(
        [
            # homogeneous part
            pi.IntegralTerm(pi.Product(ub.derive(temp_order=1), psi_ub),
                            limits=spat_bounds,
                            scale=C),
            pi.IntegralTerm(pi.Product(ub, psi_ub),
                            limits=spat_bounds,
                            scale=G),
            pi.ScalarTerm(pi.Product(ib(0), psi_ub(0)), scale=-1),
            pi.ScalarTerm(pi.Product(ib(l), psi_ub(l)), scale=1),
            pi.IntegralTerm(pi.Product(ib, psi_ub.derive(1)),
                            limits=spat_bounds,
                            scale=-1),

            # inhomogeneous part (input derivative is ignored)
            # pi.IntegralTerm(pi.Product(inp.derive(temp_order=1), psi_ub),
            #                 limits=spat_bounds,
            #                 scale=C),
            pi.IntegralTerm(pi.Product(pi.Product(psi_u0, psi_ub),
                                       inp),
                            limits=spat_bounds,
                            scale=G),
            pi.ScalarTerm(pi.Product(pi.Product(ub(l), psi_il(0)),
                                     psi_ub(0)), scale=-1/Z),
            pi.ScalarTerm(pi.Product(pi.Product(ub(l), psi_il(l)),
                                     psi_ub(l)), scale=1/Z),
            pi.IntegralTerm(pi.Product(pi.Product(ub(l), psi_il),
                                       psi_ub.derive(1)),
                            limits=spat_bounds,
                            scale=-1/Z),
        ],
        name="line voltage"
    )
    weak_form_i = pi.WeakFormulation(
        [
            # homogeneous part
            pi.IntegralTerm(pi.Product(ib.derive(temp_order=1), psi_ib),
                            limits=spat_bounds,
                            scale=L),
            pi.IntegralTerm(pi.Product(ib, psi_ib),
                            limits=spat_bounds,
                            scale=R),
            pi.ScalarTerm(pi.Product(ub(0), psi_ib(0)), scale=-1),
            pi.ScalarTerm(pi.Product(ub(l), psi_ib(l)), scale=1),
            pi.IntegralTerm(pi.Product(ub, psi_ib.derive(1)),
                            limits=spat_bounds,
                            scale=-1),

            # inhomogeneous part
            # pi.IntegralTerm(pi.Product(pi.Product(ub(l), psi_il),
            #                            psi_ib),
            #                 limits=spat_bounds,
            #                 scale=G/C * L/Z),
            # pi.IntegralTerm(pi.Product(pi.Product(ub.derive(spat_order=1)(l),
            #                                       psi_il),
            #                            psi_ib),
            #                 limits=spat_bounds,
            #                 scale=1/(C * Z) * L/Z),
            # pi.IntegralTerm(pi.Product(pi.Product(ib.derive(spat_order=1)(l),
            #                                       psi_il),
            #                            psi_ib),
            #                 limits=spat_bounds,
            #                 scale=1/C * L/Z),
            pi.IntegralTerm(pi.Product(pi.Product(ub(l), psi_il), psi_ib),
                            limits=spat_bounds,
                            scale=R/Z),
            pi.ScalarTerm(pi.Product(pi.Product(psi_u0(0), psi_ib(0)),
                                     inp),
                          scale=-1),
            pi.ScalarTerm(pi.Product(pi.Product(psi_u0(l), psi_ib(l)),
                                     inp),
                          scale=1),
            pi.IntegralTerm(pi.Product(pi.Product(psi_u0, psi_ib.derive(1)),
                                       inp),
                            limits=spat_bounds,
                            scale=-1),
        ],
        name="line current"
    )

    ics = {
        weak_form_u.name: [pi.Function(lambda z: 0, domain=spat_bounds)],
        weak_form_i.name: [pi.Function(lambda z: 0, domain=spat_bounds)]
    }
    spat_domains = {
        weak_form_u.name: spat_domain,
        weak_form_i.name: spat_domain
    }
    eval_ub, eval_ib = pi.simulate_systems([weak_form_u, weak_form_i],
                                         ics,
                                         temp_domain,
                                         spat_domains)

    # add the inhomogeneous parts
    u0_weights = np.reshape(d.get_results(temp_domain), (len(temp_domain), 1))
    eval_u0 = pi.evaluate_approximation("base_u0", u0_weights,
                                        temp_domain, spat_domain, spat_order=0)
    eval_u = eval_u0 + eval_ub
    il_weights = eval_u.output_data[:, -1:] / Z
    eval_il = pi.evaluate_approximation("base_il", il_weights,
                                        temp_domain, spat_domain, spat_order=0)
    eval_i = eval_ib + eval_il

    if show_plots:
        win1 = pi.PgAnimatedPlot([eval_u, eval_i], labels=dict(bottom='z'),
                                 replay_gain=t_end/5)
        win2 = pi.surface_plot(eval_u0, title=weak_form_u.name)
        win3 = pi.surface_plot(eval_ub, title=weak_form_u.name)
        win4 = pi.surface_plot(eval_u, title=weak_form_u.name)
        win5 = pi.surface_plot(eval_ib, title=weak_form_i.name)
        win6 = pi.surface_plot(eval_il, title=weak_form_i.name)
        win7 = pi.surface_plot(eval_i, title=weak_form_i.name)
        pi.show()

    # cleanup
    pi.tear_down(["base_u0", "base_ub", "base_ib", "base_il"])

if __name__ == "__main__":
    run(True)
