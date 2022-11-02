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
L = 241e-6  # [] = H
C = 100e-12  # [] = F
Z = 10
Ue = 12

t_end = 10e-6


def run(show_plots):
    spat_bounds = (0, l)
    spat_domain = pi.Domain(bounds=spat_bounds, num=51)
    temp_domain = pi.Domain(bounds=(0, t_end), num=1000)

    base_u = pi.LagrangeFirstOrder.cure_interval(spat_domain)
    red_base_u = base_u[1:]
    pi.register_base("base_u", red_base_u)
    base_i = pi.LagrangeFirstOrder.cure_interval(spat_domain)
    red_base_i = base_i[:-1]
    pi.register_base("base_i", red_base_i)

    d = pi.SimulationInputSum([
        # pi.SignalGenerator("square", temp_domain, frequency=.01,
        #                    scale=Ue, offset=0, phase_shift=0),
        pi.ConstantTrajectory(Ue)
    ])

    u = pi.FieldVariable("base_u")
    psi_u = pi.TestFunction("base_u")
    i = pi.FieldVariable("base_i")
    psi_i = pi.TestFunction("base_i")

    weak_form1 = pi.WeakFormulation(
        [
            pi.IntegralTerm(pi.Product(u.derive(temp_order=1), psi_u),
                            limits=spat_bounds,
                            scale=C),
            pi.IntegralTerm(pi.Product(u, psi_u),
                            limits=spat_bounds,
                            scale=G),
            pi.ScalarTerm(pi.Product(i(0), psi_u(0)), scale=-1),
            pi.ScalarTerm(pi.Product(u(l), psi_u(l)), scale=1/Z),
            pi.IntegralTerm(pi.Product(i, psi_u.derive(1)),
                            limits=spat_bounds,
                            scale=-1),
        ],
        name="line voltage"
    )
    weak_form2 = pi.WeakFormulation(
        [
            pi.IntegralTerm(pi.Product(i.derive(temp_order=1), psi_i),
                            limits=spat_bounds,
                            scale=L),
            pi.IntegralTerm(pi.Product(i, psi_i),
                            limits=spat_bounds,
                            scale=R),
            pi.ScalarTerm(pi.Product(pi.Input(d), psi_i(0)), scale=-1),
            pi.ScalarTerm(pi.Product(u(l), psi_i(l)), scale=1),
            pi.IntegralTerm(pi.Product(u, psi_i.derive(1)),
                            limits=spat_bounds,
                            scale=-1),
        ],
        name="line current"
    )

    ics = {
        weak_form1.name: [pi.Function(lambda z: 0, domain=spat_bounds)],
        weak_form2.name: [pi.Function(lambda z: 0, domain=spat_bounds)]
    }
    spat_domains = {
        weak_form1.name: spat_domain,
        weak_form2.name: spat_domain
    }
    evald1, evald2 = pi.simulate_systems([weak_form1, weak_form2],
                                         ics,
                                         temp_domain,
                                         spat_domains)
    pi.tear_down(["base_u", "base_i"])

    if show_plots:
        win1 = pi.PgAnimatedPlot([evald1, evald2], labels=dict(bottom='z'),
                                 replay_gain=1e-6)
        win3 = pi.surface_plot(evald1, title=weak_form1.name)
        win4 = pi.surface_plot(evald2, title=weak_form2.name)
        pi.show()


if __name__ == "__main__":
    run(True)
