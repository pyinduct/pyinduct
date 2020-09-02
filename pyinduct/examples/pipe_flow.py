r"""
This example considers the thermal behavior (simulation) of plug flow of an
incompressible fluid through a pipe from [BacEtAl17]_, which can be described
with the normed variables/parameters:

    - :math:`x_1(z,t)` ~ fluid temperature

    - :math:`x_2(z,t)` ~ pipe wall temperature

    - :math:`x_3(z,t)=0` ~ ambient temperature

    - :math:`u(t)` ~ system input

    - :math:`H(t)` ~ heaviside step function

    - :math:`v` ~ fluid velocity

    - :math:`c_1` ~ heat transfer coefficient (fluid - wall)

    - :math:`c_2` ~ heat transfer coefficient (wall - ambient)

by the following equations:

.. math::
    :nowrap:

    \begin{align*}
        \dot{x}_1(z,t) + v x_1'(z,t) &=
            c_1(x_2(z,t) - x_1(z,t)), && z\in (0,l] \\
        \dot{x}_2(z,t) &=
            c_1(x_1(z,t) - x_2(z,t)) + c_2(x_3(z,t) - x_2(z,t)), &&z\in [0,l] \\
        x_1(z,0) &= 0 \\
        x_2(z,0) &= 0 \\
        x_1(0,t) &= u(t) = 2 H(t)
    \end{align*}


.. [BacEtAl17] On thermal modelling of incrompressible pipe flows (Zur
        thermischen Modellierung inkompressibler Rohrströmungen),
        Simon Bachler, Johannes Huber and Frank Woittennek,
        at-Automatisierungstechnik, DE GRUYTER, 2017
"""
import numpy as np
import pyinduct as pi


# (sphinx directive) start actual script
def run(show_plots):
    v = 10
    c1, c2 = [1, 1]
    l = 5
    T = 5
    spat_bounds = (0, l)
    spat_domain = pi.Domain(bounds=spat_bounds, num=51)
    temp_domain = pi.Domain(bounds=(0, T), num=100)

    init_funcs1 = pi.LagrangeSecondOrder.cure_interval(spat_domain)
    nodes = pi.Domain(spat_domain.bounds, num=30)
    init_funcs2 = pi.LagrangeFirstOrder.cure_interval(nodes)
    pi.register_base("x1_funcs", init_funcs1)
    pi.register_base("x2_funcs", init_funcs2)

    u = pi.SimulationInputSum([
        pi.SignalGenerator('square', temp_domain, frequency=.03,
                           scale=2, offset=4, phase_shift=1),
    ])

    x1 = pi.FieldVariable("x1_funcs")
    psi1 = pi.TestFunction("x1_funcs")
    x2 = pi.FieldVariable("x2_funcs")
    psi2 = pi.TestFunction("x2_funcs")

    weak_form1 = pi.WeakFormulation(
        [
            pi.IntegralTerm(pi.Product(x1.derive(temp_order=1), psi1),
                            limits=spat_bounds),
            pi.IntegralTerm(pi.Product(x1, psi1.derive(1)),
                            limits=spat_bounds,
                            scale=-v),
            pi.ScalarTerm(pi.Product(x1(l), psi1(l)), scale=v),
            pi.ScalarTerm(pi.Product(pi.Input(u), psi1(0)), scale=-v),
            pi.IntegralTerm(pi.Product(x1, psi1),
                            limits=spat_bounds,
                            scale=c1),
            pi.IntegralTerm(pi.Product(x2, psi1),
                            limits=spat_bounds,
                            scale=-c1),
        ],
        name="fluid temperature"
    )
    weak_form2 = pi.WeakFormulation(
        [
            pi.IntegralTerm(pi.Product(x2.derive(temp_order=1), psi2),
                            limits=spat_bounds),
            pi.IntegralTerm(pi.Product(x1, psi2),
                            limits=spat_bounds,
                            scale=-c2),
            pi.IntegralTerm(pi.Product(x2, psi2),
                            limits=spat_bounds,
                            scale=c2 + c1),
        ],
        name="wall temperature"
    )

    ics = {weak_form1.name: [pi.Function(lambda z: np.sin(z/2),
                                         domain=spat_bounds)],
           weak_form2.name: [pi.Function(lambda z: 0, domain=spat_bounds)]}
    spat_domains = {weak_form1.name: spat_domain, weak_form2.name: spat_domain}
    evald1, evald2 = pi.simulate_systems([weak_form1, weak_form2],
                                         ics,
                                         temp_domain,
                                         spat_domains)
    pi.tear_down(["x1_funcs", "x2_funcs"])

    if show_plots:
        win1 = pi.PgAnimatedPlot([evald1, evald2], labels=dict(bottom='z'))
        win3 = pi.surface_plot(evald1, title=weak_form1.name)
        win4 = pi.surface_plot(evald2, title=weak_form2.name)
        pi.show()


if __name__ == "__main__":
    run(True)
