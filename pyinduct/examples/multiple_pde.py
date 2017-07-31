import pyinduct as pi

v = 10
c1, c2, c3 = [1, 1, 1]
l = 5
T = 5
spat_domain = pi.Domain(bounds=(0, l), num=51)
temp_domain = pi.Domain(bounds=(0, T), num=100)

_, init_funcs1 = pi.cure_interval(pi.LagrangeSecondOrder,
                                  spat_domain.bounds,
                                  node_count=51)
_, init_funcs2 = pi.cure_interval(pi.LagrangeFirstOrder,
                                  spat_domain.bounds,
                                  node_count=30)
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
        pi.IntegralTerm(pi.Product(x1.derive(temp_order=1), psi1), limits=spat_domain.bounds),
        pi.IntegralTerm(pi.Product(x1, psi1.derive(1)), limits=spat_domain.bounds, scale=-v),
        pi.ScalarTerm(pi.Product(x1(l), psi1(l)), scale=v),
        pi.ScalarTerm(pi.Product(pi.Input(u), psi1(0)), scale=-v),
        pi.IntegralTerm(pi.Product(x1, psi1), limits=spat_domain.bounds, scale=c1),
        pi.IntegralTerm(pi.Product(x2, psi1), limits=spat_domain.bounds, scale=-c1),
    ],
    name="fluid temperature"
)
weak_form2 = pi.WeakFormulation(
    [
        pi.IntegralTerm(pi.Product(x2.derive(temp_order=1), psi2),limits=spat_domain.bounds),
        pi.IntegralTerm(pi.Product(x1, psi2), limits=spat_domain.bounds, scale=-c2),
        pi.IntegralTerm(pi.Product(x2, psi2), limits=spat_domain.bounds, scale=c2 + c3),
    ],
    name="wall temperature"
)

ics = {weak_form1.name: [pi.Function(lambda z: 0)],
       weak_form2.name: [pi.Function(lambda z: 0)]}
spat_domains = {weak_form1.name: spat_domain, weak_form2.name: spat_domain}
evald1, evald2 = pi.simulate_systems([weak_form1, weak_form2], ics, temp_domain,
                                     spat_domains)

win1 = pi.PgAnimatedPlot([evald1, evald2], labels=dict(bottom='z'))
win3 = pi.PgSurfacePlot(evald1, title=weak_form1.name)
win4 = pi.PgSurfacePlot(evald2, title=weak_form2.name)
pi.show()
