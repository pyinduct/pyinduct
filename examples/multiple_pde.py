import pyinduct.core as cr
import pyinduct.placeholder as ph
import pyinduct.registry as reg
import pyinduct.shapefunctions as sh
import pyinduct.simulation as sim
import pyinduct.trajectory as tr
import pyinduct.visualization as vis
import numpy as np
import pyqtgraph as pg

v = 10
c1, c2, c3 = [1, 1, 1]
l = 5
T = 5
spat_domain = sim.Domain(bounds=(0, l), num=51)
temp_domain = sim.Domain(bounds=(0, T), num=1e2)

init_x1 = np.zeros(len(spat_domain))
init_x2 = np.zeros(len(spat_domain) - 21)

nodes, init_funcs1 = sh.cure_interval(sh.LagrangeSecondOrder, spat_domain.bounds, node_count=len(init_x1))
nodes, init_funcs2 = sh.cure_interval(sh.LagrangeFirstOrder, spat_domain.bounds, node_count=len(init_x2))
reg.register_base("x1_funcs", init_funcs1)
reg.register_base("x2_funcs", init_funcs2)

u = sim.SimulationInputSum([
    tr.SignalGenerator('square', np.array(temp_domain), frequency=.03, scale=2, offset=4, phase_shift=1),
])

x1 = ph.FieldVariable("x1_funcs")
psi1 = ph.TestFunction("x1_funcs")
x2 = ph.FieldVariable("x2_funcs")
psi2 = ph.TestFunction("x2_funcs")

weak_form1 = sim.WeakFormulation(
    [
        ph.IntegralTerm(ph.Product(x1.derive_temp(1), psi1), limits=spat_domain.bounds),
        ph.IntegralTerm(ph.Product(x1, psi1.derive(1)), limits=spat_domain.bounds, scale=-v),
        ph.ScalarTerm(ph.Product(x1(l), psi1(l)), scale=v),
        ph.ScalarTerm(ph.Product(ph.Input(u), psi1(0)), scale=-v),
        ph.IntegralTerm(ph.Product(x1, psi1), limits=spat_domain.bounds, scale=c1),
        ph.IntegralTerm(ph.Product(x2, psi1), limits=spat_domain.bounds, scale=-c1),
    ],
    dynamic_weights="x1_funcs"
)
weak_form2 = sim.WeakFormulation(
    [
        ph.IntegralTerm(ph.Product(x2.derive_temp(1), psi2), limits=spat_domain.bounds),
        ph.IntegralTerm(ph.Product(x1, psi2), limits=spat_domain.bounds, scale=-c2),
        ph.IntegralTerm(ph.Product(x2, psi2), limits=spat_domain.bounds, scale=c2 + c3),
    ],
    dynamic_weights="x2_funcs"
)

cfs1 = sim.parse_weak_formulation(weak_form1)
cfs2 = sim.parse_weak_formulation(weak_form2)
state_space = sim.convert_cfs_to_state_space([cfs1, cfs2])

t, q1, q2 = sim.simulate_state_space(state_space, np.hstack((init_x1, init_x2)), temp_domain)
evald1 = sim.evaluate_approximation("x1_funcs", q1, temp_domain, spat_domain, spat_order=0)
evald2 = sim.evaluate_approximation("x2_funcs", q2, temp_domain, spat_domain, spat_order=0)

win1 = vis.PgAnimatedPlot(evald1, labels=dict(left='x_1(z,t)', bottom='z'), title="fluid temperature")
win2 = vis.PgAnimatedPlot(evald2, labels=dict(left='x_2(z,t)', bottom='z'), title="wall temperature")
win3 = vis.PgSurfacePlot(evald1, title="fluid temperature")
win4 = vis.PgSurfacePlot(evald2, title="wall temperature")
pg.QtGui.QApplication.instance().exec_()
