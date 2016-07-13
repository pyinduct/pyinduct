import pyinduct.core as cr
import pyinduct.placeholder as ph
import pyinduct.registry as reg
import pyinduct.shapefunctions as sh
import pyinduct.simulation as sim
import pyinduct.trajectory as tr
import pyinduct.visualization as vis
import numpy as np
import pyqtgraph as pg

sys_name = 'transport system'
v = 10
l = 5
T = 5
spat_domain = sim.Domain(bounds=(0, l), num=51)
temp_domain = sim.Domain(bounds=(0, T), num=1e2)

init_x = cr.Function(lambda z: 0)

nodes, init_funcs = sh.cure_interval(sh.LagrangeSecondOrder, spat_domain.bounds, node_count=len(spat_domain))
func_label = 'init_funcs'
reg.register_base(func_label, init_funcs)

u = sim.SimulationInputSum([
    tr.SignalGenerator('square', np.array(temp_domain), frequency=0.3, scale=2, offset=4, phase_shift=1),
    tr.SignalGenerator('gausspulse', np.array(temp_domain), phase_shift=temp_domain[15]),
    tr.SignalGenerator('gausspulse', np.array(temp_domain), phase_shift=temp_domain[25], scale=-4),
    tr.SignalGenerator('gausspulse', np.array(temp_domain), phase_shift=temp_domain[35]),
    tr.SignalGenerator('gausspulse', np.array(temp_domain), phase_shift=temp_domain[60], scale=-2),
])

weak_form = sim.WeakFormulation([
    ph.IntegralTerm(ph.Product(ph.TemporalDerivedFieldVariable(func_label, 1), ph.TestFunction(func_label)),
                    spat_domain.bounds),
    ph.IntegralTerm(ph.Product(ph.FieldVariable(func_label), ph.TestFunction(func_label, order=1)), spat_domain.bounds,
                    scale=-v),
    ph.ScalarTerm(ph.Product(ph.FieldVariable(func_label, location=l), ph.TestFunction(func_label, location=l)),
                  scale=v),
    ph.ScalarTerm(ph.Product(ph.Input(u), ph.TestFunction(func_label, location=0)),
                  scale=-v),
], name=sys_name)

eval_data = sim.simulate_system(weak_form, init_x, temp_domain, spat_domain)

win0 = pg.plot(np.array(eval_data[0].input_data[0]), u.get_results(eval_data[0].input_data[0]),
               labels=dict(left='u(t)', bottom='t'), pen='b')
win0.showGrid(x=False, y=True, alpha=0.5)
vis.save_2d_pg_plot(win0, 'transport_system')
win1 = vis.PgAnimatedPlot(eval_data, title=eval_data[0].name,
                          save_pics=True, labels=dict(left='x(z,t)', bottom='z'))
pg.QtGui.QApplication.instance().exec_()

# ffmpeg -r "10" -i Fri_Jun_24_15:03:21_2016_%04d.png -c:v libx264 -pix_fmt yuv420p transport_system.mp4
# ffmpeg -i Fri_Jun_24_16:14:50_2016_%04d.png transport_system.gif
# convert Fri_Jun_24_16:14:50_2016_00*.png out.gif
