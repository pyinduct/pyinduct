from __future__ import division
import unittest
import numpy as np
import pyqtgraph as pg
from pyinduct import core as cr, simulation as sim, utils as ut, visualization as vis, trajectory as tr

__author__ = 'marcus'

actuation = 'robin'
boundary_condition = 'robin'
param = [2., 1.5, -3., -1., -.5]
adjoint_param = ut.get_adjoint_rad_robin_evp_param(param)
a2, a1, a0, alpha, beta = param
l = 1.; spatial_domain = (0, l); spatial_disc = 50
T = 1.; temporal_domain = (0, T); temporal_disc = 1e3
n = 20

rad_eig_val = ut.ReaAdvDifRobinEigenvalues(param, l, n)
eig_val = rad_eig_val.eig_values
om_squared = rad_eig_val.om_squared

rad_eig_funcs = np.array([ut.ReaAdvDifRobinEigenfunction(ii, param, spatial_domain) for ii in om_squared])
rad_adjoint_eig_funcs = np.array([ut.ReaAdvDifRobinEigenfunction(ii, adjoint_param, spatial_domain) for ii in om_squared])

for i in xrange(len(rad_adjoint_eig_funcs)):
    scale = ut.normalize(rad_eig_funcs[i], rad_adjoint_eig_funcs[i], l)
    rad_eig_funcs[i] = rad_eig_funcs[i].scale(scale)
    rad_adjoint_eig_funcs[i] = rad_adjoint_eig_funcs[i].scale(scale)

u = tr.ReaAdvDifTrajectory(l, T, param, boundary_condition, actuation)

if False:
    # integral terms
    int1 = sim.IntegralTerm(sim.Product(sim.TemporalDerivedFieldVariable(rad_eig_funcs, order=1),
                                        sim.TestFunctions(rad_adjoint_eig_funcs, order=0)), spatial_domain)
    int2 = sim.IntegralTerm(sim.Product(sim.SpatialDerivedFieldVariable(rad_eig_funcs, order=1),
                                        sim.TestFunctions(rad_adjoint_eig_funcs, order=1)), spatial_domain, a2)
    int3 = sim.IntegralTerm(sim.Product(sim.SpatialDerivedFieldVariable(rad_eig_funcs, order=1),
                                        sim.TestFunctions(rad_adjoint_eig_funcs, order=0)), spatial_domain, -a1)
    int4 = sim.IntegralTerm(sim.Product(sim.SpatialDerivedFieldVariable(rad_eig_funcs, order=0),
                                        sim.TestFunctions(rad_adjoint_eig_funcs, order=0)), spatial_domain, -a0)

    # scalar terms
    s1 = sim.ScalarTerm(sim.Product(sim.SpatialDerivedFieldVariable(rad_eig_funcs, order=0, location=0),
                                    sim.TestFunctions(rad_adjoint_eig_funcs, order=0, location=0)), a2*alpha)
    s2 = sim.ScalarTerm(sim.Product(sim.SpatialDerivedFieldVariable(rad_eig_funcs, order=0, location=l),
                                    sim.TestFunctions(rad_adjoint_eig_funcs, order=0, location=l)), a2*beta)
    s3 = sim.ScalarTerm(sim.Product(sim.Input(u),
                                    sim.TestFunctions(rad_adjoint_eig_funcs, order=0, location=l)), -a2)

    # derive state-space system
    rad_pde = sim.WeakFormulation([int1, int2, int3, int4, s1, s2, s3])
    cf = sim.parse_weak_formulation(rad_pde)
    A, B = cf.convert_to_state_space()

    start_state = cr.Function(lambda z: 0., domain=(0, l))
    initial_weights = cr.project_on_initial_functions(start_state, rad_adjoint_eig_funcs)
    t, q = sim.simulate_state_space(A, B, cf.input_function, initial_weights, temporal_domain, time_step=1e-3)
else:
    A = np.diag(eig_val)
    B = a2*np.array([rad_adjoint_eig_funcs[i](l) for i in xrange(len(om_squared))])

    start_state = cr.Function(lambda z: 0., domain=(0, l))
    initial_weights = cr.project_on_initial_functions(start_state, rad_adjoint_eig_funcs)
    t, q = sim.simulate_state_space(A, B, u, initial_weights, temporal_domain, time_step=1e-3)

nodes = np.linspace(0, l, spatial_disc)
x_zt = cr.back_project_weights_matrix(q, rad_eig_funcs, nodes, order=0)
d_x_zt = cr.back_project_weights_matrix(q, rad_eig_funcs, nodes, order=1)

# display results
pd1 = vis.EvalData([t, nodes], x_zt)
pd2 = vis.EvalData([t, nodes], d_x_zt)
app = pg.QtGui.QApplication([])
win = vis.AnimatedPlot([pd1, pd2], title="Test")
win2 = vis.SurfacePlot(pd1)
app.exec_()
del app
