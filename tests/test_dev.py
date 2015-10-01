from __future__ import division
import unittest
import numpy as np
import pyqtgraph as pg
from pyinduct import core as cr, simulation as sim, utils as ut, visualization as vis, trajectory as tr
import pyinduct.placeholder as ph

__author__ = 'marcus'

actuation = 'dirichlet'
boundary_condition = 'dirichlet'
param = [1., -2., -1., None, None]
adjoint_param = ut.get_adjoint_rad_dirichlet_evp_param(param)
a2, a1, a0, _, _ = param
l = 1.; spatial_domain = (0, l); spatial_disc = 50
T = 1.; temporal_domain = (0, T); temporal_disc = 1e3
n = 10

omega = np.array([(i+1)*np.pi/l for i in xrange(n)])
eig_values = a0 - a2*omega**2 - a1**2/4./a2
norm_fak = np.ones(omega.shape)*np.sqrt(2)
rad_eig_funcs = np.array([ut.ReaAdvDifDirichletEigenfunction(omega[i], param, spatial_domain, norm_fak[i]) for i in range(n)])
rad_adjoint_eig_funcs = np.array([ut.ReaAdvDifDirichletEigenfunction(omega[i], adjoint_param, spatial_domain, norm_fak[i]) for i in range(n)])

# derive initial field variable x(z,0) and weights
start_state = cr.Function(lambda z: 0., domain=(0, l))
initial_weights = cr.project_on_initial_functions(start_state, rad_adjoint_eig_funcs)

u = tr.ReaAdvDifTrajectory(l, T, param, boundary_condition, actuation)

## determine (A,B) with weak-formulation (pyinduct)
# integral terms
int1 = ph.IntegralTerm(ph.Product(ph.TemporalDerivedFieldVariable(rad_eig_funcs, order=1),
                                  ph.TestFunctions(rad_adjoint_eig_funcs, order=0)), spatial_domain)
int2 = ph.IntegralTerm(ph.Product(ph.SpatialDerivedFieldVariable(rad_eig_funcs, order=0),
                                  ph.TestFunctions(rad_adjoint_eig_funcs, order=2)), spatial_domain, -a2)
int3 = ph.IntegralTerm(ph.Product(ph.SpatialDerivedFieldVariable(rad_eig_funcs, order=0),
                                  ph.TestFunctions(rad_adjoint_eig_funcs, order=1)), spatial_domain, a1)
int4 = ph.IntegralTerm(ph.Product(ph.SpatialDerivedFieldVariable(rad_eig_funcs, order=0),
                                  ph.TestFunctions(rad_adjoint_eig_funcs, order=0)), spatial_domain, -a0)
# scalar terms
s1 = ph.ScalarTerm(ph.Product(ph.Input(u),
                              ph.TestFunctions(rad_adjoint_eig_funcs, order=1, location=l)), a2)
s2 = ph.ScalarTerm(ph.Product(ph.Input(u),
                              ph.TestFunctions(rad_adjoint_eig_funcs, order=0, location=l)), -a1)
# derive sate-space system
rad_pde = sim.WeakFormulation([int1, int2, int3, int4, s1, s2])
cf = sim.parse_weak_formulation(rad_pde)
ss_weak = cf.convert_to_state_space()

## determine (A,B) with modal-transfomation
A = np.diag(eig_values)
B = -a2*np.array([rad_adjoint_eig_funcs[i].derive()(l) for i in xrange(n)])
ss_modal = sim.StateSpace(A,B)

self.assertTrue(np.allclose(np.sort(np.linalg.eigvals(ss_weak.A)), np.sort(np.linalg.eigvals(ss_modal.A)), rtol=3e-01, atol=0.))
self.assertTrue(np.allclose(np.array([i[0] for i in ss_weak.B]), ss_modal.B))

t, q = sim.simulate_state_space(ss_modal, u, initial_weights, temporal_domain, time_step=T/temporal_disc)



nodes = np.linspace(0, l, spatial_disc)
x_zt = cr.back_project_weights_matrix(q, rad_eig_funcs, nodes, order=0)
d_x_zt = cr.back_project_weights_matrix(q, rad_eig_funcs, nodes, order=1)

# display results
pd1 = vis.EvalData([t, nodes], x_zt)
pd2 = vis.EvalData([t, nodes], d_x_zt)
self.app = pg.QtGui.QApplication([])
win = vis.AnimatedPlot([pd1, pd1], title="Test")
win2 = vis.SurfacePlot(pd1)
self.app.exec_()
del self.app