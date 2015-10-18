from __future__ import division
import unittest
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyinduct import get_initial_functions, register_functions, \
    core as cr, \
    utils as ut, \
    eigenfunctions as ef, \
    visualization as vt, \
    placeholder as ph
import sys
import pyqtgraph as pg


__author__ = 'marcus'


if any([arg == 'discover' for arg in sys.argv]):
    show_plots = False
else:
    show_plots = True
    app = pg.QtGui.QApplication([])


class TestSecondOrderRobinEigenvalueProblemFuctions(unittest.TestCase):

    def setUp(self):

        self.param = [2., 1.5, -3., -1., -.5]
        a2, a1, a0, alpha, beta = self.param
        l = 1.; spatial_domain = (0, l); self.z = np.linspace(0, l, 1e2)
        self.n = 10

        eig_freq, self.eig_val = ut.compute_rad_robin_eigenfrequencies(self.param, l, self.n)
        self.eig_funcs = np.array([ut.SecondOrderRobinEigenfunction(om, self.param, spatial_domain) for om in eig_freq])
        self.a2_z = lambda z: a2
        self.a1_z = a1
        self.a0_z = lambda z: a0
        self.transformed_eig_funcs = [ef.TransformedSecondOrderEigenfunction(self.eig_val[i],
                                                                            [self.eig_funcs[i](0), self.eig_funcs[i].derive(1)(0), 0, 0],
                                                                            [self.a2_z, self.a1_z, self.a0_z],
                                                                            self.z)
                                      for i in range(len(self.eig_funcs)) ]

    def test_constant_coefficient(self):

        a2, a1, a0, alpha, beta = self.param
        z = self.z
        for i in range(len(self.eig_funcs)):
            eig_v = self.eig_val[i]
            eig_f = self.eig_funcs[i]
            self.assertTrue(all(np.isclose(a2*eig_f.derive(2)(z) +
                                           a1*eig_f.derive(1)(z) +
                                           a0*eig_f(z),
                                           eig_v.real*eig_f(z))))

    def test_spatially_varying_coefficient(self):

        # TODO: provide second derivative of transformed eigenfunctions
        for i in range(len(self.eig_funcs)):
            eig_f = self.transformed_eig_funcs[i]
            eig_v = self.eig_val[i]
            self.assertTrue(all(np.isclose(self.a2_z(self.z)*self.eig_funcs[i].derive(2)(self.z) +
                                           self.a1_z*eig_f.derive(1)(self.z) +
                                           self.a0_z(self.z)*eig_f(self.z),
                                           eig_v.real*eig_f(self.z),
                                           rtol=1e-3)))
