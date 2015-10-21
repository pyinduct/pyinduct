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


class TestAddMulFunction(unittest.TestCase):

    def test_it(self):

        A = np.diag(np.ones(3))
        b = np.array([ef.AddMulFunction(lambda z: z), ef.AddMulFunction(lambda z: 2*z), ef.AddMulFunction(lambda z: 3*z)])
        x = np.dot(b, A)
        self.assertAlmostEqual([4, 40, 300], [x[0](4), x[1](20), x[2](100)])


class FiniteTransformTest(unittest.TestCase):

    def test_paper_example(self):

        l = 5.
        k = 5
        b_desired = 2
        k1, k2, b = ut.split_domain(k, b_desired, l, mode='coprime')[0:3]
        M = np.linalg.inv(ut.get_inn_domain_transformation_matrix(k1, k2, mode="2n"))
        func = lambda z: np.cos(z)
        shifted_func = ef.FiniteTransformFunction(func, M, b, l)
        z = np.linspace(0, l, 1e3)
        if show_plots:
            for i in [0]:
                plt.figure()
                plt.plot(z, shifted_func(z))
                plt.plot(z, func(z))
            plt.show()

    def test_it(self):

        param = [2., 1.5, -3., -1., -.5]
        l = 5.; spatial_domain = (0, l)
        n = 1
        k = 5
        b_desired = 2
        k1, k2, b = ut.split_domain(k, b_desired, l, mode='coprime')[0:3]
        M = np.linalg.inv(ut.get_inn_domain_transformation_matrix(k1, k2, mode="2n"))
        eig_freq, eig_val = ef.compute_rad_robin_eigenfrequencies(param, l, n)
        eig_funcs = np.array([ef.SecondOrderRobinEigenfunction(om, param, spatial_domain) for om in eig_freq])
        shifted_eig_funcs = np.array([ef.FiniteTransformFunction(func, M, b, l) for func in eig_funcs])
        z = np.linspace(0, l, 1e3)
        if show_plots:
            for i in range(n):
                plt.figure()
                plt.plot(z, shifted_eig_funcs[i](z))
                plt.plot(z, eig_funcs[i](z))
            plt.show()

    def test_var(self):

        if show_plots:
            plt.figure()
            fun_end = list()
            for k in [5, 7, 9, 11, 13, 15, 17, 19]:
                param = [2., 1.5, -3., -1., -.5]
                l = 5.; spatial_domain = (0, l)
                n = 1
                b_desired = 2
                k1, k2, b = ut.split_domain(k, b_desired, l, mode='coprime')[0:3]
                M = np.linalg.inv(ut.get_inn_domain_transformation_matrix(k1, k2, mode="2n"))
                eig_freq, eig_val = ef.compute_rad_robin_eigenfrequencies(param, l, n)
                eig_funcs = np.array([ef.SecondOrderRobinEigenfunction(om, param, spatial_domain) for om in eig_freq])
                shifted_eig_funcs = np.array([ef.FiniteTransformFunction(func, M, b, l) for func in eig_funcs])
                z = np.linspace(0, l, 1e3)
                plt.plot(z, shifted_eig_funcs[0](z), label=str(b))
                plt.plot(z, eig_funcs[0](z))
            plt.legend()
            plt.show()


class TestSecondOrderRobinEigenvalueProblemFuctions(unittest.TestCase):

    def setUp(self):

        self.param = [2., 1.5, -3., -1., -.5]
        a2, a1, a0, alpha, beta = self.param
        l = 1.; spatial_domain = (0, l); self.z = np.linspace(0, l, 1e2)
        self.n = 10

        eig_freq, self.eig_val = ef.compute_rad_robin_eigenfrequencies(self.param, l, self.n)
        self.eig_funcs = np.array([ef.SecondOrderRobinEigenfunction(om, self.param, spatial_domain) for om in eig_freq])
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


class ReturnRealPartTest(unittest.TestCase):

    def test_it(self):
        self.assertTrue(np.isreal(ef.return_real_part(1)))
        self.assertTrue(np.isreal(ef.return_real_part(1+0j)))
        self.assertTrue(np.isreal(ef.return_real_part(1+1e-20j)))
        self.assertRaises(TypeError, ef.return_real_part, None)
        self.assertRaises(TypeError, ef.return_real_part, (1, 2., 2+2j))
        self.assertRaises(TypeError, ef.return_real_part, [None, 2., 2+2j])
        self.assertRaises(ValueError, ef.return_real_part, [1, 2., 2+2j])
        self.assertRaises(ValueError, ef.return_real_part, 1+1e-10j)
        self.assertRaises(ValueError, ef.return_real_part, 1j)
