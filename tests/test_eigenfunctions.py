import sys
import unittest

import matplotlib.pyplot as plt
import numpy as np

from pyinduct import core as cr, utils as ut, eigenfunctions as ef

if any([arg in {'discover', 'setup.py', 'test'} for arg in sys.argv]):
    show_plots = False
else:
    # show_plots = True
    show_plots = False

if show_plots:
    import pyqtgraph as pg

    app = pg.QtGui.QApplication([])


class TestAddMulFunction(unittest.TestCase):
    def test_it(self):
        A = np.diag(np.ones(3))
        b = np.array(
            [ef.AddMulFunction(lambda z: z), ef.AddMulFunction(lambda z: 2 * z), ef.AddMulFunction(lambda z: 3 * z)])
        x = np.dot(b, A)
        self.assertAlmostEqual([4, 40, 300], [x[0](4), x[1](20), x[2](100)])

class TextSecondOrderEigenfunction(unittest.TestCase):
    def test_error_raiser(self):
        param = [1, 1, 1, 1, 1]
        l = 1
        n = 10
        eig_val, eig_funcs = ef.SecondOrderDirichletEigenfunction.solve_evp_hint(param, l, scale=np.ones(n))
        eig_freq = ef.SecondOrderDirichletEigenfunction.eigval_tf_eigfreq(param, eig_val=eig_val)
        _, _ = ef.SecondOrderDirichletEigenfunction.solve_evp_hint(param, l, n)
        _, _ = ef.SecondOrderDirichletEigenfunction.solve_evp_hint(param, l, n=n, scale=np.ones(n))
        _, _ = ef.SecondOrderDirichletEigenfunction.solve_evp_hint(param, l, eig_val=eig_val, scale=np.ones(n))
        _, _ = ef.SecondOrderDirichletEigenfunction.solve_evp_hint(param, l, eig_freq=eig_freq, scale=np.ones(n))

        with self.assertRaises(ValueError):
            _, _ = ef.SecondOrderDirichletEigenfunction.solve_evp_hint(param, l, n, scale=np.ones(n+1))
        with self.assertRaises(ValueError):
            _, _ = ef.SecondOrderDirichletEigenfunction.solve_evp_hint(param, l, eig_val=eig_val, scale=np.ones(n+1))
        with self.assertRaises(ValueError):
            _, _ = ef.SecondOrderDirichletEigenfunction.solve_evp_hint(param, l, n, eig_freq=eig_freq)
        with self.assertRaises(ValueError):
            _, _ = ef.SecondOrderDirichletEigenfunction.solve_evp_hint(param, l, eig_val=eig_val, eig_freq=eig_freq)
        with self.assertRaises(ValueError):
            _, _ = ef.SecondOrderDirichletEigenfunction.solve_evp_hint(param, l, n, eig_val=eig_val, eig_freq=eig_freq)


class FiniteTransformTest(unittest.TestCase):
    def setUp(self):
        # self.nested_lambda = True
        self.nested_lambda = False

    def test_trivial(self):

        l = 5.
        k = 5
        b_desired = 0
        k1, k2, b = ut.split_domain(k, b_desired, l, mode='coprime')[0:3]
        A = ut.get_inn_domain_transformation_matrix(k1, k2, mode="2n")
        self.assertAlmostEqual(b, 0)
        self.assertTrue(all(np.isclose(A, np.linalg.inv(A)).all(1)))
        b_desired = l
        k1, k2, b = ut.split_domain(k, b_desired, l, mode='coprime')[0:3]
        B = ut.get_inn_domain_transformation_matrix(k1, k2, mode="2n")
        self.assertAlmostEqual(b, l)
        self.assertTrue(all(np.isclose(B, np.diag(np.ones(B.shape[0]))).all(1)))
        A = ut.get_inn_domain_transformation_matrix(k1, k2, mode="2n")

    def test_paper_example(self):

        l = 5.
        k = 5
        b_desired = 2
        k1, k2, b = ut.split_domain(k, b_desired, l, mode='coprime')[0:3]
        M = np.linalg.inv(ut.get_inn_domain_transformation_matrix(k1, k2, mode="2n"))
        func = lambda z: np.cos(z)
        shifted_func = ef.FiniteTransformFunction(func, M, l, nested_lambda=self.nested_lambda)
        z = np.linspace(0, l, 1e3)
        if show_plots:
            for i in [0]:
                plt.figure()
                plt.plot(z, shifted_func(z))
                plt.plot(z, func(z))
            plt.show()

    def test_const(self):

        param = [2., 1.5, -3., -1., -.5]
        l = 5.
        spatial_domain = (0, l)
        n = 1
        k = 5
        b_desired = 2
        k1, k2, b = ut.split_domain(k, b_desired, l, mode='coprime')[0:3]
        M = np.linalg.inv(ut.get_inn_domain_transformation_matrix(k1, k2, mode="2n"))
        eig_freq, eig_val = ef.SecondOrderRobinEigenfunction.eigfreq_eigval_hint(param, l, n, show_plot=show_plots)
        eig_funcs = np.array([ef.SecondOrderRobinEigenfunction(om, param, l) for om in eig_freq])
        shifted_eig_funcs = np.array(
            [ef.FiniteTransformFunction(func, M, l, nested_lambda=self.nested_lambda) for func in eig_funcs])
        z = np.linspace(0, l, 1e3)
        if show_plots:
            for i in range(n):
                plt.figure()
                plt.plot(z, shifted_eig_funcs[i](z))
                plt.plot(z, eig_funcs[i](z))
            plt.show()

    def test_segmentation_fault(self):

        if show_plots:
            plt.figure()
            fun_end = list()
            for k in [5, 7, 9, 11, 13, 15, 17, 19]:
                param = [2., 1.5, -3., -1., -.5]
                l = 5.
                spatial_domain = (0, l)
                n = 1
                b_desired = 2
                k1, k2, b = ut.split_domain(k, b_desired, l, mode='coprime')[0:3]
                M = np.linalg.inv(ut.get_inn_domain_transformation_matrix(k1, k2, mode="2n"))
                eig_freq, eig_val = ef.SecondOrderRobinEigenfunction.eigfreq_eigval_hint(param, l, n)
                eig_funcs = np.array([ef.SecondOrderRobinEigenfunction(om, param, l) for om in eig_freq])
                shifted_eig_funcs = np.array(
                    [ef.FiniteTransformFunction(func, M, l, nested_lambda=self.nested_lambda) for func in eig_funcs])
                z = np.linspace(0, l, 1e3)
                y = shifted_eig_funcs[0](z)
                self.assertLess(max(np.diff(y)), 0.1)
                plt.plot(z, y, label=str(k) + " " + str(b))
                plt.plot(z, eig_funcs[0](z))
            plt.legend()
            plt.show()


class TestEigenvalues(unittest.TestCase):
    def test_dirichlet(self):
        desired_eig_freq = [(i + 1) * np.pi for i in range(4)]
        eig_freq, _ = ef.SecondOrderDirichletEigenfunction.eigfreq_eigval_hint([1, 2, 3, None, None], 1, 4)
        self.assertTrue(all(np.isclose(eig_freq, desired_eig_freq)))

    def test_robin(self):
        param_desired_ef_pairs = [([1, 0, 1, -2, -2], [2.39935728j, 0, 5.59677209, 8.98681892]),
                                  ([1, 0, 1, 0, 0], [0j, 3.14159265, 6.28318531, 9.42477796]),
                                  ([1, 2, 1, 3, 4], [2.06301691, 4.46395118, 7.18653501, 10.09113552]),
                                  ([1, -6, 0, -5, -5], [8.000003j, 1.84683426j, 4.86945051, 8.43284888])]
        for param, desired_eig_freq in param_desired_ef_pairs:
            eig_freq, _ = ef.SecondOrderRobinEigenfunction.eigfreq_eigval_hint(param, 1, 4)
            self.assertTrue(all(np.isclose(eig_freq, desired_eig_freq)))


class TestSecondOrderRobinEigenvalueProblemFuctions(unittest.TestCase):
    def setUp(self):

        self.param = [2, 1.5, -3, -5, -.5]
        a2, a1, a0, alpha, beta = self.param
        l = 1
        spatial_domain = (0, l)
        self.z = np.linspace(0, l, 1e2)
        self.n = 10

        eig_freq, self.eig_val = ef.SecondOrderRobinEigenfunction.eigfreq_eigval_hint(self.param, l, self.n,
                                                                                      show_plot=show_plots)
        self.eig_funcs = np.array([ef.SecondOrderRobinEigenfunction(om, self.param, l) for om in eig_freq])
        self.a2_z = lambda z: a2
        self.a1_z = a1
        self.a0_z = lambda z: a0
        self.alpha = alpha
        self.beta = beta
        self.transformed_eig_funcs = [ef.TransformedSecondOrderEigenfunction(self.eig_val[i], [self.eig_funcs[i](0),
                                                                                               self.eig_funcs[i].derive(
                                                                                                   1)(0), 0, 0],
                                                                             [self.a2_z, self.a1_z, self.a0_z], self.z)
                                      for i in range(len(self.eig_funcs))]

    def test_constant_coefficient(self):

        a2, a1, a0, alpha, beta = self.param
        z = self.z
        if show_plots:
            plt.figure()
        for i in range(len(self.eig_funcs)):
            eig_v = self.eig_val[i]
            eig_f = self.eig_funcs[i]
            if show_plots:
                plt.plot(self.z, eig_f.derive(1)(self.z))
            self.assertTrue(all(
                np.isclose(a2 * eig_f.derive(2)(z) + a1 * eig_f.derive(1)(z) + a0 * eig_f(z), eig_v.real * eig_f(z))))
            self.assertTrue(np.isclose(eig_f.derive(1)(self.z[0]), self.alpha * eig_f(self.z[0])))
            self.assertTrue(np.isclose(eig_f.derive(1)(self.z[-1]), - self.beta * eig_f(self.z[-1])))
        if show_plots:
            plt.show()

    def test_spatially_varying_coefficient(self):

        # TODO: provide second derivative of transformed eigenfunctions
        for i in range(len(self.eig_funcs)):
            eig_f = self.transformed_eig_funcs[i]
            eig_v = self.eig_val[i]
            self.assertTrue(all(np.isclose(
                self.a2_z(self.z) * self.eig_funcs[i].derive(2)(self.z) + self.a1_z * eig_f.derive(1)(
                    self.z) + self.a0_z(self.z) * eig_f(self.z), eig_v.real * eig_f(self.z), rtol=1e-3)))
            self.assertTrue(np.isclose(eig_f.derive(1)(self.z[0]), self.alpha * eig_f(self.z[0]), atol=1e-4))
            self.assertTrue(np.isclose(eig_f.derive(1)(self.z[-1]), - self.beta * eig_f(self.z[-1]), atol=1e-4))


class IntermediateTransformationTest(unittest.TestCase):
    def test_it(self):
        # original system parameters
        a2 = 1.5
        a1 = 2.5
        a0 = 28
        alpha = -2
        beta = -3
        self.param = [a2, a1, a0, alpha, beta]
        adjoint_param = ef.SecondOrderEigenfunction.get_adjoint_problem(self.param)

        # target system parameters (controller parameters)
        a1_t = -5
        a0_t = -25
        alpha_t = 3
        beta_t = 2
        # a1_t = a1; a0_t = a0; alpha_t = alpha; beta_t = beta
        self.param_t = [a2, a1_t, a0_t, alpha_t, beta_t]

        # original intermediate ("_i") and traget intermediate ("_ti") system parameters
        _, _, a0_i, self.alpha_i, self.beta_i = ef.transform_to_intermediate(self.param)
        self.param_i = a2, 0, a0_i, self.alpha_i, self.beta_i
        _, _, a0_ti, self.alpha_ti, self.beta_ti = ef.transform_to_intermediate(self.param_t)
        self.param_ti = a2, 0, a0_ti, self.alpha_ti, self.beta_ti

        # system/simulation parameters
        self.l = 1
        self.spatial_domain = (0, self.l)
        self.spatial_disc = 30
        self.n = 10

        # create (not normalized) eigenfunctions
        self.eig_freq, self.eig_val = ef.SecondOrderRobinEigenfunction.eigfreq_eigval_hint(self.param, self.l, self.n)
        init_eig_funcs = np.array([ef.SecondOrderRobinEigenfunction(om, self.param, self.l) for om in self.eig_freq])
        init_adjoint_eig_funcs = np.array(
            [ef.SecondOrderRobinEigenfunction(om, adjoint_param, self.l) for om in self.eig_freq])

        # normalize eigenfunctions and adjoint eigenfunctions
        self.eig_funcs, self.adjoint_eig_funcs = cr.normalize_base(init_eig_funcs, init_adjoint_eig_funcs)

        # eigenvalues and -frequencies test
        eig_freq_i, eig_val_i = ef.SecondOrderRobinEigenfunction.eigfreq_eigval_hint(self.param_i, self.l, self.n)
        self.assertTrue(all(np.isclose(self.eig_val, eig_val_i)))
        calc_eig_freq = np.sqrt((a0_i - eig_val_i) / a2)
        self.assertTrue(all(np.isclose(calc_eig_freq, eig_freq_i)))

        # intermediate (_i) eigenfunction test
        eig_funcs_i = np.array(
            [ef.SecondOrderRobinEigenfunction(eig_freq_i[i], self.param_i, self.l, self.eig_funcs[i](0)) for i in
             range(self.n)])
        self.assertTrue(all(np.isclose([func(0) for func in eig_funcs_i], [func(0) for func in self.eig_funcs])))
        test_vec = np.linspace(0, self.l, 100)
        for i in range(self.n):
            self.assertTrue(all(
                np.isclose(self.eig_funcs[i](test_vec), eig_funcs_i[i](test_vec) * np.exp(-a1 / 2 / a2 * test_vec))))


class ReturnRealPartTest(unittest.TestCase):
    def test_it(self):
        self.assertTrue(np.isreal(ef.return_real_part(1)))
        self.assertTrue(np.isreal(ef.return_real_part(1 + 0j)))
        self.assertTrue(np.isreal(ef.return_real_part(1 + 1e-20j)))
        self.assertRaises(TypeError, ef.return_real_part, None)
        self.assertRaises(TypeError, ef.return_real_part, (1, 2., 2 + 2j))
        self.assertRaises(TypeError, ef.return_real_part, [None, 2., 2 + 2j])
        self.assertRaises(ValueError, ef.return_real_part, [1, 2., 2 + 2j])
        self.assertRaises(ValueError, ef.return_real_part, 1 + 1e-10j)
        self.assertRaises(ValueError, ef.return_real_part, 1j)
