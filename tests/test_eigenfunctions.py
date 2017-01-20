import sys
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pyinduct as pi
import pyinduct.parabolic as parabolic

if any([arg in {'discover', 'setup.py', 'test'} for arg in sys.argv]):
    show_plots = False
else:
    show_plots = True

if show_plots:
    import pyqtgraph as pg
    pg.mkQApp()


class TestAddMulFunction(unittest.TestCase):
    def test_it(self):
        a_mat = np.diag(np.ones(3))
        b = np.array(
            [pi.AddMulFunction(lambda z: z), pi.AddMulFunction(lambda z: 2 * z), pi.AddMulFunction(lambda z: 3 * z)])
        x = np.dot(b, a_mat)
        self.assertAlmostEqual([4, 40, 300], [x[0](4), x[1](20), x[2](100)])


class TestSecondOrderEigenfunction(unittest.TestCase):
    def test_error_raiser(self):
        param = [1, 1, 1, 1, 1]
        l = 1
        n = 10
        eig_val, eig_funcs = pi.SecondOrderDirichletEigenfunction.solve_evp_hint(param, l, scale=np.ones(n))
        eig_freq = pi.SecondOrderDirichletEigenfunction.eigval_tf_eigfreq(param, eig_val=eig_val)
        _, _ = pi.SecondOrderDirichletEigenfunction.solve_evp_hint(param, l, n)
        _, _ = pi.SecondOrderDirichletEigenfunction.solve_evp_hint(param, l, n=n, scale=np.ones(n))
        _, _ = pi.SecondOrderDirichletEigenfunction.solve_evp_hint(param, l, eig_val=eig_val, scale=np.ones(n))
        _, _ = pi.SecondOrderDirichletEigenfunction.solve_evp_hint(param, l, eig_freq=eig_freq, scale=np.ones(n))

        with self.assertRaises(ValueError):
            _, _ = pi.SecondOrderDirichletEigenfunction.solve_evp_hint(param, l, n, scale=np.ones(n+1))
        with self.assertRaises(ValueError):
            _, _ = pi.SecondOrderDirichletEigenfunction.solve_evp_hint(param, l, eig_val=eig_val, scale=np.ones(n+1))
        with self.assertRaises(ValueError):
            _, _ = pi.SecondOrderDirichletEigenfunction.solve_evp_hint(param, l, n, eig_freq=eig_freq)
        with self.assertRaises(ValueError):
            _, _ = pi.SecondOrderDirichletEigenfunction.solve_evp_hint(param, l, eig_val=eig_val, eig_freq=eig_freq)
        with self.assertRaises(ValueError):
            _, _ = pi.SecondOrderDirichletEigenfunction.solve_evp_hint(param, l, n, eig_val=eig_val, eig_freq=eig_freq)


class FiniteTransformTest(unittest.TestCase):
    def setUp(self):
        # self.nested_lambda = True
        self.nested_lambda = False

    def test_trivial(self):
        l = 5
        k = 5
        b_desired = 0
        k1, k2, b = parabolic.control.split_domain(k, b_desired, l, mode='coprime')[0:3]
        a_mat = parabolic.general.get_in_domain_transformation_matrix(k1, k2, mode="2n")
        self.assertAlmostEqual(b, 0)
        self.assertTrue(all(np.isclose(a_mat, np.linalg.inv(a_mat)).all(1)))
        b_desired = l
        k1, k2, b = parabolic.control.split_domain(k, b_desired, l, mode='coprime')[0:3]
        b_mat = parabolic.general.get_in_domain_transformation_matrix(k1, k2, mode="2n")
        self.assertAlmostEqual(b, l)
        self.assertTrue(all(np.isclose(b_mat, np.diag(np.ones(b_mat.shape[0]))).all(1)))

        a_mat = parabolic.general.get_in_domain_transformation_matrix(k1, k2, mode="2n")
        # TODO add test here

    def test_paper_example(self):
        l = 5
        k = 5
        b_desired = 2
        k1, k2, b = parabolic.control.split_domain(k,
                                                   b_desired,
                                                   l,
                                                   mode='coprime')[0:3]
        m_mat = np.linalg.inv(
            parabolic.general.get_in_domain_transformation_matrix(k1,
                                                                  k2,
                                                                  mode="2n"))
        shifted_func = pi.FiniteTransformFunction(
            np.cos,
            m_mat,
            l,
            nested_lambda=self.nested_lambda)

        z = np.linspace(0, l, 1000)
        if show_plots:
            for i in [0]:
                plt.figure()
                plt.plot(z, shifted_func(z))
                plt.plot(z, np.cos(z))
            plt.show()

    def test_const(self):

        n = 5
        k = 5
        b_desired = 2
        l = 5

        params = pi.SecondOrderOperator(a2=2., a1=1.5, a0=-3.,
                                        alpha1=1, alpha0=1.,
                                        beta1=1, beta0=-.5)
        limits = (0, l)

        k1, k2, b = parabolic.control.split_domain(k,
                                                   b_desired,
                                                   l,
                                                   mode='coprime')[0:3]
        M = np.linalg.inv(
            parabolic.general.get_in_domain_transformation_matrix(k1,
                                                                  k2,
                                                                  mode="2n"))

        if 0:
            p = [params.a2, params.a1, params.a0, -params.alpha0, params.beta0]
            eig_freq, eig_val = pi.SecondOrderRobinEigenfunction.eigfreq_eigval_hint(
                p, l, n, show_plot=show_plots)
            eig_base = pi.Base(
                [pi.SecondOrderRobinEigenfunction(om, p, limits[-1])
                 for om in eig_freq])
            shifted_eig_base = pi.Base([pi.FiniteTransformFunction(
                func, M, l, nested_lambda=self.nested_lambda)
                                         for func in eig_base.fractions])

        else:
            nodes, eig_base = pi.cure_interval(pi.SecondOrderEigenVector,
                                               interval=limits,
                                               node_count=n,
                                               params=params,
                                               count=n,
                                               derivative_order=2,
                                               debug=True)

            shifted_eig_base = pi.Base([pi.FiniteTransformFunction(
                func, M, l, nested_lambda=self.nested_lambda)
                                         for func in eig_base.fractions])

        if show_plots:
            pi.visualize_functions(eig_base.fractions, 1e3)
            pi.visualize_functions(shifted_eig_base.fractions, 1e3)

    @unittest.skip
    def test_segmentation_fault(self):
        """
        I can't see the difference to test_const
        """

        if show_plots:
            plt.figure()
            fun_end = list()
            for k in [5, 7, 9, 11, 13, 15, 17, 19]:
                param = [2., 1.5, -3., -1., -.5]
                l = 5.
                spatial_domain = (0, l)
                n = 1
                b_desired = 2
                k1, k2, b = parabolic.control.split_domain(k,
                                                           b_desired,
                                                           l,
                                                           mode='coprime')[0:3]
                M = np.linalg.inv(
                    parabolic.general.get_in_domain_transformation_matrix(
                        k1, k2, mode="2n"))
                eig_freq, eig_val = pi.SecondOrderRobinEigenfunction.eigfreq_eigval_hint(param, l, n)
                eig_funcs = np.array(
                    [pi.SecondOrderRobinEigenfunction(om,
                                                      param,
                                                      spatial_domain[-1])
                     for om in eig_freq])

                shifted_eig_funcs = np.array([pi.FiniteTransformFunction(
                    func,
                    M,
                    l,
                    nested_lambda=self.nested_lambda)
                                              for func in eig_funcs])

                z = np.linspace(0, l, 1000)
                y = shifted_eig_funcs[0](z)
                self.assertLess(max(np.diff(y)), 0.1)
                plt.plot(z, y, label=str(k) + " " + str(b))
                plt.plot(z, eig_funcs[0](z))
            plt.legend()
            plt.show()


def calc_dirichlet_eigenvalues(params):
    """
    Estimate the eigenvalues of a 2nd order dirichlet problem .
    by approximating it using polynomial shapefunctions.
    """
    spat_dom, lag_base = pi.cure_interval(pi.LagrangeNthOrder,
                                          interval=params.domain,
                                          order=3,
                                          node_count=31)
    pi.register_base("fem_base", lag_base)

    old_params = [params.a2, params.a1, params.a0, -params.alpha0, params.beta0]
    weak_form = pi.parabolic.get_parabolic_dirichlet_weak_form("fem_base",
                                                               "fem_base",
                                                               None,
                                                               old_params,
                                                               params.domain)
    can_form = pi.parse_weak_formulation(weak_form, finalize=True)
    ss_form = pi.create_state_space(can_form)
    sys_mat = ss_form.A[1]
    eig_vals, eig_vecs = np.linalg.eig(sys_mat)
    real_idx = np.where(np.imag(eig_vals) == 0)
    abs_idx = np.argsort(np.abs(eig_vals[real_idx]))
    filtered_vals = eig_vals[real_idx][abs_idx]
    print(filtered_vals)

    return filtered_vals


class TestSecondOrderEigenVector(unittest.TestCase):

    def setUp(self):
        self.domain = pi.Domain(bounds=(0, 1), num=100)
        self.cnt = 10

        self.params_dirichlet = pi.SecondOrderOperator(a2=1,
                                                       a1=0,
                                                       a0=1,
                                                       alpha1=0,
                                                       alpha0=1,
                                                       beta1=0,
                                                       beta0=1,
                                                       domain=(0, 1))

        if 1:
            self.eig_dirichlet = None
            self.p_dirichlet = [(1j*n * np.pi, -1j * n * np.pi)
                                for n in range(1, self.cnt + 1)]

        else:
            self.eig_dirichlet = \
                calc_dirichlet_eigenvalues(self.params_dirichlet)[:self.cnt]
            self.p_dirichlet = \
                pi.SecondOrderEigenVector.convert_to_characteristic_root(
                    self.params_dirichlet,
                    self.eig_dirichlet
                )

        self.params_neumann = pi.SecondOrderOperator(a2=1,
                                                     a1=0,
                                                     a0=1,
                                                     alpha1=1,
                                                     alpha0=0,
                                                     beta1=1,
                                                     beta0=0)
        self.eig_neumann = None
        self.p_neumann = None
        # self.p_neumann = np.array([0, np.pi, 2 * np.pi, 3 * np.pi],
        #                           dtype=complex)

        self.params_robin = pi.Parameters(a2=1,
                                          a1=0,
                                          a0=1,
                                          alpha1=1,
                                          alpha0=2,
                                          beta1=1,
                                          beta0=-2)
        self.eig_robin = None
        self.p_robin = None
        # self.p_robin = np.array([(2.39935728j, -2.39935728j,),
        #                          (5.59677209j, -5.59677209j),
        #                          (8.98681892j, -8.98681892j)])

    def test_dirichlet(self):
        print("dirichlet case")
        self._test_helper(self.params_dirichlet,
                          self.eig_dirichlet,
                          self.p_dirichlet)

    def test_neumann(self):
        print("neumann case")
        self._test_helper(self.params_neumann,
                          self.eig_neumann,
                          self.p_neumann)

    def test_robin(self):
        print("robin case")
        self._test_helper(self.params_robin,
                          self.eig_robin,
                          self.p_robin)

    def _test_helper(self, params, l_ref, p_ref):
        eig_values, eig_base = pi.SecondOrderEigenVector.cure_hint(
            self.domain,
            params,
            count=self.cnt,
            derivative_order=2,
            debug=False)

        if show_plots:
            pi.visualize_functions(eig_base.fractions)

        # test eigenvalues
        self.assertEqual(len(eig_values), self.cnt)
        if l_ref is not None:
            np.testing.assert_array_equal(eig_values, l_ref, verbose=True)

        char_roots = pi.SecondOrderEigenVector.convert_to_characteristic_root(
            params,
            eig_values)
        if p_ref is not None:
            np.testing.assert_array_almost_equal(char_roots, p_ref,
                                                 decimal=4, verbose=True)

        # test eigenvectors
        for fraction, lam in zip(eig_base.fractions, eig_values):
            # test whether the operator is satisfied
            left = (params.a2 * fraction.derive(2)(self.domain.points)
                    + params.a1 * fraction.derive(1)(self.domain.points)
                    + params.a0 * fraction(self.domain.points))
            right = lam * fraction(self.domain.points)
            np.testing.assert_array_almost_equal(left, right, verbose=True)

            # test whether the bcs are fulfilled
            bc1 = (params.alpha0 * fraction(self.domain.bounds[0])
                   + params.alpha1 * fraction.derive(1)(self.domain.bounds[0]))
            bc2 = (params.beta0 * fraction(self.domain.bounds[1])
                   + params.beta1 * fraction.derive(1)(self.domain.bounds[1]))

            np.testing.assert_array_almost_equal(bc1, 0, decimal=5)
            np.testing.assert_array_almost_equal(bc2, 0, decimal=5)

        # check if they are orthonormal
        product_mat = pi.calculate_scalar_product_matrix(pi.dot_product_l2,
                                                         eig_base,
                                                         eig_base)
        np.testing.assert_array_almost_equal(product_mat,
                                             np.eye(self.cnt))

        return eig_base


class TestEigenvalues(unittest.TestCase):
    def test_dirichlet(self):
        desired_eig_freq = [(i + 1) * np.pi for i in range(4)]
        eig_freq, _ = pi.SecondOrderDirichletEigenfunction.eigfreq_eigval_hint(
            [1, 2, 3, None, None],
            1,
            4)
        self.assertTrue(all(np.isclose(eig_freq, desired_eig_freq)))

    def test_robin(self):
        param_desired_ef_pairs = [
            ([1, 0, 1, -2, -2], [2.39935728j, 0, 5.59677209, 8.98681892]),
            ([1, 0, 1, 0, 0], [0j, 3.14159265, 6.28318531, 9.42477796]),
            ([1, 2, 1, 3, 4], [2.06301691, 4.46395118, 7.18653501, 10.09113552]),
            ([1, -6, 0, -5, -5], [8.000003j, 1.84683426j, 4.86945051, 8.43284888])]

        for param, desired_eig_freq in param_desired_ef_pairs:
            eig_freq, _ = pi.SecondOrderRobinEigenfunction.eigfreq_eigval_hint(
                param, 1, 4, show_plot=True)
            self.assertTrue(all(np.isclose(eig_freq, desired_eig_freq)))


class TestSecondOrderRobinEigenvalueProblemFunctions(unittest.TestCase):
    def setUp(self):
        self.param = [2, 1.5, -3, -5, -.5]
        a2, a1, a0, alpha, beta = self.param
        l = 1
        limits = (0, l)

        self.spatial_domain = pi.Domain(limits, num=100)
        self.n = 10

        eig_freq, self.eig_val \
            = pi.SecondOrderRobinEigenfunction.eigfreq_eigval_hint(
            self.param,
            l,
            self.n,
            show_plot=show_plots)

        self.eig_funcs = np.array([pi.SecondOrderRobinEigenfunction(om,
                                                                    self.param,
                                                                    l)
                                   for om in eig_freq])

        self.a2_z = pi.Function.from_constant(a2)
        self.a1_z = pi.Function.from_constant(a1)
        self.a0_z = pi.Function.from_constant(a0)

        self.alpha = alpha
        self.beta = beta

        self.transformed_eig_funcs = [pi.TransformedSecondOrderEigenfunction(
            self.eig_val[i],
            [self.eig_funcs[i](0), self.eig_funcs[i].derive(1)(0), 0, 0],
            [self.a2_z, self.a1_z, self.a0_z],
            self.spatial_domain)
                                      for i in range(len(self.eig_funcs))]

    def test_constant_coefficient(self):
        """
        Transform an operator whose coefficients do not vary spatially
        """
        a2, a1, a0, alpha, beta = self.param
        z = self.spatial_domain.points

        if show_plots:
            plt.figure()

        for i in range(len(self.eig_funcs)):
            eig_v = self.eig_val[i]
            eig_f = self.eig_funcs[i]

            if show_plots:
                plt.plot(z, eig_f.derive(1)(z))

            # check transient behaviour
            self.assertTrue(np.allclose(a2 * eig_f.derive(2)(z)
                                        + a1 * eig_f.derive(1)(z)
                                        + a0 * eig_f(z),
                                        eig_v.real * eig_f(z)))

            # check boundaries
            self.assertTrue(np.isclose(eig_f.derive(1)(z[0]),
                                       self.alpha * eig_f(z[0])))
            self.assertTrue(np.isclose(eig_f.derive(1)(z[-1]),
                                       - self.beta * eig_f(z[-1])))

        if show_plots:
            plt.show()

    def test_spatially_varying_coefficient(self):
        """
        Transform an operator whose coefficients do vary spatially
        """
        # TODO: provide second derivative of transformed eigenfunctions
        for i in range(len(self.eig_funcs)):
            eig_f = self.transformed_eig_funcs[i]
            eig_v = self.eig_val[i]

            # interval
            self.assertTrue(np.allclose(
                self.a2_z(self.z) * self.eig_funcs[i].derive(2)(self.z)
                + self.a1_z(self.z) * eig_f.derive(1)(self.z)
                + self.a0_z(self.z) * eig_f(self.z),
                eig_v.real * eig_f(self.z),
                rtol=1e-3))

            # boundaries
            self.assertTrue(np.isclose(eig_f.derive(1)(self.z[0]),
                                       self.alpha * eig_f(self.z[0]),
                                       atol=1e-4))
            self.assertTrue(np.isclose(eig_f.derive(1)(self.z[-1]),
                                       -self.beta * eig_f(self.z[-1]),
                                       atol=1e-4))


class IntermediateTransformationTest(unittest.TestCase):
    def test_it(self):
        # original system parameters
        a2 = 1.5
        a1 = 2.5
        a0 = 28
        alpha = -2
        beta = -3
        self.param = [a2, a1, a0, alpha, beta]
        adjoint_param = pi.SecondOrderEigenfunction.get_adjoint_problem(self.param)

        # target system parameters (controller parameters)
        a1_t = -5
        a0_t = -25
        alpha_t = 3
        beta_t = 2
        # a1_t = a1; a0_t = a0; alpha_t = alpha; beta_t = beta
        self.param_t = [a2, a1_t, a0_t, alpha_t, beta_t]

        # original intermediate ("_i") and target intermediate ("_ti") system parameters
        _, _, a0_i, self.alpha_i, self.beta_i = parabolic.general.eliminate_advection_term(self.param)
        self.param_i = a2, 0, a0_i, self.alpha_i, self.beta_i
        _, _, a0_ti, self.alpha_ti, self.beta_ti = parabolic.general.eliminate_advection_term(self.param_t)
        self.param_ti = a2, 0, a0_ti, self.alpha_ti, self.beta_ti

        # system/simulation parameters
        self.l = 1
        self.spatial_domain = (0, self.l)
        self.spatial_disc = 30
        self.n = 10

        # create (not normalized) eigenfunctions
        self.eig_freq, self.eig_val = \
            pi.SecondOrderRobinEigenfunction.eigfreq_eigval_hint(self.param,
                                                                 self.l,
                                                                 self.n)
        init_eig_base = pi.Base([pi.SecondOrderRobinEigenfunction(om,
                                                                  self.param,
                                                                  self.spatial_domain[-1])
                                 for om in self.eig_freq])
        init_adjoint_eig_funcs = pi.Base([pi.SecondOrderRobinEigenfunction(om,
                                                                           adjoint_param,
                                                                           self.spatial_domain[-1])
                                          for om in self.eig_freq])

        # normalize eigenfunctions and adjoint eigenfunctions
        self.eig_base, self.adjoint_eig_funcs = pi.normalize_base(init_eig_base, init_adjoint_eig_funcs)

        # eigenvalues and -frequencies test
        eig_freq_i, eig_val_i = pi.SecondOrderRobinEigenfunction.eigfreq_eigval_hint(self.param_i, self.l, self.n)
        self.assertTrue(all(np.isclose(self.eig_val, eig_val_i)))
        calc_eig_freq = np.sqrt((a0_i - eig_val_i) / a2)
        self.assertTrue(all(np.isclose(calc_eig_freq, eig_freq_i)))

        # intermediate (_i) eigenfunction test
        eig_funcs_i = np.array([pi.SecondOrderRobinEigenfunction(eig_freq_i[i], self.param_i, self.spatial_domain[-1],
                                                                 self.eig_base.fractions[i](0))
                                for i in range(self.n)])
        self.assertTrue(all(np.isclose([func(0) for func in eig_funcs_i],
                                       [func(0) for func in self.eig_base.fractions])))
        test_vec = np.linspace(0, self.l, 100)
        for i in range(self.n):
            self.assertTrue(all(np.isclose(self.eig_base.fractions[i](test_vec),
                                           eig_funcs_i[i](test_vec) * np.exp(-a1 / 2 / a2 * test_vec))))
