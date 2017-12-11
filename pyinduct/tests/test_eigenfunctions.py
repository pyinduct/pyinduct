import unittest

import matplotlib.pyplot as plt
import numpy as np
import pyinduct as pi
import pyinduct.parabolic as parabolic
from pyinduct.tests import show_plots


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
    def test_trivial(self):
        l = 5
        k = 5

        k1, k2, b = parabolic.control.split_domain(k, 0, l, mode='coprime')[0:3]
        a_mat = parabolic.general.get_in_domain_transformation_matrix(
            k1, k2, mode="2n")
        self.assertAlmostEqual(b, 0)
        self.assertTrue(all(np.isclose(a_mat, np.linalg.inv(a_mat)).all(1)))

        k1, k2, b = parabolic.control.split_domain(k, l, l, mode='coprime')[0:3]
        b_mat = parabolic.general.get_in_domain_transformation_matrix(
            k1, k2, mode="2n")
        self.assertAlmostEqual(b, l)
        self.assertTrue(
            all(np.isclose(b_mat, np.diag(np.ones(b_mat.shape[0]))).all(1)))

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
            nested_lambda=False)
        shifted_func_nl = pi.FiniteTransformFunction(
            np.cos,
            m_mat,
            l,
            nested_lambda=True)

        z = np.linspace(0, l, 1000)
        np.testing.assert_array_almost_equal(
            shifted_func(z), shifted_func_nl(z))

        if show_plots:
            plt.figure()
            plt.plot(z, shifted_func(z))
            plt.plot(z, np.cos(z))
            plt.show()

    def test_const(self):
        n = 5
        k = 5
        b_desired = 2
        l = 5
        params = [2, 1.5, -3, 1, .5]

        k1, k2, b = parabolic.control.split_domain(k,
                                                   b_desired,
                                                   l,
                                                   mode='coprime')[0:3]
        M = np.linalg.inv(
            parabolic.general.get_in_domain_transformation_matrix(k1,
                                                                  k2,
                                                                  mode="2n"))

        eig_val, eig_base = pi.SecondOrderRobinEigenfunction.solve_evp_hint(
            params, l, n)
        shifted_eig_base = pi.Base(np.array(
            [pi.FiniteTransformFunction(
                func, M, l, nested_lambda=False)
                for func in eig_base]))
        shifted_eig_base_nl = pi.Base(np.array(
            [pi.FiniteTransformFunction(
                func, M, l, nested_lambda=True)
                for func in eig_base]))

        z = np.linspace(0, l, 1000)
        for f1, f2 in zip(shifted_eig_base, shifted_eig_base_nl):
            np.testing.assert_array_almost_equal(f1(z), f2(z))

        if show_plots:
            pi.visualize_functions(eig_base.fractions, 1000)
            pi.visualize_functions(shifted_eig_base.fractions, 1000)


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
            # TODO make computation by approximation work to check to other two
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

        # if show_plots:
        #     pi.visualize_functions(eig_base.fractions)

        # test eigenvalues
        self.assertEqual(len(eig_values), self.cnt)
        if l_ref is not None:
            np.testing.assert_array_equal(eig_values, l_ref, verbose=True)

        char_roots = pi.SecondOrderEigenVector.convert_to_characteristic_root(
            params,
            eig_values)
        if p_ref is not None:
            print(char_roots)
            print(p_ref)
            np.testing.assert_array_almost_equal(char_roots, p_ref,
                                                 decimal=5, verbose=True)

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
                param, 1, 4, show_plot=False)
            self.assertTrue(all(np.isclose(eig_freq, desired_eig_freq)))


class TestSecondOrderEigenvalueProblemFunctions(unittest.TestCase):
    def setUp(self):
        self.param = [2, 1.5, -3, -5, -.5]
        self.spatial_domain = pi.Domain((0, 1), num=100)
        self.z = self.spatial_domain.points
        self.n = 10

    def evp_eq(self, a2, a1, a0, boundary_check):
        for eig_v, eig_f in zip(self.eig_val, self.eig_funcs):
            np.testing.assert_array_almost_equal(
                (a2 * eig_f.derive(2)(self.z)
                 + a1 * eig_f.derive(1)(self.z)
                 + a0 * eig_f(self.z)) / eig_v,
                 eig_v.real * eig_f(self.z) / eig_v,
            decimal=4)
            boundary_check(eig_v, eig_f, self.z[-1])


    @unittest.skip("not implemented")
    def test_dirichlet_robin_constant_coefficient(self):

        def boundary_check(eig_v, eig_f, l):
            np.testing.assert_array_almost_equal(eig_f(0) / eig_v, 0)
            np.testing.assert_array_almost_equal(eig_f.derive(1)(l) / eig_v,
                                                 -beta * eig_f(l) / eig_v)

        a2, a1, a0, _, beta = self.param
        param = [a2, a1, a0, None, beta]

        eig_freq, self.eig_val \
            = pi.SecondOrderDiriRobEigenfunction.eigfreq_eigval_hint(
            param, self.z[-1], self.n, show_plot=True)

        _, self.eig_funcs = pi.SecondOrderDiriRobEigenfunction.solve_evp_hint(
            param, self.z[-1], eig_freq=eig_freq)

        [plt.plot(self.z, func(self.z)) for func in self.eig_funcs]
        plt.show()

        self.evp_eq(a2, a1, a0, boundary_check)

        self.spatially_varying_coefficient(boundary_check)


    @unittest.skip("not implemented")
    def test_robin_dirichlet_constant_coefficient(self):

        def boundary_check(eig_v, eig_f, l):
            np.testing.assert_array_almost_equal(eig_f.derive(1)(0) / eig_v,
                                                 alpha * eig_f(0) / eig_v)
            np.testing.assert_array_almost_equal(eig_f(l) / eig_v, 0)

        a2, a1, a0, alpha, _ = self.param
        param = [a2, a1, a0, alpha, None]

        eig_freq, self.eig_val \
            = pi.SecondOrderRobDiriEigenfunction.eigfreq_eigval_hint(
            param, self.z[-1], self.n, show_plot=True)

        _, self.eig_funcs = pi.SecondOrderRobDiriEigenfunction.solve_evp_hint(
            param, self.z[-1], eig_freq=eig_freq)

        [plt.plot(self.z, func(self.z)) for func in self.eig_funcs]
        plt.show()

        self.evp_eq(a2, a1, a0, boundary_check)

        self.spatially_varying_coefficient(boundary_check)


    def test_dirichlet_constant_coefficient(self):

        def boundary_check(eig_v, eig_f, l):
            np.testing.assert_array_almost_equal(eig_f(0) / eig_v, 0)
            np.testing.assert_array_almost_equal(eig_f(l) / eig_v, 0)

        a2, a1, a0, _, _ = self.param
        param = [a2, a1, a0, None, None]
        z = self.spatial_domain.points

        eig_freq, self.eig_val \
            = pi.SecondOrderDirichletEigenfunction.eigfreq_eigval_hint(
            param, self.z[-1], self.n)

        _, self.eig_funcs = pi.SecondOrderDirichletEigenfunction.solve_evp_hint(
            param, self.z[-1], eig_freq=eig_freq)

        self.evp_eq(a2, a1, a0, boundary_check)

        self.spatially_varying_coefficient(boundary_check)


    def test_robin_constant_coefficient(self):

        def boundary_check(eig_v, eig_f, l):
            np.testing.assert_array_almost_equal(eig_f.derive(1)(0) / eig_v,
                                                 alpha * eig_f(0) / eig_v)
            np.testing.assert_array_almost_equal(eig_f.derive(1)(l) / eig_v,
                                                 - beta * eig_f(l) / eig_v)

        a2, a1, a0, alpha, beta = self.param

        eig_freq, self.eig_val \
            = pi.SecondOrderRobinEigenfunction.eigfreq_eigval_hint(
            self.param,
            self.z[-1],
            self.n,
            show_plot=show_plots)

        self.eig_funcs = np.array([pi.SecondOrderRobinEigenfunction(om,
                                                                    self.param,
                                                                    self.z[-1])
                                   for om in eig_freq])

        self.evp_eq(a2, a1, a0, boundary_check)

        self.spatially_varying_coefficient(boundary_check)

        if show_plots:
            plt.show()

    def spatially_varying_coefficient(self, boundary_check):
        a2, a1, a0, _, _ = self.param
        a2_z = pi.Function.from_constant(a2)
        a1_z = pi.Function.from_constant(a1)
        a0_z = pi.Function.from_constant(a0)

        transformed_eig_funcs = [pi.TransformedSecondOrderEigenfunction(
            self.eig_val[i],
            [self.eig_funcs[i](0), self.eig_funcs[i].derive(1)(0), 0, 0],
            [a2_z, a1_z, a0_z],
            self.spatial_domain)
                                      for i in range(len(self.eig_funcs))]
        # TODO: provide second derivative of transformed eigenfunctions
        for i in range(len(self.eig_funcs)):
            eig_f = transformed_eig_funcs[i]
            eig_v = self.eig_val[i]

            # interval
            np.testing.assert_array_almost_equal(
                a2_z(self.spatial_domain) * self.eig_funcs[i].derive(2)(self.spatial_domain)
                + a1_z(self.spatial_domain) * eig_f.derive(1)(self.spatial_domain)
                + a0_z(self.spatial_domain) * eig_f(self.spatial_domain),
                eig_v.real * eig_f(self.spatial_domain),
                decimal=2)

            boundary_check(eig_v, eig_f, self.spatial_domain[-1])


class IntermediateTransformationTest(unittest.TestCase):
    def test_it(self):

        # system/simulation parameters
        self.l = 1
        self.spatial_domain = (0, self.l)
        self.spatial_disc = 30
        self.n = 10

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
        _, _, a0_i, self.alpha_i, self.beta_i =\
            parabolic.general.eliminate_advection_term(self.param, self.l)
        self.param_i = a2, 0, a0_i, self.alpha_i, self.beta_i
        _, _, a0_ti, self.alpha_ti, self.beta_ti =\
            parabolic.general.eliminate_advection_term(self.param_t, self.l)
        self.param_ti = a2, 0, a0_ti, self.alpha_ti, self.beta_ti

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
