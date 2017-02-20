import unittest

import numpy as np
from scipy import integrate
import pyinduct as pi
import pyinduct.control as control
import pyinduct.parabolic as parabolic
from tests import show_plots

if show_plots:
    import pyqtgraph as pg

    app = pg.QtGui.QApplication([])


class CollocatedTestCase(unittest.TestCase):
    def setUp(self):
        spat_dom = pi.Domain((0, 1), 10)
        nodes, base = pi.cure_interval(pi.LagrangeFirstOrder,
                                       spat_dom.bounds,
                                       3)
        pi.register_base("base", base)

        x = pi.FieldVariable("base")
        x_dt_at1 = x.derive(temp_order=1)(1)
        x_dz_at0 = x.derive(spat_order=1)(0)

        exp_base = pi.Base(pi.Function(np.exp))
        pi.register_base("exp_base", exp_base, overwrite=True)
        exp_at1 = pi.ScalarFunction("exp_base")(1)

        alpha = 2
        self.term1 = pi.ScalarTerm(x_dt_at1, 1 + alpha)
        self.term2 = pi.ScalarTerm(x_dz_at0, 2)
        self.term3 = pi.ScalarTerm(pi.Product(x(1), exp_at1))

        self.weight_label = "base"
        self.weights = np.array([1, 1, 1, 2, 2, 2])

    def _build_case(self, term):
        ce = pi.parse_weak_formulation(pi.WeakFormulation([term], name="test"),
                                       finalize=False)
        evaluator = pi.LawEvaluator(ce)
        return evaluator(self.weights, self.weight_label)["output"]

    def test_temp_term(self):
        res = self._build_case(self.term1)
        self.assertAlmostEqual(res, 6)

    def test_spat_term(self):
        res = self._build_case(self.term2)
        self.assertAlmostEqual(res, 0)

    def test_product_term(self):
        res = self._build_case(self.term3)
        self.assertAlmostEqual(res, 1 * np.exp(1))

    def tearDown(self):
        pi.deregister_base("base")
        pi.deregister_base("exp_base")


class ContinuousTestCase(unittest.TestCase):
    def setUp(self):
        self.weight_label = "base"

        interval = (0, 1)
        nodes, funcs = pi.cure_interval(pi.LagrangeFirstOrder, interval, 3)
        pi.register_base(self.weight_label, funcs)

        x = pi.FieldVariable(self.weight_label)
        x_dt = x.derive(temp_order=1)
        x_dz = x.derive(spat_order=1)

        pi.register_base("scalar_func", pi.Base(pi.Function(np.exp)))
        exp = pi.ScalarFunction("scalar_func")

        alpha = 2
        self.term1 = pi.IntegralTerm(x_dt, interval, 1 + alpha)
        self.term2 = pi.IntegralTerm(x_dz, interval, 2)
        self.term3 = pi.IntegralTerm(pi.Product(x, exp), interval)

        self.weights = np.array([1, 1, 1, 2, 2, 2])

    def _build_case(self, term):
        ce = pi.parse_weak_formulation(pi.WeakFormulation([term], name="test"),
                                       finalize=False)
        evaluator = pi.LawEvaluator(ce)
        return evaluator(self.weights, self.weight_label)["output"]

    def test_temp_term(self):
        res = self._build_case(self.term1)
        self.assertTrue(np.equal(res, 6))

    def test_spat_term(self):
        res = self._build_case(self.term2)
        self.assertAlmostEqual(res, 0)

    def test_product_term(self):
        res = self._build_case(self.term3)
        # TODO calculate expected result
        # self.assertAlmostEqual(res, 1*np.exp(1))

    def tearDown(self):
        pi.deregister_base(self.weight_label)
        pi.deregister_base("scalar_func")


class RadDirichletControlApproxTest(unittest.TestCase):
    def setUp(self):
        # original system parameters
        a2 = 1
        a1 = 0  # attention: only a2 = 1., a1 =0 supported in this test case
        a0 = 0
        self.param = [a2, a1, a0, None, None]

        # target system parameters (controller parameters)
        # attention: only a2 = 1., a1 =0 and a0 =0 supported in this test case
        a1_t = 0
        a0_t = 0
        self.param_t = [a2, a1_t, a0_t, None, None]

        # system/simulation parameters
        self.actuation_type = 'dirichlet'
        self.bound_cond_type = 'dirichlet'

        self.l = 1.
        spatial_disc = 10
        self.dz = pi.Domain(bounds=(0, self.l), num=spatial_disc)

        self.t_end = 1.
        temporal_disc = 100
        self.dt = pi.Domain(bounds=(0, self.t_end), num=temporal_disc)

        self.modal_order = 10

        # eigenvalues /-functions of the original system
        eig_freq, self.eig_values = \
            pi.SecondOrderDirichletEigenfunction.eigfreq_eigval_hint(
                self.param,
                self.l,
                self.modal_order)
        norm_fac = np.ones(eig_freq.shape) * np.sqrt(2)
        self.eig_base = pi.Base([pi.SecondOrderDirichletEigenfunction(
            eig_freq[i],
            self.param,
            self.dz.bounds[-1],
            scale=norm_fac[i])
                                 for i in range(self.modal_order)])
        pi.register_base("eig_base", self.eig_base)

        # eigenfunctions of the target system
        eig_freq_t = np.sqrt(-self.eig_values.astype(complex))
        norm_fac_t = norm_fac * eig_freq / eig_freq_t
        self.eig_base_t = pi.Base([pi.SecondOrderDirichletEigenfunction(
            eig_freq_t[i],
            self.param_t,
            self.dz.bounds[-1],
            norm_fac_t[i])
                                   for i in range(self.modal_order)])
        pi.register_base("eig_base_t", self.eig_base_t)

    def test_controller(self):
        # derive initial field variable x(z,0) and weights
        start_state = pi.Function.from_constant(0, domain=(0, self.l))
        initial_weights = pi.project_on_base(start_state, self.eig_base)

        # init trajectory / input of target system
        trajectory = parabolic.RadTrajectory(self.l,
                                             self.t_end,
                                             self.param_t,
                                             self.bound_cond_type,
                                             self.actuation_type)

        # init controller
        x_at_1 = pi.FieldVariable("eig_base", location=1)
        xt_at_1 = pi.FieldVariable("eig_base_t",
                                   weight_label="eig_base",
                                   location=1)
        controller = pi.Controller(
            pi.WeakFormulation([pi.ScalarTerm(x_at_1, 1),
                                pi.ScalarTerm(xt_at_1, -1)],
                               name="control_law"))

        # input with feedback
        control_law = pi.SimulationInputSum([trajectory, controller])

        # determine (A, B) with modal transformation
        a_mat = np.diag(self.eig_values)
        b_mat = -self.param[0] * np.atleast_2d(
            [fraction.derive()(self.l)
             for fraction in self.eig_base.fractions]).T
        ss = pi.StateSpace(a_mat, b_mat, base_lbl="eig_base",
                           input_handle=control_law)

        # simulate
        t, q = pi.simulate_state_space(ss, initial_weights, self.dt)

        eval_d = pi.evaluate_approximation("eig_base", q, t, self.dz)
        x_0t = eval_d.output_data[:, 0]
        yc, tc = pi.gevrey_tanh(self.t_end, 1)
        x_0t_desired = np.interp(t, tc, yc[0, :])
        self.assertLess(np.average((x_0t - x_0t_desired) ** 2), 0.5)

        # display results
        if show_plots:
            eval_d = pi.evaluate_approximation("eig_base", q, t, self.dz)
            win2 = pi.PgSurfacePlot(eval_d)
            app.exec_()

    def tearDown(self):
        pi.deregister_base("eig_base")
        pi.deregister_base("eig_base_t")


@unittest.skip
class RadRobinControlApproxTest(unittest.TestCase):
    """
    Conversion to new interface is in progress.
    """

    def setUp(self):
        # original system parameters
        self.l = 1
        domain = (0, self.l)
        a2 = 1.5
        a1 = 2.5
        a0 = 28
        alpha = -2
        beta = -3

        self.param = pi.SecondOrderOperator(a2=a2, a1=a1, a0=a0,
                                            alpha1=1, alpha0=-alpha,
                                            beta1=1, beta0=beta)
        self.adjoint_param = self.param.get_adjoint_problem()

        # target system parameters (controller parameters)
        a1_t = -5
        a0_t = -25
        alpha_t = 3
        beta_t = 2

        self.param_t = pi.SecondOrderOperator(a2=a2, a1=a1_t, a0=a0_t,
                                              alpha1=1, alpha0=-alpha_t,
                                              beta1=1, beta0=beta_t)

        # original intermediate ("_i") and target intermediate ("_ti")
        # system parameters
        self.param_i = parabolic.eliminate_advection_term(self.param)
        self.param_ti = parabolic.eliminate_advection_term(self.param_t)

        # system/simulation parameters
        self.actuation_type = 'robin'
        self.bound_cond_type = 'robin'
        spatial_disc = 10
        self.dz = pi.Domain(bounds=domain, num=spatial_disc)

        self.t_end = 1.
        temporal_disc = 100
        self.dt = pi.Domain(bounds=domain, num=temporal_disc)

        self.modal_order = 3

        # calculate eigenvalues and eigenvectors of original and adjoint system
        self.eig_values, _eig_base = pi.SecondOrderEigenVector.cure_hint(
            domain=self.dz,
            params=self.param,
            count=self.modal_order,
            derivative_order=2)
        self.adjoint_eig_values, _adjoint_eig_base = \
            pi.SecondOrderEigenVector.cure_hint(domain=self.dz,
                                                params=self.adjoint_param,
                                                count=self.modal_order,
                                                derivative_order=2)

        # pi.visualize_functions(_eig_base.fractions)
        # pi.visualize_functions(_adjoint_eig_base.fractions)

        # normalize
        self.eig_base, self.adjoint_eig_base = pi.normalize_base(
            _eig_base, _adjoint_eig_base
        )

        # pi.visualize_functions(self.eig_base.fractions)
        # pi.visualize_functions(self.adjoint_eig_base.fractions)

        # bases should be bi-orthonormal
        test_mat = pi.calculate_scalar_product_matrix(pi.dot_product_l2,
                                                      self.eig_base,
                                                      self.adjoint_eig_base)
        np.testing.assert_array_almost_equal(test_mat,
                                             np.eye(self.modal_order),
                                             decimal=5)

        # adjoint operator must have the same eigenvalues
        np.testing.assert_array_almost_equal(self.eig_values,
                                             self.adjoint_eig_values)

        if 0:
            # create (un-normalized) eigenfunctions
            self.eig_freq, self.eig_val = \
                pi.SecondOrderRobinEigenfunction.eigfreq_eigval_hint(
                    self.param,
                    self.l,
                    self.modal_order)

            init_eig_funcs = pi.Base(
                [pi.SecondOrderRobinEigenfunction(om, self.param, self.l)
                 for om in self.eig_freq])
            init_adjoint_eig_funcs = pi.Base(
                [pi.SecondOrderRobinEigenfunction(om, self.adjoint_param, self.l)
                 for om in self.eig_freq])

            # normalize eigenfunctions and adjoint eigenfunctions
            self.eig_base, self.adjoint_eig_base = pi.normalize_base(
                init_eig_funcs,
                init_adjoint_eig_funcs)

        # eigenfunctions of the target system ("_t")
        _eig_values_t, _eig_base_t = pi.SecondOrderEigenVector.cure_hint(
            domain=self.dz,
            params=self.param_t,
            count=self.modal_order,
            derivative_order=2)

        char_roots = pi.SecondOrderEigenVector.convert_to_characteristic_root(
            self.param,
            self.eig_values
        )

        eig_values_t = pi.SecondOrderEigenVector.convert_to_eigenvalue(
            self.param_t,
            char_roots
        )

        np.testing.assert_array_almost_equal(_eig_values_t, eig_values_t)

        scale_factors = np.divide(
            np.array([frac(0) for frac in self.eig_base.fractions]),
            np.array([frac(0) for frac in _eig_base_t.fractions]))
        self.eig_base_t = pi.Base([frac.scale(factor) for frac, factor in
                                   zip(_eig_base_t.fractions, scale_factors)])

        if 0:
            _eig_freq_t = np.sqrt(-a1_t ** 2 / 4 / a2 ** 2
                                  + (a0_t - self.eig_val) / a2)
            _roots_t = pi.SecondOrderEigenVector.convert_to_characteristic_root(
                self.param_t,
                _eig_freq_t)

            self.eig_base_t = pi.Base([pi.SecondOrderEigenVector(
                    eig_freq_t[i],
                    self.param_t,
                    self.l).scale(eig_base.fractions[i](0))
                 for i in range(self.modal_order)])

        # register eigenfunctions
        pi.register_base("eig_base", self.eig_base)
        pi.register_base("adjoint_eig_base", self.adjoint_eig_base)
        pi.register_base("eig_base_t", self.eig_base_t)

        # pi.visualize_functions(self.eig_base.fractions)
        # pi.visualize_functions(self.adjoint_eig_base.fractions)
        # pi.visualize_functions(self.eig_base_t.fractions)

    def test_controller(self):
        # controller initialization
        x = pi.FieldVariable("eig_base")
        x_at_l = x(self.l)
        xdz_at_l = x.derive(spat_order=1)(self.l)

        x_t = pi.FieldVariable("eig_base_t", weight_label="eig_base")
        x_t_at_l = x_t(self.l)
        xdz_t_at_l = x_t.derive(spat_order=1)(self.l)

        def combined_transform(z):
            return np.exp((self.param_t.a1
                           - self.param.a1) / 2 / self.param.a2 * z)

        def int_kernel_zz(z):
            return (-self.param_ti.alpha0
                    + self.param_i.alpha0
                    + (self.param_i.a0
                       - self.param_ti.a0) / 2 / self.param.a2 * z)

        law = pi.WeakFormulation([
            pi.ScalarTerm(x_at_l,
                          self.param_i.beta0
                          - self.param_ti.beta0
                          - int_kernel_zz(self.l)),
            pi.ScalarTerm(x_t_at_l,
                          -self.param_ti.beta0 * combined_transform(self.l)),
            pi.ScalarTerm(x_at_l, self.param_ti.beta0),
            pi.ScalarTerm(xdz_t_at_l, -combined_transform(self.l)),
            pi.ScalarTerm(x_t_at_l,
                          -self.param_t.a1 / 2 /
                          self.param.a2 * combined_transform(self.l)),
            pi.ScalarTerm(xdz_at_l, 1),
            pi.ScalarTerm(x_at_l,
                          self.param.a1 / 2 / self.param.a2
                          + int_kernel_zz(self.l))],
            name="Rad_Robin-Controller")
        controller = control.Controller(law)

        # init trajectory
        trajectory = parabolic.RadTrajectory(self.l,
                                             self.t_end,
                                             self.param_t,
                                             self.bound_cond_type,
                                             self.actuation_type)
        trajectory.scale = combined_transform(self.l)

        # input with feedback
        control_law = pi.SimulationInputSum([trajectory, controller])

        # determine (A,B) with modal-transformation
        a_mat = np.diag(np.real(self.eig_values))
        b_mat = self.param.a2 * np.atleast_2d(
            [fraction(self.l)
             for fraction in self.adjoint_eig_base.fractions]).T
        ss_modal = pi.StateSpace(a_mat, b_mat,
                                 base_lbl="adjoint_eig_base",
                                 input_handle=control_law)

        # derive initial field variable x(z,0) and weights
        start_state = pi.Function.from_constant(0, domain=(0, self.l))
        initial_weights = pi.project_on_base(start_state, self.adjoint_eig_base)

        # simulate
        t, q = pi.simulate_state_space(ss_modal, initial_weights, self.dt)

        eval_d = pi.evaluate_approximation("eig_base", q, t, self.dz)
        x_0t = eval_d.output_data[:, 0]
        yc, tc = pi.gevrey_tanh(self.t_end, 1)
        x_0t_desired = np.interp(t, tc, yc[0, :])
        self.assertLess(np.average((x_0t - x_0t_desired) ** 2), 1e-4)

        # display results
        if show_plots:
            win1 = pi.PgAnimatedPlot([eval_d], title="Test")
            win2 = pi.PgSurfacePlot(eval_d)
            app.exec_()

    def tearDown(self):
        pi.deregister_base("eig_base")
        pi.deregister_base("adjoint_eig_base")
        pi.deregister_base("eig_base_t")


class RadRobinGenericBacksteppingControllerTest(unittest.TestCase):
    """
    """

    def setUp(self):
        # original system parameters
        a2 = 1.5
        a1 = 2.5
        a0 = 28
        alpha = -2
        beta = -3
        self.param = [a2, a1, a0, alpha, beta]
        adjoint_param = pi.SecondOrderEigenfunction.get_adjoint_problem(
            self.param)

        # target system parameters (controller parameters)
        a1_t = -5
        a0_t = -25
        alpha_t = 3
        beta_t = 2
        # a1_t = a1; a0_t = a0; alpha_t = alpha; beta_t = beta
        self.param_t = [a2, a1_t, a0_t, alpha_t, beta_t]

        # original intermediate ("_i") and target intermediate ("_ti")
        # system parameters
        _, _, a0_i, self.alpha_i, self.beta_i = \
            pi.parabolic.eliminate_advection_term(self.param)
        self.param_i = a2, 0, a0_i, self.alpha_i, self.beta_i
        _, _, a0_ti, self.alpha_ti, self.beta_ti = \
            pi.parabolic.eliminate_advection_term(self.param_t)
        self.param_ti = a2, 0, a0_ti, self.alpha_ti, self.beta_ti

        # system/simulation parameters
        actuation_type = 'robin'
        bound_cond_type = 'robin'
        self.l = 1
        spatial_disc = 10
        self.dz = pi.Domain(bounds=(0, self.l), num=spatial_disc)

        self.T = 1
        temporal_disc = 100
        self.dt = pi.Domain(bounds=(0, self.T), num=temporal_disc)
        self.n = 10

        # create (not normalized) eigenfunctions
        eig_freq, self.eig_val = \
            pi.SecondOrderRobinEigenfunction.eigfreq_eigval_hint(self.param,
                                                                 self.l,
                                                                 self.n)

        init_eig_base = pi.Base([pi.SecondOrderRobinEigenfunction(om,
                                                                  self.param,
                                                                  self.l)
                                 for om in eig_freq])
        init_adjoint_eig_base = pi.Base(
            [pi.SecondOrderRobinEigenfunction(om, adjoint_param, self.l)
             for om in eig_freq])

        # normalize eigenfunctions and adjoint eigenfunctions
        eig_base, self.adjoint_eig_base =\
            pi.normalize_base(init_eig_base, init_adjoint_eig_base)

        # eigenfunctions from target system ("_t")
        eig_freq_t = np.sqrt(-a1_t ** 2 / 4 / a2 ** 2
                             + (a0_t - self.eig_val) / a2)
        eig_base_t = pi.Base(
            [pi.SecondOrderRobinEigenfunction(eig_freq_t[i],
                                              self.param_t,
                                              self.l).scale(frac(0))
             for i, frac in enumerate(eig_base.fractions)])

        # create test functions
        nodes, self.fem_base = pi.cure_interval(pi.LagrangeFirstOrder,
                                                self.dz.bounds,
                                                node_count=self.n)

        # register eigenfunctions
        pi.register_base("eig_base", eig_base)
        pi.register_base("eig_base_t", eig_base_t)
        # pi.register_base("adjoint_eig_base", self.adjoint_eig_base)
        pi.register_base("fem_base", self.fem_base)

        # original () and target (_t) field variable
        fem_field_variable = pi.FieldVariable("fem_base", location=self.l)

        field_variable = pi.FieldVariable("eig_base", location=self.l)
        d_field_variable = field_variable.derive(spat_order=1)

        field_variable_t = pi.FieldVariable("eig_base_t",
                                            weight_label="eig_base",
                                            location=self.l)
        d_field_variable_t = field_variable_t.derive(spat_order=1)

        # intermediate (_i) and target intermediate (_ti) transformations by z=l
        def _transform_i(z):
            """
            x_i  = x   * transform_i
            """
            return np.exp(a1 / 2 / a2 * z)

        self.transform_i = _transform_i

        def _transform_ti(z):
            """
            x_ti = x_t * transform_ti
            """
            return np.exp(a1_t / 2 / a2 * z)

        self.transform_ti = _transform_ti

        # intermediate (_i) and target intermediate (_ti) field variable
        # (list of scalar terms = sum of scalar terms)
        self.x_fem_i_at_l = [pi.ScalarTerm(fem_field_variable,
                                           self.transform_i(self.l))]

        self.x_i_at_l = [pi.ScalarTerm(field_variable,
                                       self.transform_i(self.l))]

        self.xd_i_at_l = [pi.ScalarTerm(d_field_variable,
                                        self.transform_i(self.l)),

                          pi.ScalarTerm(field_variable,
                                        self.transform_i(self.l) * a1 / 2 / a2)]

        self.x_ti_at_l = [pi.ScalarTerm(field_variable_t,
                                        self.transform_ti(self.l))]

        self.xd_ti_at_l = [pi.ScalarTerm(d_field_variable_t,
                                         self.transform_ti(self.l)),
                           pi.ScalarTerm(field_variable_t,
                                         self.transform_ti(self.l) * a1_t / 2 / a2)]

        # discontinuous operator (Kx)(t) = int_kernel_zz(l)*x(l,t)
        self.int_kernel_zz = lambda z: (self.alpha_ti - self.alpha_i
                                        + (a0_i - a0_ti) / 2 / a2 * z)

        # init trajectory
        self.traj = pi.parabolic.RadTrajectory(self.l,
                                               self.T,
                                               self.param_ti,
                                               bound_cond_type,
                                               actuation_type,
                                               scale=self.transform_i(-self.l))

    def test_fem(self):
        self.act_funcs = "fem_base"
        controller = parabolic.get_parabolic_robin_backstepping_controller(
            state=self.x_fem_i_at_l,
            approx_state=self.x_i_at_l,
            d_approx_state=self.xd_i_at_l,
            approx_target_state=self.x_ti_at_l,
            d_approx_target_state=self.xd_ti_at_l,
            integral_kernel_zz=self.int_kernel_zz( self.l),
            original_beta=self.beta_i,
            target_beta=self.beta_ti,
            scale=self.transform_i(-self.l))

        system_input = pi.SimulationInputSum([self.traj, controller])

        # determine (A,B) via approximation
        rad_pde, lbls = parabolic.get_parabolic_robin_weak_form(self.act_funcs,
                                                                self.act_funcs,
                                                                system_input,
                                                                self.param,
                                                                self.dz.bounds)
        cf = pi.parse_weak_formulation(rad_pde)
        ss_weak = pi.create_state_space(cf)

        # simulate
        self.t, self.q = pi.simulate_state_space(ss_weak,
                                                 np.zeros((self.n,)),
                                                 self.dt)

        eval_d = pi.evaluate_approximation(self.act_funcs,
                                           self.q,
                                           self.t,
                                           self.dz)
        x_0t = eval_d.output_data[:, 0]
        yc, tc = pi.gevrey_tanh(self.T, 1)
        x_0t_desired = np.interp(self.t, tc, yc[0, :])
        self.assertLess(np.average((x_0t - x_0t_desired) ** 2), 1e-3)

        # display results
        if show_plots:
            win1 = pi.PgAnimatedPlot([eval_d], title="Test")
            win2 = pi.PgSurfacePlot(eval_d)
            app.exec_()

        for lbl in lbls:
            pi.deregister_base(lbl)

    def test_modal(self):
        self.act_funcs = "eig_base"
        a2, a1, a0, alpha, beta = self.param
        controller = parabolic.get_parabolic_robin_backstepping_controller(
            state=self.x_i_at_l,
            approx_state=self.x_i_at_l,
            d_approx_state=self.xd_i_at_l,
            approx_target_state=self.x_ti_at_l,
            d_approx_target_state=self.xd_ti_at_l,
            integral_kernel_zz=self.int_kernel_zz(self.l),
            original_beta=self.beta_i,
            target_beta=self.beta_ti,
            scale=self.transform_i(-self.l))

        system_input = pi.SimulationInputSum([self.traj, controller])

        # determine (A,B) with modal transformation
        a_mat = np.diag(np.real(self.eig_val))
        b_mat = a2 * np.atleast_2d([self.adjoint_eig_base.fractions[i](self.l)
                                    for i in range(self.n)]).T

        ss_modal = pi.StateSpace(a_mat,
                                 b_mat,
                                 input_handle=system_input,
                                 base_lbl=self.act_funcs)

        # simulate
        self.t, self.q = pi.simulate_state_space(ss_modal,
                                                 np.zeros((self.n,)),
                                                 self.dt)

        eval_d = pi.evaluate_approximation(self.act_funcs,
                                           self.q,
                                           self.t,
                                           self.dz)
        x_0t = eval_d.output_data[:, 0]
        yc, tc = pi.gevrey_tanh(self.T, 1)
        x_0t_desired = np.interp(self.t, tc, yc[0, :])
        self.assertLess(np.average((x_0t - x_0t_desired) ** 2), 1e-2)

        # display results
        if show_plots:
            win1 = pi.PgAnimatedPlot([eval_d], title="Test")
            win2 = pi.PgSurfacePlot(eval_d)
            app.exec_()

    def tearDown(self):
        pi.deregister_base("eig_base")
        # pi.deregister_base("adjoint_eig_funcs")
        pi.deregister_base("eig_base_t")
        pi.deregister_base("fem_base")


class RadRobinSpatiallyVaryingCoefficientControllerTest(unittest.TestCase):
    """
    """

    @unittest.skip("This takes about 22 minutes on my machine, Travis will"
                   "quit after 10 minutes.")
    def test_it(self):
        # system/simulation parameters
        actuation_type = 'robin'
        bound_cond_type = 'robin'

        self.l = 1.
        spatial_disc = 10
        self.dz = pi.Domain(bounds=(0, self.l), num=spatial_disc)

        self.T = 1.
        temporal_disc = 10
        self.dt = pi.Domain(bounds=(0, self.T), num=temporal_disc)

        self.n = 10

        # original system parameters
        a2 = 1.5
        a2_z = pi.Function.from_constant(1.5)
        a1_z = pi.Function.from_constant(1)
        a0_z = pi.Function.from_constant(3)
        alpha = -2
        beta = -3
        self.param = [a2, a1_z, a0_z, alpha, beta]

        # target system parameters (controller parameters)
        a1_t = -0
        a0_t = -1
        alpha_t = 1
        beta_t = 1
        self.param_t = [a2, a1_t, a0_t, alpha_t, beta_t]
        adjoint_param_t =\
            pi.SecondOrderEigenfunction.get_adjoint_problem(self.param_t)

        # original intermediate ("_i") and
        # target intermediate ("_ti") system parameters
        _, _, a0_i, alpha_i, beta_i = \
            pi.parabolic.eliminate_advection_term(self.param, self.l)
        self.param_i = a2, 0, a0_i, alpha_i, beta_i
        _, _, a0_ti, alpha_ti, beta_ti = \
            pi.parabolic.eliminate_advection_term(self.param_t)
        self.param_ti = a2, 0, a0_ti, alpha_ti, beta_ti

        # create (not normalized) target (_t) eigenfunctions
        eig_freq_t, self.eig_val_t =\
            pi.SecondOrderRobinEigenfunction.eigfreq_eigval_hint(self.param_t,
                                                                 self.l,
                                                                 self.n)
        init_eig_funcs_t = pi.Base(
            [pi.SecondOrderRobinEigenfunction(om, self.param_t, self.l)
             for om in eig_freq_t])

        init_adjoint_eig_funcs_t = pi.Base(
            [pi.SecondOrderRobinEigenfunction(om, adjoint_param_t, self.l)
             for om in eig_freq_t])

        # normalize eigenfunctions and adjoint eigenfunctions
        eig_funcs_t, self.adjoint_eig_base_t =\
            pi.normalize_base(init_eig_funcs_t, init_adjoint_eig_funcs_t)

        # transformed original eigenfunctions
        self.eig_base = pi.Base(
            [pi.TransformedSecondOrderEigenfunction(
                self.eig_val_t[i],
                [eig_funcs_t.fractions[i](0),
                 alpha * eig_funcs_t.fractions[i](0),
                 0,
                 0],
                [a2_z, a1_z, a0_z],
                pi.Domain((0, self.l), num=1e4)
            ) for i in range(self.n)])

        # create test-functions
        nodes, self.fem_base = pi.cure_interval(pi.LagrangeFirstOrder,
                                                self.dz.bounds,
                                                node_count=self.n)

        # if show_plots:
        #     pi.visualize_functions(self.fem_base.fractions)
        #     pi.visualize_functions(self.eig_base.fractions)

        # register functions
        pi.register_base("eig_base", self.eig_base)
        pi.register_base("eig_base_t", eig_funcs_t)
        pi.register_base("fem_funcs", self.fem_base)

        # original () and target (_t) field variable
        fem_field_variable = pi.FieldVariable("fem_funcs", location=self.l)
        field_variable = pi.FieldVariable("eig_base", location=self.l)
        d_field_variable = field_variable.derive(spat_order=1)
        field_variable_t = pi.FieldVariable("eig_base_t",
                                            weight_label="eig_base",
                                            location=self.l)
        d_field_variable_t = field_variable_t.derive(spat_order=1)

        # intermediate (_i) and target intermediate (_ti) transformations by z=l

        #  x_i  = x   * transform_i_at_l
        self.transform_i_at_l = np.exp(
            integrate.quad(lambda z: a1_z(z) / 2 / a2, 0, self.l)[0])

        # x  = x_i   * inv_transform_i_at_l
        self.inv_transform_i_at_l = np.exp(
            -integrate.quad(lambda z: a1_z(z) / 2 / a2, 0, self.l)[0])

        # x_ti = x_t * transform_ti_at_l
        self.transform_ti_at_l = np.exp(a1_t / 2 / a2 * self.l)

        # intermediate (_i) and target intermediate (_ti) field variable
        # (list of scalar terms = sum of scalar terms)
        self.x_fem_i_at_l = [pi.ScalarTerm(fem_field_variable,
                                           self.transform_i_at_l)]

        self.x_i_at_l = [pi.ScalarTerm(field_variable, self.transform_i_at_l)]

        self.xd_i_at_l = [pi.ScalarTerm(d_field_variable,
                                        self.transform_i_at_l),
                          pi.ScalarTerm(field_variable,
                                        self.transform_i_at_l * a1_z(self.l) / 2 / a2)]

        self.x_ti_at_l = [pi.ScalarTerm(field_variable_t,
                                        self.transform_ti_at_l)]

        self.xd_ti_at_l = [pi.ScalarTerm(d_field_variable_t,
                                         self.transform_ti_at_l),
                           pi.ScalarTerm(field_variable_t,
                                         self.transform_ti_at_l * a1_t / 2 / a2)]

        # discontinuous operator (Kx)(t) = int_kernel_zz(l)*x(l,t)
        self.int_kernel_zz = alpha_ti - alpha_i + integrate.quad(
            lambda z: (a0_i(z) - a0_ti) / 2 / a2, 0, self.l)[0]

        controller = parabolic.get_parabolic_robin_backstepping_controller(
            state=self.x_fem_i_at_l,
            approx_state=self.x_i_at_l,
            d_approx_state=self.xd_i_at_l,
            approx_target_state=self.x_ti_at_l,
            d_approx_target_state=self.xd_ti_at_l,
            integral_kernel_zz=self.int_kernel_zz,
            original_beta=beta_i, target_beta=beta_ti,
            scale=self.inv_transform_i_at_l)

        # init feedforward
        self.traj = pi.parabolic.RadTrajectory(self.l,
                                               self.T,
                                               self.param_ti,
                                               bound_cond_type,
                                               actuation_type,
                                               scale=self.inv_transform_i_at_l)

        system_input = pi.SimulationInputSum([self.traj, controller])

        rad_pde, lbls = parabolic.get_parabolic_robin_weak_form("fem_funcs",
                                                                "fem_funcs",
                                                                system_input,
                                                                self.param,
                                                                self.dz.bounds)
        cf = pi.parse_weak_formulation(rad_pde)
        ss_weak = pi.create_state_space(cf)

        # simulate
        t, q = pi.simulate_state_space(ss_weak,
                                       np.zeros((self.n,)),
                                       self.dt)

        eval_d = pi.evaluate_approximation("fem_funcs", q, t, self.dz)
        x_0t = eval_d.output_data[:, 0]
        yc, tc = pi.gevrey_tanh(self.T, 1)
        x_0t_desired = np.interp(t, tc, yc[0, :])
        self.assertLess(np.average((x_0t - x_0t_desired) ** 2), 1e-4)

        # display results
        if show_plots:
            win1 = pi.PgAnimatedPlot([eval_d], title="Test")
            win2 = pi.PgSurfacePlot(eval_d)
            app.exec_()

        for lbl in lbls:
            pi.deregister_base(lbl)

        pi.deregister_base("eig_base")
        pi.deregister_base("eig_base_t")
        pi.deregister_base("fem_funcs")


class RadRobinInDomainBacksteppingControllerTest(unittest.TestCase):
    """
    """
    def test_fem(self):

        # system/simulation parameters
        actuation_type = 'robin'
        bound_cond_type = 'robin'

        self.l = 1.
        spatial_disc = 30
        self.dz = pi.Domain(bounds=(0, self.l), num=spatial_disc)

        self.T = 1.
        temporal_disc = 100
        self.dt = pi.Domain(bounds=(0, self.T), num=temporal_disc)
        self.n = 12

        # original system parameters
        a2 = 1.5
        a1 = 2.5
        a0 = 28
        alpha = -2
        beta = -3
        self.param = [a2, a1, a0, alpha, beta]
        adjoint_param = pi.SecondOrderEigenfunction.get_adjoint_problem(
            self.param)

        # target system parameters (controller parameters)
        a1_t = -5
        a0_t = -25
        alpha_t = 3
        beta_t = 2
        self.param_t = [a2, a1_t, a0_t, alpha_t, beta_t]

        # actuation_type by b which is close to
        # b_desired on a k times subdivided spatial domain
        b_desired = self.l / 2
        k = 51  # = k1 + k2
        k1, k2, self.b = parabolic.split_domain(k,
                                                b_desired,
                                                self.l,
                                                mode='coprime')[0:3]
        m_mat = np.linalg.inv(
            parabolic.get_in_domain_transformation_matrix(k1, k2, mode="2n"))

        # original intermediate ("_i") and
        # target intermediate ("_ti") system parameters
        _, _, a0_i, self.alpha_i, self.beta_i =\
            pi.parabolic.eliminate_advection_term(self.param)
        self.param_i = a2, 0, a0_i, self.alpha_i, self.beta_i
        _, _, a0_ti, self.alpha_ti, self.beta_ti =\
            pi.parabolic.eliminate_advection_term(self.param_t)
        self.param_ti = a2, 0, a0_ti, self.alpha_ti, self.beta_ti

        # create (not normalized) eigenfunctions
        eig_freq, self.eig_val =\
            pi.SecondOrderRobinEigenfunction.eigfreq_eigval_hint(self.param,
                                                                 self.l,
                                                                 self.n)

        init_eig_funcs = pi.Base(
            [pi.SecondOrderRobinEigenfunction(om, self.param, self.l)
             for om in eig_freq])

        init_adjoint_eig_funcs = pi.Base(
            [pi.SecondOrderRobinEigenfunction(om, adjoint_param, self.l)
             for om in eig_freq])

        # normalize eigenfunctions and adjoint eigenfunctions
        eig_base, self.adjoint_eig_base = pi.normalize_base(
            init_eig_funcs, init_adjoint_eig_funcs)

        # eigenfunctions of the in-domain intermediate (_id)
        # and the intermediate (_i) system
        eig_freq_i, eig_val_i =\
            pi.SecondOrderRobinEigenfunction.eigfreq_eigval_hint(self.param_i,
                                                                 self.l,
                                                                 self.n)

        # eigenvalues should not be touched by this transformation
        self.assertTrue(all(np.isclose(eig_val_i, self.eig_val)))

        eig_base_id = pi.Base(
            [pi.SecondOrderRobinEigenfunction(eig_freq_i[i],
                                              self.param_i,
                                              self.l, fraction(0))
             for i, fraction in enumerate(eig_base.fractions)])

        eig_base_i = pi.Base(
            [pi.SecondOrderRobinEigenfunction(eig_freq_i[i],
                                              self.param_i,
                                              self.l,
                                              frac(0) * frac_id(self.l) /
                                              frac_id(self.b))
             for i, (frac, frac_id) in enumerate(zip(eig_base.fractions,
                                                     eig_base_id.fractions))])

        # eigenfunctions from target system ("_ti")
        eig_freq_ti = np.sqrt((a0_ti - self.eig_val) / a2)
        eig_base_ti = pi.Base(
            [pi.SecondOrderRobinEigenfunction(eig_freq_ti[i],
                                              self.param_ti,
                                              self.l,
                                              frac(0))
             for i, frac in enumerate(eig_base_i.fractions)])

        # create test functions
        nodes, self.fem_base = pi.cure_interval(pi.LagrangeFirstOrder,
                                                self.dz.bounds,
                                                node_count=self.n)

        # register eigenfunctions
        pi.register_base("eig_base", eig_base)
        pi.register_base("eig_base_i", eig_base_i)
        pi.register_base("eig_base_ti", eig_base_ti)
        pi.register_base("fem_base", self.fem_base)

        # original () and target (_t) field variable
        fem_field_variable = pi.FieldVariable("fem_base", location=self.l)

        field_variable_i = pi.FieldVariable("eig_base_i",
                                            weight_label="eig_base",
                                            location=self.l)
        d_field_variable_i = field_variable_i.derive(spat_order=1)

        field_variable_ti = pi.FieldVariable("eig_base_ti",
                                             weight_label="eig_base",
                                             location=self.l)
        d_field_variable_ti = field_variable_ti.derive(spat_order=1)

        # intermediate (_i) and target intermediate (_ti) field variable
        # (list of scalar terms = sum of scalar terms)
        self.x_fem_i_at_l = [pi.ScalarTerm(fem_field_variable)]
        self.x_i_at_l = [pi.ScalarTerm(field_variable_i)]
        self.xd_i_at_l = [pi.ScalarTerm(d_field_variable_i)]
        self.x_ti_at_l = [pi.ScalarTerm(field_variable_ti)]
        self.xd_ti_at_l = [pi.ScalarTerm(d_field_variable_ti)]

        # shift transformation

        def scale_func(z):
            return np.exp(a1 / 2 / a2 * z)

        shifted_fem_base_i = pi.Base(
            [pi.FiniteTransformFunction(func,
                                        m_mat,
                                        self.l,
                                        scale_func=scale_func)
             for func in self.fem_base.fractions])

        shifted_eig_base_id = pi.Base([pi.FiniteTransformFunction(frac,
                                                                  m_mat,
                                                                  self.l)
                                       for frac in eig_base_id.fractions])

        pi.register_base("sh_fem_base_i", shifted_fem_base_i)
        pi.register_base("sh_eig_base_id", shifted_eig_base_id)

        sh_fem_field_variable_i = pi.FieldVariable("sh_fem_base_i",
                                                   weight_label="fem_base",
                                                   location=self.l)
        sh_field_variable_id = pi.FieldVariable("sh_eig_base_id",
                                                weight_label="eig_base",
                                                location=self.l)
        self.sh_x_fem_i_at_l = [pi.ScalarTerm(sh_fem_field_variable_i),
                                pi.ScalarTerm(field_variable_i),
                                pi.ScalarTerm(sh_field_variable_id, -1)]

        # discontinuous operator (Kx)(t) = int_kernel_zz(l)*x(l,t)
        self.int_kernel_zz = lambda z: (self.alpha_ti - self.alpha_i
                                        + (a0_i - a0_ti) / 2 / a2 * z)

        a2, a1, _, _, _ = self.param
        controller = parabolic.get_parabolic_robin_backstepping_controller(
            state=self.sh_x_fem_i_at_l,
            approx_state=self.x_i_at_l,
            d_approx_state=self.xd_i_at_l,
            approx_target_state=self.x_ti_at_l,
            d_approx_target_state=self.xd_ti_at_l,
            integral_kernel_zz=self.int_kernel_zz(self.l),
            original_beta=self.beta_i,
            target_beta=self.beta_ti,
            scale=np.exp(-a1 / 2 / a2 * self.b))

        # init trajectory
        self.traj = pi.parabolic.RadTrajectory(self.l,
                                               self.T,
                                               self.param_ti,
                                               bound_cond_type,
                                               actuation_type,
                                               scale=np.exp(-a1 / 2 / a2 * self.b))

        system_input = pi.SimulationInputSum([self.traj, controller])

        # determine (A,B) with modal transformation
        rad_pde, lbls = parabolic.get_parabolic_robin_weak_form("fem_base",
                                                                "fem_base",
                                                                system_input,
                                                                self.param,
                                                                self.dz.bounds,
                                                                self.b)
        cf = pi.parse_weak_formulation(rad_pde)
        ss_weak = pi.create_state_space(cf)

        # simulate
        t, q = pi.simulate_state_space(ss_weak,
                                       np.zeros((self.n)),
                                       self.dt)

        # weights of the intermediate system
        mat = pi.calculate_base_transformation_matrix(self.fem_base, eig_base)
        q_i = np.zeros((q.shape[0], self.n))
        for i in range(q.shape[0]):
            q_i[i, :] = np.dot(q[i, :], np.transpose(mat))

        eval_i = pi.evaluate_approximation("eig_base_i", q_i, t, self.dz)
        x_0t = eval_i.output_data[:, 0]
        yc, tc = pi.gevrey_tanh(self.T, 1)
        x_0t_desired = np.interp(t, tc, yc[0, :])
        self.assertLess(np.average((x_0t - x_0t_desired) ** 2), 1e-2)

        # display results
        if show_plots:
            eval_d = pi.evaluate_approximation("fem_base", q, t, self.dz)
            win1 = pi.PgSurfacePlot(eval_i)
            win2 = pi.PgSurfacePlot(eval_d)
            app.exec_()

        pi.deregister_base("eig_base")
        pi.deregister_base("eig_base_i")
        pi.deregister_base("eig_base_ti")
        pi.deregister_base("fem_base")
        pi.deregister_base("sh_fem_base_i")
        pi.deregister_base("sh_eig_base_id")

        for lbl in lbls:
            pi.deregister_base(lbl)
