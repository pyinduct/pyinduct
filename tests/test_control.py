import sys
import numpy as np
from scipy import integrate
import unittest

from pyinduct import register_base
from pyinduct import core as cr
from pyinduct import control as ct
from pyinduct import placeholder as ph
from pyinduct import utils as ut
from pyinduct import trajectory as tr
from pyinduct import eigenfunctions as ef
from pyinduct import shapefunctions as sf
from pyinduct import simulation as sim
from pyinduct import visualization as vis

if any([arg in {'discover', 'setup.py', 'test'} for arg in sys.argv]):
    show_plots = False
else:
    # show_plots = True
    show_plots = False

if show_plots:
    import pyqtgraph as pg
    app = pg.QtGui.QApplication([])

# TODO Test for ControlLaw and LawEvaluator


class CollocatedTestCase(unittest.TestCase):
    def setUp(self):
        interval = (0, 1)
        nodes, funcs = sf.cure_interval(sf.LagrangeFirstOrder, interval, 3)
        register_base("funcs", funcs, overwrite=True)
        x_at1 = ph.FieldVariable("funcs", location=1)
        x_dt_at1 = ph.TemporalDerivedFieldVariable("funcs", 1, location=1)
        x_dz_at0 = ph.SpatialDerivedFieldVariable("funcs", 1, location=0)

        exp_func = cr.Function(np.exp)
        register_base("exp_func", exp_func, overwrite=True)
        exp_at1 = ph.ScalarFunction("exp_func", location=1)

        alpha = 2
        self.term1 = ph.ScalarTerm(x_dt_at1, 1 + alpha)
        self.term2 = ph.ScalarTerm(x_dz_at0, 2)
        self.term3 = ph.ScalarTerm(ph.Product(x_at1, exp_at1))

        self.weight_label = "funcs"
        self.weights = np.array([1, 1, 1, 2, 2, 2])

    def test_temp_term(self):
        law = ct.LawEvaluator(ct.approximate_control_law(ct.ControlLaw([self.term1])))
        res = law(self.weights, self.weight_label)["output"]
        self.assertAlmostEqual(res, 6)

    def test_spat_term(self):
        law = ct.LawEvaluator(ct.approximate_control_law(ct.ControlLaw([self.term2])))
        res = law(self.weights, self.weight_label)["output"]
        self.assertAlmostEqual(res, 0)

    def test_product_term(self):
        law = ct.LawEvaluator(ct.approximate_control_law(ct.ControlLaw([self.term3])))
        res = law(self.weights, self.weight_label)["output"]
        self.assertAlmostEqual(res, 1 * np.exp(1))


class ContinuousTestCase(unittest.TestCase):

    def setUp(self):
        interval = (0, 1)
        nodes, funcs = sf.cure_interval(sf.LagrangeFirstOrder, interval, 3)
        register_base("funcs", funcs, overwrite=True)
        x = ph.FieldVariable("funcs")
        x_dt = ph.TemporalDerivedFieldVariable("funcs", 1)
        x_dz = ph.SpatialDerivedFieldVariable("funcs", 1)
        register_base("scal_func", cr.Function(np.exp), overwrite=True)
        exp = ph.ScalarFunction("scal_func")

        alpha = 2
        self.term1 = ph.IntegralTerm(x_dt, interval, 1 + alpha)
        self.term2 = ph.IntegralTerm(x_dz, interval, 2)
        self.term3 = ph.IntegralTerm(ph.Product(x, exp), interval)

        self.weight_label = "funcs"
        self.weights = np.hstack([1, 1, 1, 2, 2, 2])

    def test_temp_term(self):
        law = ct.LawEvaluator(ct.approximate_control_law(ct.ControlLaw([self.term1])))
        res = law(self.weights, self.weight_label)["output"]
        self.assertTrue(np.equal(res, 6))

    def test_spat_term(self):
        law = ct.LawEvaluator(ct.approximate_control_law(ct.ControlLaw([self.term2])))
        res = law(self.weights, self.weight_label)["output"]
        self.assertAlmostEqual(res, 0)

    def test_product_term(self):
        law = ct.LawEvaluator(ct.approximate_control_law(ct.ControlLaw([self.term3])))
        res = law(self.weights, self.weight_label)["output"]
        # TODO calculate expected result
        # self.assertAlmostEqual(res, 1*np.exp(1))


class RadDirichletControlApproxTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_it(self):
        # original system parameters
        a2 = 1
        a1 = 0  # attention: only a2 = 1., a1 =0 supported in this test case
        a0 = 0
        param = [a2, a1, a0, None, None]

        # target system parameters (controller parameters)
        a1_t = 0
        a0_t = 0  # attention: only a2 = 1., a1 =0 and a0 =0 supported in this test case
        param_t = [a2, a1_t, a0_t, None, None]

        # system/simulation parameters
        actuation_type = 'dirichlet'
        bound_cond_type = 'dirichlet'

        l = 1.
        spatial_disc = 10
        dz = sim.Domain(bounds=(0, l), num=spatial_disc)

        T = 1.
        temporal_disc = 1e2
        dt = sim.Domain(bounds=(0, T), num=temporal_disc)

        n = 10

        # eigenvalues /-functions original system
        eig_freq = np.array([(i + 1) * np.pi / l for i in range(n)])
        eig_values = a0 - a2 * eig_freq ** 2 - a1 ** 2 / 4. / a2
        norm_fac = np.ones(eig_freq.shape) * np.sqrt(2)
        eig_funcs = np.asarray([ef.SecondOrderDirichletEigenfunction(eig_freq[i], param, dz.bounds, norm_fac[i])
                                for i in range(n)])
        register_base("eig_funcs", eig_funcs, overwrite=True)

        # eigenfunctions target system
        eig_freq_t = np.sqrt(-eig_values.astype(complex))
        norm_fac_t = norm_fac * eig_freq / eig_freq_t
        eig_funcs_t = np.asarray([ef.SecondOrderDirichletEigenfunction(eig_freq_t[i], param_t, dz.bounds, norm_fac_t[i])
                                  for i in range(n)])
        register_base("eig_funcs_t", eig_funcs_t, overwrite=True)

        # derive initial field variable x(z,0) and weights
        start_state = cr.Function(lambda z: 0., domain=(0, l))
        initial_weights = cr.project_on_base(start_state, eig_funcs)

        # init trajectory / input of target system
        traj = tr.RadTrajectory(l, T, param_t, bound_cond_type, actuation_type)

        # init controller
        x_at_1 = ph.FieldVariable("eig_funcs", location=1)
        xt_at_1 = ph.FieldVariable("eig_funcs_t", weight_label="eig_funcs", location=1)
        controller = ct.Controller(ct.ControlLaw([ph.ScalarTerm(x_at_1, 1), ph.ScalarTerm(xt_at_1, -1)]))

        # input with feedback
        control_law = sim.SimulationInputSum([traj, controller])

        # determine (A,B) with modal-transfomation
        A = np.diag(eig_values)
        B = -a2 * np.array([eig_funcs[i].derive()(l) for i in range(n)])
        ss = sim.StateSpace("eig_funcs", A, B, input_handle=control_law)

        # simulate
        t, q = sim.simulate_state_space(ss, initial_weights, dt)

        eval_d = sim.evaluate_approximation("eig_funcs", q, t, dz)
        x_0t = eval_d.output_data[:, 0]
        yc, tc = tr.gevrey_tanh(T, 1)
        x_0t_desired = np.interp(t, tc, yc[0, :])
        self.assertLess(np.average((x_0t - x_0t_desired) ** 2), 0.5)

        # display results
        if show_plots:
            eval_d = sim.evaluate_approximation("eig_funcs", q, t, dz)
            win2 = vis.PgSurfacePlot(eval_d)
            app.exec_()


class RadRobinControlApproxTest(unittest.TestCase):
    """
    """

    def setUp(self):
        pass

    def test_it(self):
        # original system parameters
        a2 = 1.5;
        a1 = 2.5;
        a0 = 28;
        alpha = -2;
        beta = -3
        param = [a2, a1, a0, alpha, beta]
        adjoint_param = ef.get_adjoint_rad_evp_param(param)

        # target system parameters (controller parameters)
        a1_t = -5;
        a0_t = -25;
        alpha_t = 3;
        beta_t = 2
        # a1_t = a1; a0_t = a0; alpha_t = alpha; beta_t = beta
        param_t = [a2, a1_t, a0_t, alpha_t, beta_t]

        # original intermediate ("_i") and traget intermediate ("_ti") system parameters
        _, _, a0_i, alpha_i, beta_i = ef.transform2intermediate(param)
        _, _, a0_ti, alpha_ti, beta_ti = ef.transform2intermediate(param_t)

        # system/simulation parameters
        actuation_type = 'robin'
        bound_cond_type = 'robin'
        self.l = 1.
        spatial_disc = 10
        dz = sim.Domain(bounds=(0, self.l), num=spatial_disc)

        T = 1.
        temporal_disc = 1e2
        dt = sim.Domain(bounds=(0, T), num=temporal_disc)
        n = 10

        # create (not normalized) eigenfunctions
        eig_freq, eig_val = ef.compute_rad_robin_eigenfrequencies(param, self.l, n)
        init_eig_funcs = np.array([ef.SecondOrderRobinEigenfunction(om, param, dz.bounds) for om in eig_freq])
        init_adjoint_eig_funcs = np.array([ef.SecondOrderRobinEigenfunction(om, adjoint_param, dz.bounds)
                                           for om in eig_freq])

        # normalize eigenfunctions and adjoint eigenfunctions
        adjoint_and_eig_funcs = [cr.normalize_function(init_eig_funcs[i], init_adjoint_eig_funcs[i]) for i in range(n)]
        eig_funcs = np.array([f_tuple[0] for f_tuple in adjoint_and_eig_funcs])
        adjoint_eig_funcs = np.array([f_tuple[1] for f_tuple in adjoint_and_eig_funcs])

        # eigenfunctions from target system ("_t")
        eig_freq_t = np.sqrt(-a1_t ** 2 / 4 / a2 ** 2 + (a0_t - eig_val) / a2)
        eig_funcs_t = np.array([ef.SecondOrderRobinEigenfunction(eig_freq_t[i],
                                                                 param_t, dz.bounds).scale(eig_funcs[i](0))
                                for i in range(n)])

        # register eigenfunctions
        register_base("eig_funcs", eig_funcs, overwrite=True)
        register_base("adjoint_eig_funcs", adjoint_eig_funcs, overwrite=True)
        register_base("eig_funcs_t", eig_funcs_t, overwrite=True)

        # derive initial field variable x(z,0) and weights
        start_state = cr.Function(lambda z: 0., domain=(0, self.l))
        initial_weights = cr.project_on_base(start_state, adjoint_eig_funcs)

        # controller initialization
        x_at_l = ph.FieldVariable("eig_funcs", location=self.l)
        xd_at_l = ph.SpatialDerivedFieldVariable("eig_funcs", 1, location=self.l)
        x_t_at_l = ph.FieldVariable("eig_funcs_t", weight_label="eig_funcs", location=self.l)
        xd_t_at_l = ph.SpatialDerivedFieldVariable("eig_funcs_t", 1, weight_label="eig_funcs", location=self.l)
        combined_transform = lambda z: np.exp((a1_t - a1) / 2 / a2 * z)
        int_kernel_zz = lambda z: alpha_ti - alpha_i + (a0_i - a0_ti) / 2 / a2 * z
        controller = ct.Controller(
            ct.ControlLaw([ph.ScalarTerm(x_at_l, (beta_i - beta_ti - int_kernel_zz(self.l))),
                           ph.ScalarTerm(x_t_at_l, -beta_ti * combined_transform(self.l)),
                           ph.ScalarTerm(x_at_l, beta_ti),
                           ph.ScalarTerm(xd_t_at_l, -combined_transform(self.l)),
                           ph.ScalarTerm(x_t_at_l, -a1_t / 2 / a2 * combined_transform(self.l)),
                           ph.ScalarTerm(xd_at_l, 1),
                           ph.ScalarTerm(x_at_l, a1 / 2 / a2 + int_kernel_zz(self.l))
                           ]))

        # init trajectory
        traj = tr.RadTrajectory(self.l, T, param_t, bound_cond_type, actuation_type)
        traj.scale = combined_transform(self.l)

        # input with feedback
        control_law = sim.SimulationInputSum([traj, controller])
        # control_law = sim.simInputSum([traj])

        # determine (A,B) with modal-transformation
        A = np.diag(np.real(eig_val))
        B = a2 * np.array([adjoint_eig_funcs[i](self.l) for i in range(len(eig_freq))])
        ss_modal = sim.StateSpace("eig_funcs", A, B, input_handle=control_law)

        # simulate
        t, q = sim.simulate_state_space(ss_modal, initial_weights, dt)

        eval_d = sim.evaluate_approximation("eig_funcs", q, t, dz)
        x_0t = eval_d.output_data[:, 0]
        yc, tc = tr.gevrey_tanh(T, 1)
        x_0t_desired = np.interp(t, tc, yc[0, :])
        self.assertLess(np.average((x_0t - x_0t_desired) ** 2), 1e-4)

        # display results
        if show_plots:
            win1 = vis.PgAnimatedPlot([eval_d], title="Test")
            win2 = vis.PgSurfacePlot(eval_d)
            app.exec_()


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
        adjoint_param = ef.get_adjoint_rad_evp_param(self.param)

        # target system parameters (controller parameters)
        a1_t = -5
        a0_t = -25
        alpha_t = 3
        beta_t = 2
        # a1_t = a1; a0_t = a0; alpha_t = alpha; beta_t = beta
        self.param_t = [a2, a1_t, a0_t, alpha_t, beta_t]

        # original intermediate ("_i") and target intermediate ("_ti") system parameters
        _, _, a0_i, self.alpha_i, self.beta_i = ef.transform2intermediate(self.param)
        self.param_i = a2, 0, a0_i, self.alpha_i, self.beta_i
        _, _, a0_ti, self.alpha_ti, self.beta_ti = ef.transform2intermediate(self.param_t)
        self.param_ti = a2, 0, a0_ti, self.alpha_ti, self.beta_ti

        # system/simulation parameters
        actuation_type = 'robin'
        bound_cond_type = 'robin'
        self.l = 1.
        spatial_disc = 10
        self.dz = sim.Domain(bounds=(0, self.l), num=spatial_disc)

        self.T = 1.
        temporal_disc = 1e2
        self.dt = sim.Domain(bounds=(0, self.T), num=temporal_disc)
        self.n = 10

        # create (not normalized) eigenfunctions
        eig_freq, self.eig_val = ef.compute_rad_robin_eigenfrequencies(self.param, self.l, self.n)
        init_eig_funcs = np.array([ef.SecondOrderRobinEigenfunction(om, self.param, self.dz.bounds) for om in eig_freq])
        init_adjoint_eig_funcs = np.array([ef.SecondOrderRobinEigenfunction(om, adjoint_param, self.dz.bounds)
                                           for om in eig_freq])

        # normalize eigenfunctions and adjoint eigenfunctions
        adjoint_and_eig_funcs = [cr.normalize_function(init_eig_funcs[i], init_adjoint_eig_funcs[i])
                                 for i in range(self.n)]
        eig_funcs = np.array([f_tuple[0] for f_tuple in adjoint_and_eig_funcs])
        self.adjoint_eig_funcs = np.array([f_tuple[1] for f_tuple in adjoint_and_eig_funcs])

        # eigenfunctions from target system ("_t")
        eig_freq_t = np.sqrt(-a1_t ** 2 / 4 / a2 ** 2 + (a0_t - self.eig_val) / a2)
        eig_funcs_t = np.array([ef.SecondOrderRobinEigenfunction(eig_freq_t[i],
                                                                 self.param_t, self.dz.bounds).scale(eig_funcs[i](0))
                                for i in range(self.n)])

        # create testfunctions
        nodes, self.fem_funcs = sf.cure_interval(sf.LagrangeFirstOrder,
                                                                      self.dz.bounds,
                                                                      node_count=self.n)

        # register eigenfunctions
        register_base("eig_funcs", eig_funcs, overwrite=True)
        register_base("adjoint_eig_funcs", self.adjoint_eig_funcs, overwrite=True)
        register_base("eig_funcs_t", eig_funcs_t, overwrite=True)
        register_base("fem_funcs", self.fem_funcs, overwrite=True)

        # init trajectory
        self.traj = tr.RadTrajectory(self.l, self.T, self.param_ti, bound_cond_type, actuation_type)

        # original () and target (_t) field variable
        fem_field_variable = ph.FieldVariable("fem_funcs", location=self.l)
        field_variable = ph.FieldVariable("eig_funcs", location=self.l)
        d_field_variable = ph.SpatialDerivedFieldVariable("eig_funcs", 1, location=self.l)
        field_variable_t = ph.FieldVariable("eig_funcs_t", weight_label="eig_funcs", location=self.l)
        d_field_variable_t = ph.SpatialDerivedFieldVariable("eig_funcs_t", 1, weight_label="eig_funcs", location=self.l)

        # intermediate (_i) and target intermediate (_ti) transformations by z=l
        self.transform_i = lambda z: np.exp(a1 / 2 / a2 * z)  # x_i  = x   * transform_i
        self.transform_ti = lambda z: np.exp(a1_t / 2 / a2 * z)  # x_ti = x_t * transform_ti

        # intermediate (_i) and target intermediate (_ti) field variable (list of scalar terms = sum of scalar terms)
        self.x_fem_i_at_l = [ph.ScalarTerm(fem_field_variable, self.transform_i(self.l))]
        self.x_i_at_l = [ph.ScalarTerm(field_variable, self.transform_i(self.l))]
        self.xd_i_at_l = [ph.ScalarTerm(d_field_variable, self.transform_i(self.l)),
                          ph.ScalarTerm(field_variable, self.transform_i(self.l) * a1 / 2 / a2)]
        self.x_ti_at_l = [ph.ScalarTerm(field_variable_t, self.transform_ti(self.l))]
        self.xd_ti_at_l = [ph.ScalarTerm(d_field_variable_t, self.transform_ti(self.l)),
                           ph.ScalarTerm(field_variable_t, self.transform_ti(self.l) * a1_t / 2 / a2)]

        # discontinuous operator (Kx)(t) = int_kernel_zz(l)*x(l,t)
        self.int_kernel_zz = lambda z: self.alpha_ti - self.alpha_i + (a0_i - a0_ti) / 2 / a2 * z

    def test_fem(self):
        self.act_funcs = "fem_funcs"
        controller = ut.get_parabolic_robin_backstepping_controller(state=self.x_fem_i_at_l,
                                                                    approx_state=self.x_i_at_l,
                                                                    d_approx_state=self.xd_i_at_l,
                                                                    approx_target_state=self.x_ti_at_l,
                                                                    d_approx_target_state=self.xd_ti_at_l,
                                                                    integral_kernel_zz=self.int_kernel_zz(self.l),
                                                                    original_beta=self.beta_i,
                                                                    target_beta=self.beta_ti,
                                                                    trajectory=self.traj,
                                                                    scale=self.transform_i(-self.l))

        # determine (A,B) with modal-transfomation
        rad_pde = ut.get_parabolic_robin_weak_form(self.act_funcs, self.act_funcs, controller, self.param,
                                                   self.dz.bounds)
        cf = sim.parse_weak_formulation(rad_pde)
        ss_weak = cf.convert_to_state_space()

        # simulate
        self.t, self.q = sim.simulate_state_space(ss_weak, np.zeros((len(self.fem_funcs))),
                                                  self.dt)

        eval_d = sim.evaluate_approximation(self.act_funcs, self.q, self.t, self.dz)
        x_0t = eval_d.output_data[:, 0]
        yc, tc = tr.gevrey_tanh(self.T, 1)
        x_0t_desired = np.interp(self.t, tc, yc[0, :])
        self.assertLess(np.average((x_0t - x_0t_desired) ** 2), 1e-3)

        # display results
        if show_plots:
            win1 = vis.PgAnimatedPlot([eval_d], title="Test")
            win2 = vis.PgSurfacePlot(eval_d)
            app.exec_()

    def test_modal(self):
        self.act_funcs = "eig_funcs"
        a2, a1, a0, alpha, beta = self.param
        controller = ut.get_parabolic_robin_backstepping_controller(state=self.x_i_at_l,
                                                                    approx_state=self.x_i_at_l,
                                                                    d_approx_state=self.xd_i_at_l,
                                                                    approx_target_state=self.x_ti_at_l,
                                                                    d_approx_target_state=self.xd_ti_at_l,
                                                                    integral_kernel_zz=self.int_kernel_zz(self.l),
                                                                    original_beta=self.beta_i,
                                                                    target_beta=self.beta_ti,
                                                                    trajectory=self.traj,
                                                                    scale=self.transform_i(-self.l))

        # determine (A,B) with modal-transfomation
        A = np.diag(np.real(self.eig_val))
        B = a2 * np.array([self.adjoint_eig_funcs[i](self.l) for i in range(self.n)])
        ss_modal = sim.StateSpace(self.act_funcs, A, B, input_handle=controller)

        # simulate
        self.t, self.q = sim.simulate_state_space(ss_modal, np.zeros((len(self.adjoint_eig_funcs))),
                                                  self.dt)

        eval_d = sim.evaluate_approximation(self.act_funcs, self.q, self.t, self.dz)
        x_0t = eval_d.output_data[:, 0]
        yc, tc = tr.gevrey_tanh(self.T, 1)
        x_0t_desired = np.interp(self.t, tc, yc[0, :])
        self.assertLess(np.average((x_0t - x_0t_desired) ** 2), 1e-2)

        # display results
        if show_plots:
            win1 = vis.PgAnimatedPlot([eval_d], title="Test")
            win2 = vis.PgSurfacePlot(eval_d)
            app.exec_()


class RadRobinSpatiallyVaryingCoefficientControllerTest(unittest.TestCase):
    """
    """

    def test_it(self):
        # system/simulation parameters
        actuation_type = 'robin'
        bound_cond_type = 'robin'

        self.l = 1.
        spatial_disc = 10
        self.dz = sim.Domain(bounds=(0, self.l), num=spatial_disc)

        self.T = 1.
        temporal_disc = 1e2
        self.dt = sim.Domain(bounds=(0, self.T), num=temporal_disc)

        self.n = 10

        # original system parameters
        a2 = 1.5
        a1_z = cr.Function(lambda z: 1, derivative_handles=[lambda z: 0])
        a0_z = lambda z: 3
        alpha = -2
        beta = -3
        self.param = [a2, a1_z, a0_z, alpha, beta]

        # target system parameters (controller parameters)
        a1_t = -5
        a0_t = -25
        alpha_t = 3
        beta_t = 2
        self.param_t = [a2, a1_t, a0_t, alpha_t, beta_t]
        adjoint_param_t = ef.get_adjoint_rad_evp_param(self.param_t)

        # original intermediate ("_i") and traget intermediate ("_ti") system parameters
        _, _, a0_i, alpha_i, beta_i = ef.transform2intermediate(self.param, d_end=self.l)
        self.param_i = a2, 0, a0_i, alpha_i, beta_i
        _, _, a0_ti, alpha_ti, beta_ti = ef.transform2intermediate(self.param_t)
        self.param_ti = a2, 0, a0_ti, alpha_ti, beta_ti

        # create (not normalized) target (_t) eigenfunctions
        eig_freq_t, self.eig_val_t = ef.compute_rad_robin_eigenfrequencies(self.param_t, self.l, self.n)
        init_eig_funcs_t = np.array([ef.SecondOrderRobinEigenfunction(om, self.param_t, self.dz.bounds)
                                     for om in eig_freq_t])
        init_adjoint_eig_funcs_t = np.array([ef.SecondOrderRobinEigenfunction(om, adjoint_param_t, self.dz.bounds)
                                             for om in eig_freq_t])

        # normalize eigenfunctions and adjoint eigenfunctions
        adjoint_and_eig_funcs_t = [cr.normalize_function(init_eig_funcs_t[i], init_adjoint_eig_funcs_t[i])
                                   for i in range(self.n)]
        eig_funcs_t = np.array([f_tuple[0] for f_tuple in adjoint_and_eig_funcs_t])
        self.adjoint_eig_funcs_t = np.array([f_tuple[1] for f_tuple in adjoint_and_eig_funcs_t])

        # # transformed original eigenfunctions
        self.eig_funcs = np.array([ef.TransformedSecondOrderEigenfunction(self.eig_val_t[i],
                                                                          [eig_funcs_t[i](0), alpha * eig_funcs_t[i](0),
                                                                           0, 0],
                                                                          [a2, a1_z, a0_z],
                                                                          np.linspace(0, self.l, 1e4))
                                   for i in range(self.n)])

        # create testfunctions
        nodes, self.fem_funcs = sf.cure_interval(sf.LagrangeFirstOrder,
                                                                      self.dz.bounds,
                                                                      node_count=self.n)

        # register functions
        register_base("eig_funcs_t", eig_funcs_t, overwrite=True)
        register_base("adjoint_eig_funcs_t", self.adjoint_eig_funcs_t, overwrite=True)
        register_base("eig_funcs", self.eig_funcs, overwrite=True)
        register_base("fem_funcs", self.fem_funcs, overwrite=True)

        # init trajectory
        self.traj = tr.RadTrajectory(self.l, self.T, self.param_ti, bound_cond_type, actuation_type)

        # original () and target (_t) field variable
        fem_field_variable = ph.FieldVariable("fem_funcs", location=self.l)
        field_variable_t = ph.FieldVariable("eig_funcs_t", weight_label="eig_funcs", location=self.l)
        d_field_variable_t = ph.SpatialDerivedFieldVariable("eig_funcs_t", 1, weight_label="eig_funcs", location=self.l)
        field_variable = ph.FieldVariable("eig_funcs", location=self.l)
        d_field_variable = ph.SpatialDerivedFieldVariable("eig_funcs", 1, location=self.l)
        # intermediate (_i) and target intermediate (_ti) transformations by z=l

        #  x_i  = x   * transform_i_at_l
        self.transform_i_at_l = np.exp(integrate.quad(lambda z: a1_z(z) / 2 / a2, 0, self.l)[0])

        # x  = x_i   * inv_transform_i_at_l
        self.inv_transform_i_at_l = np.exp(-integrate.quad(lambda z: a1_z(z) / 2 / a2, 0, self.l)[0])

        # x_ti = x_t * transform_ti_at_l
        self.transform_ti_at_l = np.exp(a1_t / 2 / a2 * self.l)

        # intermediate (_i) and target intermediate (_ti) field variable (list of scalar terms = sum of scalar terms)
        self.x_fem_i_at_l = [ph.ScalarTerm(fem_field_variable, self.transform_i_at_l)]
        self.x_i_at_l = [ph.ScalarTerm(field_variable, self.transform_i_at_l)]
        self.xd_i_at_l = [ph.ScalarTerm(d_field_variable, self.transform_i_at_l),
                          ph.ScalarTerm(field_variable, self.transform_i_at_l * a1_z(self.l) / 2 / a2)]
        self.x_ti_at_l = [ph.ScalarTerm(field_variable_t, self.transform_ti_at_l)]
        self.xd_ti_at_l = [ph.ScalarTerm(d_field_variable_t, self.transform_ti_at_l),
                           ph.ScalarTerm(field_variable_t, self.transform_ti_at_l * a1_t / 2 / a2)]

        # discontinuous operator (Kx)(t) = int_kernel_zz(l)*x(l,t)
        self.int_kernel_zz = alpha_ti - alpha_i + integrate.quad(lambda z: (a0_i(z) - a0_ti) / 2 / a2, 0, self.l)[0]

        controller = ut.get_parabolic_robin_backstepping_controller(state=self.x_fem_i_at_l,
                                                                    approx_state=self.x_i_at_l,
                                                                    d_approx_state=self.xd_i_at_l,
                                                                    approx_target_state=self.x_ti_at_l,
                                                                    d_approx_target_state=self.xd_ti_at_l,
                                                                    integral_kernel_zz=self.int_kernel_zz,
                                                                    original_beta=beta_i,
                                                                    target_beta=beta_ti,
                                                                    trajectory=self.traj,
                                                                    scale=self.inv_transform_i_at_l)

        rad_pde = ut.get_parabolic_robin_weak_form("fem_funcs", "fem_funcs", controller, self.param, self.dz.bounds)
        cf = sim.parse_weak_formulation(rad_pde)
        ss_weak = cf.convert_to_state_space()

        # simulate
        t, q = sim.simulate_state_space(ss_weak, np.zeros((len(self.fem_funcs))), self.dt)
        eval_d = sim.evaluate_approximation("fem_funcs", q, t, self.dz)
        x_0t = eval_d.output_data[:, 0]
        yc, tc = tr.gevrey_tanh(self.T, 1)
        x_0t_desired = np.interp(t, tc, yc[0, :])
        self.assertLess(np.average((x_0t - x_0t_desired) ** 2), 1e-4)

        # display results
        if show_plots:
            win1 = vis.PgAnimatedPlot([eval_d], title="Test")
            win2 = vis.PgSurfacePlot(eval_d)
            app.exec_()


class RadRobinInDomainBacksteppingControllerTest(unittest.TestCase):
    """
    """

    def test_fem(self):

        # system/simulation parameters
        actuation_type = 'robin'
        bound_cond_type = 'robin'

        self.l = 1.
        spatial_disc = 30
        self.dz = sim.Domain(bounds=(0, self.l), num=spatial_disc)

        self.T = 1.
        temporal_disc = 1e2
        self.dt = sim.Domain(bounds=(0, self.T), num=temporal_disc)
        self.n = 12

        # original system parameters
        a2 = 1.5
        a1 = 2.5
        a0 = 28
        alpha = -2
        beta = -3
        self.param = [a2, a1, a0, alpha, beta]
        adjoint_param = ef.get_adjoint_rad_evp_param(self.param)

        # target system parameters (controller parameters)
        a1_t = -5
        a0_t = -25
        alpha_t = 3
        beta_t = 2
        self.param_t = [a2, a1_t, a0_t, alpha_t, beta_t]

        # actuation_type by b which is close to b_desired on a k times subdivided spatial domain
        b_desired = self.l / 2
        k = 51  # = k1 + k2
        k1, k2, self.b = ut.split_domain(k, b_desired, self.l, mode='coprime')[0:3]
        M = np.linalg.inv(ut.get_inn_domain_transformation_matrix(k1, k2, mode="2n"))

        # original intermediate ("_i") and traget intermediate ("_ti") system parameters
        _, _, a0_i, self.alpha_i, self.beta_i = ef.transform2intermediate(self.param)
        self.param_i = a2, 0, a0_i, self.alpha_i, self.beta_i
        _, _, a0_ti, self.alpha_ti, self.beta_ti = ef.transform2intermediate(self.param_t)
        self.param_ti = a2, 0, a0_ti, self.alpha_ti, self.beta_ti

        # create (not normalized) eigenfunctions
        eig_freq, self.eig_val = ef.compute_rad_robin_eigenfrequencies(self.param, self.l, self.n)
        init_eig_funcs = np.array([ef.SecondOrderRobinEigenfunction(om, self.param, self.dz.bounds) for om in eig_freq])
        init_adjoint_eig_funcs = np.array(
            [ef.SecondOrderRobinEigenfunction(om, adjoint_param, self.dz.bounds) for om in eig_freq])

        # normalize eigenfunctions and adjoint eigenfunctions
        adjoint_and_eig_funcs = [cr.normalize_function(init_eig_funcs[i], init_adjoint_eig_funcs[i]) for i in
                                 range(self.n)]
        eig_funcs = np.array([f_tuple[0] for f_tuple in adjoint_and_eig_funcs])
        self.adjoint_eig_funcs = np.array([f_tuple[1] for f_tuple in adjoint_and_eig_funcs])

        # eigenfunctions of the in-domain intermediate (_id) and the intermediate (_i) system
        eig_freq_i, eig_val_i = ef.compute_rad_robin_eigenfrequencies(self.param_i, self.l, self.n)
        self.assertTrue(all(np.isclose(eig_val_i, self.eig_val)))
        eig_funcs_id = np.array([ef.SecondOrderRobinEigenfunction(eig_freq_i[i],
                                                                  self.param_i,
                                                                  self.dz.bounds,
                                                                  eig_funcs[i](0))
                                 for i in range(self.n)])
        eig_funcs_i = np.array([ef.SecondOrderRobinEigenfunction(eig_freq_i[i],
                                                                 self.param_i,
                                                                 self.dz.bounds,
                                                                 eig_funcs[i](0) * eig_funcs_id[i](self.l) /
                                                                 eig_funcs_id[i](self.b))
                                for i in range(self.n)])

        # eigenfunctions from target system ("_ti")
        eig_freq_ti = np.sqrt((a0_ti - self.eig_val) / a2)
        eig_funcs_ti = np.array([ef.SecondOrderRobinEigenfunction(eig_freq_ti[i],
                                                                  self.param_ti,
                                                                  self.dz.bounds,
                                                                  eig_funcs_i[i](0))
                                 for i in range(self.n)])

        # create testfunctions
        nodes, self.fem_funcs = sf.cure_interval(sf.LagrangeFirstOrder,
                                                                      self.dz.bounds,
                                                                      node_count=self.n)

        # register eigenfunctions
        # register_functions("eig_funcs", eig_funcs, overwrite=True)
        register_base("adjoint_eig_funcs", self.adjoint_eig_funcs, overwrite=True)
        register_base("eig_funcs", eig_funcs, overwrite=True)
        register_base("eig_funcs_i", eig_funcs_i, overwrite=True)
        register_base("eig_funcs_ti", eig_funcs_ti, overwrite=True)
        register_base("fem_funcs", self.fem_funcs, overwrite=True)

        # init trajectory
        self.traj = tr.RadTrajectory(self.l, self.T, self.param_ti, bound_cond_type, actuation_type)

        # original () and target (_t) field variable
        fem_field_variable = ph.FieldVariable("fem_funcs", location=self.l)
        field_variable_i = ph.FieldVariable("eig_funcs_i", weight_label="eig_funcs", location=self.l)
        d_field_variable_i = ph.SpatialDerivedFieldVariable("eig_funcs_i", 1, weight_label="eig_funcs", location=self.l)
        field_variable_ti = ph.FieldVariable("eig_funcs_ti", weight_label="eig_funcs", location=self.l)
        d_field_variable_ti = ph.SpatialDerivedFieldVariable("eig_funcs_ti", 1, weight_label="eig_funcs",
                                                             location=self.l)

        # intermediate (_i) and target intermediate (_ti) field variable (list of scalar terms = sum of scalar terms)
        self.x_fem_i_at_l = [ph.ScalarTerm(fem_field_variable)]
        self.x_i_at_l = [ph.ScalarTerm(field_variable_i)]
        self.xd_i_at_l = [ph.ScalarTerm(d_field_variable_i)]
        self.x_ti_at_l = [ph.ScalarTerm(field_variable_ti)]
        self.xd_ti_at_l = [ph.ScalarTerm(d_field_variable_ti)]

        # shift transformation
        shifted_fem_funcs_i = np.array(
            [ef.FiniteTransformFunction(func, M, self.b, self.l, scale_func=lambda z: np.exp(a1 / 2 / a2 * z))
             for func in self.fem_funcs])
        shifted_eig_funcs_id = np.array([ef.FiniteTransformFunction(func, M, self.b, self.l) for func in eig_funcs_id])
        register_base("sh_fem_funcs_i", shifted_fem_funcs_i, overwrite=True)
        register_base("sh_eig_funcs_id", shifted_eig_funcs_id, overwrite=True)
        sh_fem_field_variable_i = ph.FieldVariable("sh_fem_funcs_i", weight_label="fem_funcs", location=self.l)
        sh_field_variable_id = ph.FieldVariable("sh_eig_funcs_id", weight_label="eig_funcs", location=self.l)
        self.sh_x_fem_i_at_l = [ph.ScalarTerm(sh_fem_field_variable_i),
                                ph.ScalarTerm(field_variable_i),
                                ph.ScalarTerm(sh_field_variable_id, -1)]

        # discontinuous operator (Kx)(t) = int_kernel_zz(l)*x(l,t)
        self.int_kernel_zz = lambda z: self.alpha_ti - self.alpha_i + (a0_i - a0_ti) / 2 / a2 * z

        a2, a1, _, _, _ = self.param
        controller = ut.get_parabolic_robin_backstepping_controller(state=self.sh_x_fem_i_at_l,
                                                                    approx_state=self.x_i_at_l,
                                                                    d_approx_state=self.xd_i_at_l,
                                                                    approx_target_state=self.x_ti_at_l,
                                                                    d_approx_target_state=self.xd_ti_at_l,
                                                                    integral_kernel_zz=self.int_kernel_zz(self.l),
                                                                    original_beta=self.beta_i,
                                                                    target_beta=self.beta_ti,
                                                                    trajectory=self.traj,
                                                                    scale=np.exp(-a1 / 2 / a2 * self.b))

        # determine (A,B) with modal-transfomation
        rad_pde = ut.get_parabolic_robin_weak_form("fem_funcs", "fem_funcs", controller, self.param, self.dz.bounds,
                                                   self.b)
        cf = sim.parse_weak_formulation(rad_pde)
        ss_weak = cf.convert_to_state_space()

        # simulate
        t, q = sim.simulate_state_space(ss_weak, np.zeros((len(self.fem_funcs))), self.dt)

        # weights of the intermediate system
        mat = cr.calculate_base_transformation_matrix(self.fem_funcs, eig_funcs)
        q_i = np.zeros((q.shape[0], len(eig_funcs_i)))
        for i in range(q.shape[0]):
            q_i[i, :] = np.dot(q[i, :], np.transpose(mat))

        eval_i = sim.evaluate_approximation("eig_funcs_i", q_i, t, self.dz)
        x_0t = eval_i.output_data[:, 0]
        yc, tc = tr.gevrey_tanh(self.T, 1)
        x_0t_desired = np.interp(t, tc, yc[0, :])
        self.assertLess(np.average((x_0t - x_0t_desired) ** 2), 1e-2)

        # display results
        if show_plots:
            eval_d = sim.evaluate_approximation("fem_funcs", q, t, self.dz)
            win1 = vis.PgSurfacePlot(eval_i)
            win2 = vis.PgSurfacePlot(eval_d)
            app.exec_()
