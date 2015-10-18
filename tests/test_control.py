from __future__ import division
import unittest
import numpy as np
from pyinduct import register_functions
from pyinduct import core as cr
from pyinduct import control as ct
from pyinduct import placeholder as ph
from pyinduct import utils as ut
from pyinduct import trajectory as tr
from pyinduct import eigenfunctions as ef
from pyinduct import simulation as sim
from pyinduct import visualization as vis
import matplotlib.pyplot as plt
from numbers import Number
import pyqtgraph as pg
import sys

__author__ = 'Stefan Ecklebe'

if any([arg == 'discover' for arg in sys.argv]):
    show_plots = False
else:
    show_plots = True
    app = pg.QtGui.QApplication([])

# TODO Test for ControlLaw and LawEvaluator

class CollocatedTestCase(unittest.TestCase):

    def setUp(self):

        interval = (0, 1)
        nodes, funcs = ut.cure_interval(cr.LagrangeFirstOrder, interval, 3)
        register_functions("funcs", funcs, overwrite=True)
        x_at1 = ph.FieldVariable("funcs", location=1)
        x_dt_at1 = ph.TemporalDerivedFieldVariable("funcs", 1, location=1)
        x_dz_at0 = ph.SpatialDerivedFieldVariable("funcs", 1, location=0)

        exp_func = cr.Function(np.exp)
        register_functions("exp_func", exp_func, overwrite=True)
        exp_at1 = ph.ScalarFunction("exp_func", location=1)

        alpha = 2
        self.term1 = ph.ScalarTerm(x_dt_at1, 1 + alpha)
        self.term2 = ph.ScalarTerm(x_dz_at0, 2)
        self.term3 = ph.ScalarTerm(ph.Product(x_at1, exp_at1))

        self.weight_label = "funcs"
        self.weights = np.hstack([1, 1, 1, 2, 2, 2])

    def test_temp_term(self):
        law = ct.approximate_control_law(ct.ControlLaw([self.term1]))
        res = law(self.weights, self.weight_label)
        self.assertAlmostEqual(res, 6)

    def test_spat_term(self):
        law = ct.approximate_control_law(ct.ControlLaw([self.term2]))
        res = law(self.weights, self.weight_label)
        self.assertAlmostEqual(res, 0)

    def test_product_term(self):
        law = ct.approximate_control_law(ct.ControlLaw([self.term3]))
        res = law(self.weights, self.weight_label)
        self.assertAlmostEqual(res, 1*np.exp(1))


class ContinuousTestCase(unittest.TestCase):

    def setUp(self):
        interval = (0, 1)
        nodes, funcs = ut.cure_interval(cr.LagrangeFirstOrder, interval, 3)
        register_functions("funcs", funcs, overwrite=True)
        x = ph.FieldVariable("funcs")
        x_dt = ph.TemporalDerivedFieldVariable("funcs", 1)
        x_dz = ph.SpatialDerivedFieldVariable("funcs", 1)
        register_functions("scal_func", cr.Function(np.exp), overwrite=True)
        exp = ph.ScalarFunction("scal_func")

        alpha = 2
        self.term1 = ph.IntegralTerm(x_dt, interval, 1 + alpha)
        self.term2 = ph.IntegralTerm(x_dz, interval, 2)
        self.term3 = ph.IntegralTerm(ph.Product(x, exp), interval)

        self.weight_label = "funcs"
        self.weights = np.hstack([1, 1, 1, 2, 2, 2])

    def test_temp_term(self):
        law = ct.approximate_control_law(ct.ControlLaw([self.term1]))
        res = law(self.weights, self.weight_label)
        self.assertAlmostEqual(res, 6)

    def test_spat_term(self):
        law = ct.approximate_control_law(ct.ControlLaw([self.term2]))
        res = law(self.weights, self.weight_label)
        self.assertAlmostEqual(res, 0)

    def test_product_term(self):
        law = ct.approximate_control_law(ct.ControlLaw([self.term3]))
        res = law(self.weights, self.weight_label)
        # TODO calculate expected result
        # self.assertAlmostEqual(res, 1*np.exp(1))


class RadDirichletControlApproxTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_it(self):

        # original system parameters
        a2 = 1; a1 =0 # attention: only a2 = 1., a1 =0 supported in this test case
        a0 = 14
        param = [a2, a1, a0, None, None]

        # target system parameters (controller parameters)
        a1_t =0; a0_t =0 # attention: only a2 = 1., a1 =0 and a0 =0 supported in this test case
        param_t = [a2, a1_t, a0_t, None, None]

        # system/simulation parameters
        actuation = 'dirichlet'
        boundary_condition = 'dirichlet'
        l = 1.; spatial_domain = (0, l); spatial_disc = 10 # attention: only l=1. supported in this test case
        T = 1; temporal_domain = (0, T); temporal_disc = 1e2
        n = 10

        # eigenvalues /-functions original system
        eig_freq = np.array([(i+1)*np.pi/l for i in xrange(n)])
        eig_values = a0 - a2*eig_freq**2 - a1**2/4./a2
        norm_fac = np.ones(eig_freq.shape)*np.sqrt(2)
        eig_funcs = np.asarray([ut.SecondOrderDirichletEigenfunction(eig_freq[i], param, spatial_domain, norm_fac[i]) for i in range(n)])
        register_functions("eig_funcs", eig_funcs, overwrite=True)

        # eigenfunctions target system
        eig_freq_t = np.sqrt(-eig_values.astype(complex))
        norm_fac_t = norm_fac * eig_freq / eig_freq_t
        eig_funcs_t = np.asarray([ut.SecondOrderDirichletEigenfunction(eig_freq_t[i], param_t, spatial_domain, norm_fac_t[i]) for i in range(n)])
        register_functions("eig_funcs_t", eig_funcs_t, overwrite=True)

        # derive initial field variable x(z,0) and weights
        start_state = cr.Function(lambda z: 0., domain=(0, l))
        initial_weights = cr.project_on_initial_functions(start_state, eig_funcs)

        # init trajectory / input of target system
        traj = tr.RadTrajectory(l, T, param_t, boundary_condition, actuation)

        # init controller
        x_at_1 = ph.FieldVariable("eig_funcs", location=1)
        xt_at_1 = ph.FieldVariable("eig_funcs_t", weight_label="eig_funcs", location=1)
        controller = ct.Controller(ct.ControlLaw([ph.ScalarTerm(x_at_1, 1), ph.ScalarTerm(xt_at_1, -1)]))

        # input with feedback
        control_law = sim.Mixer([traj, controller])

        # determine (A,B) with modal-transfomation
        A = np.diag(eig_values)
        B = -a2*np.array([eig_funcs[i].derive()(l) for i in xrange(n)])
        ss = sim.StateSpace("eig_funcs", A, B)

        # simulate
        t, q = sim.simulate_state_space(ss, control_law, initial_weights, temporal_domain, time_step=T/temporal_disc)
        # TODO: get/plot x'(z,t) data and test (assert) result

        # display results
        if show_plots:
            eval_d = ut.evaluate_approximation(q, "eig_funcs", t, spatial_domain, l/spatial_disc)
            win1 = vis.AnimatedPlot([eval_d], title="Test")
            win2 = vis.SurfacePlot(eval_d)
            app.exec_()


class RadRobinControlApproxTest(unittest.TestCase):
    """
    """
    def setUp(self):
        pass

    def test_it(self):

        # original system parameters
        a2 = 1.5; a1 = 2.5; a0 = 28; alpha = -2; beta = -3
        param = [a2, a1, a0, alpha, beta]
        adjoint_param = ut.get_adjoint_rad_robin_evp_param(param)

        # target system parameters (controller parameters)
        a1_t = -5; a0_t = -25; alpha_t = 3; beta_t = 2
        # a1_t = a1; a0_t = a0; alpha_t = alpha; beta_t = beta
        param_t = [a2, a1_t, a0_t, alpha_t, beta_t]

        # original intermediate ("_i") and traget intermediate ("_ti") system parameters
        _, _, a0_i, alpha_i, beta_i = ut.transform2intermediate(param)
        _, _, a0_ti, alpha_ti, beta_ti = ut.transform2intermediate(param_t)

        # system/simulation parameters
        actuation = 'robin'
        boundary_condition = 'robin'
        l = 1.; spatial_domain = (0, l); spatial_disc = 10
        T = 1; temporal_domain = (0, T); temporal_disc = 1e2
        n = 10

        # create (not normalized) eigenfunctions
        rad_eig_val = ut.RadRobinEigenvalues(param, l, n)
        eig_val = rad_eig_val.eig_values
        eig_freq = rad_eig_val.eig_freq
        init_eig_funcs = np.array([ut.SecondOrderRobinEigenfunction(om, param, spatial_domain) for om in eig_freq])
        init_adjoint_eig_funcs = np.array([ut.SecondOrderRobinEigenfunction(om, adjoint_param, spatial_domain) for om in eig_freq])

        # normalize eigenfunctions and adjoint eigenfunctions
        adjoint_and_eig_funcs = [cr.normalize_function(init_eig_funcs[i], init_adjoint_eig_funcs[i]) for i in range(n)]
        eig_funcs = np.array([f_tuple[0] for f_tuple in adjoint_and_eig_funcs])
        adjoint_eig_funcs = np.array([f_tuple[1] for f_tuple in adjoint_and_eig_funcs])

        # eigenfunctions from target system ("_t")
        eig_freq_t = np.sqrt(-a1_t**2/4/a2**2 + (a0_t - eig_val)/a2)
        eig_funcs_t = np.array([ut.SecondOrderRobinEigenfunction(eig_freq_t[i], param_t, spatial_domain).scale(eig_funcs[i](0)) for i in range(n)])

        # register eigenfunctions
        register_functions("eig_funcs", eig_funcs, overwrite=True)
        register_functions("adjoint_eig_funcs", adjoint_eig_funcs, overwrite=True)
        register_functions("eig_funcs_t", eig_funcs_t, overwrite=True)

        # derive initial field variable x(z,0) and weights
        start_state = cr.Function(lambda z: 0., domain=(0, l))
        initial_weights = cr.project_on_initial_functions(start_state, adjoint_eig_funcs)

        # controller initialization
        x_at_l = ph.FieldVariable("eig_funcs", location=l)
        xd_at_l = ph.SpatialDerivedFieldVariable("eig_funcs", 1, location=l)
        x_t_at_l = ph.FieldVariable("eig_funcs_t", weight_label="eig_funcs", location=l)
        xd_t_at_l = ph.SpatialDerivedFieldVariable("eig_funcs_t", 1, weight_label="eig_funcs", location=l)
        combined_transform = lambda z: np.exp((a1_t-a1)/2/a2*z)
        int_kernel_zz = lambda z: alpha_ti - alpha_i + (a0_i-a0_ti)/2/a2*z
        controller = ct.Controller(
            ct.ControlLaw([ph.ScalarTerm(x_at_l, (beta_i-beta_ti-int_kernel_zz(l))),
                           ph.ScalarTerm(x_t_at_l, -beta_ti*combined_transform(l)),
                           ph.ScalarTerm(x_at_l, beta_ti),
                           ph.ScalarTerm(xd_t_at_l, -combined_transform(l)),
                           ph.ScalarTerm(x_t_at_l, -a1_t/2/a2*combined_transform(l)),
                           ph.ScalarTerm(xd_at_l, 1),
                           ph.ScalarTerm(x_at_l, a1/2/a2+int_kernel_zz(l))
                           ]))

        # init trajectory
        traj = tr.RadTrajectory(l, T, param_t, boundary_condition, actuation)
        traj.scale = combined_transform(l)

        # input with feedback
        control_law = sim.Mixer([traj, controller])
        # control_law = sim.Mixer([traj])

        # determine (A,B) with modal-transfomation
        A = np.diag(np.real(eig_val))
        B = a2*np.array([adjoint_eig_funcs[i](l) for i in xrange(len(eig_freq))])
        ss_modal = sim.StateSpace("eig_funcs", A, B)

        # simulate
        t, q = sim.simulate_state_space(ss_modal, control_law, initial_weights, temporal_domain, time_step=T/temporal_disc)

        # display results
        if show_plots:
            eval_d = ut.evaluate_approximation(q, "eig_funcs", t, spatial_domain, l/spatial_disc)
            win1 = vis.AnimatedPlot([eval_d], title="Test")
            win2 = vis.SurfacePlot(eval_d)
            app.exec_()


class RadRobinGenericBacksteppingControlllerTest(unittest.TestCase):
    """
    """
    def setUp(self):

        # original system parameters
        a2 = 1.5; a1 = 2.5; a0 = 28; alpha = -2; beta = -3
        self.param = [a2, a1, a0, alpha, beta]
        adjoint_param = ut.get_adjoint_rad_robin_evp_param(self.param)

        # target system parameters (controller parameters)
        a1_t = -5; a0_t = -25; alpha_t = 3; beta_t = 2
        # a1_t = a1; a0_t = a0; alpha_t = alpha; beta_t = beta
        self.param_t = [a2, a1_t, a0_t, alpha_t, beta_t]

        # original intermediate ("_i") and traget intermediate ("_ti") system parameters
        _, _, a0_i, self.alpha_i, self.beta_i = ut.transform2intermediate(self.param)
        self.param_i = a2, 0, a0_i, self.alpha_i, self.beta_i
        _, _, a0_ti, self.alpha_ti, self.beta_ti = ut.transform2intermediate(self.param_t)
        self.param_ti = a2, 0, a0_ti, self.alpha_ti, self.beta_ti

        # system/simulation parameters
        actuation = 'robin'
        boundary_condition = 'robin'
        self.l = 1.; self.spatial_domain = (0, self.l); self.spatial_disc = 30
        self.T = 1; self.temporal_domain = (0, self.T); self.temporal_disc = 1e2
        self.n = 10

        # create (not normalized) eigenfunctions
        eig_freq, self.eig_val = ut.compute_rad_robin_eigenfrequencies(self.param, self.l, self.n)
        init_eig_funcs = np.array([ut.SecondOrderRobinEigenfunction(om, self.param, self.spatial_domain) for om in eig_freq])
        init_adjoint_eig_funcs = np.array([ut.SecondOrderRobinEigenfunction(om, adjoint_param, self.spatial_domain) for om in eig_freq])

        # normalize eigenfunctions and adjoint eigenfunctions
        adjoint_and_eig_funcs = [cr.normalize_function(init_eig_funcs[i], init_adjoint_eig_funcs[i]) for i in range(self.n)]
        eig_funcs = np.array([f_tuple[0] for f_tuple in adjoint_and_eig_funcs])
        self.adjoint_eig_funcs = np.array([f_tuple[1] for f_tuple in adjoint_and_eig_funcs])

        # eigenfunctions from target system ("_t")
        eig_freq_t = np.sqrt(-a1_t**2/4/a2**2 + (a0_t - self.eig_val)/a2)
        eig_funcs_t = np.array([ut.SecondOrderRobinEigenfunction(eig_freq_t[i], self.param_t, self.spatial_domain).scale(eig_funcs[i](0)) for i in range(self.n)])

        for i in [0,3,6]:
            plt.figure()
            z = np.linspace(0, self.l)
            plt.plot(z, eig_funcs[i](z), z, eig_funcs_t[i](z))
            plt.show()

        # create testfunctions
        nodes, self.fem_funcs = ut.cure_interval(cr.LagrangeFirstOrder,
                                            self.spatial_domain,
                                            node_count=self.spatial_disc)

        # register eigenfunctions
        register_functions("eig_funcs", eig_funcs, overwrite=True)
        register_functions("adjoint_eig_funcs", self.adjoint_eig_funcs, overwrite=True)
        register_functions("eig_funcs_t", eig_funcs_t, overwrite=True)
        register_functions("fem_funcs", self.fem_funcs, overwrite=True)

        # init trajectory
        self.traj = tr.RadTrajectory(self.l, self.T, self.param_ti, boundary_condition, actuation)

        # original () and target (_t) field variable
        fem_field_variable = ph.FieldVariable("fem_funcs", location=self.l)
        field_variable = ph.FieldVariable("eig_funcs", location=self.l)
        d_field_variable = ph.SpatialDerivedFieldVariable("eig_funcs", 1, location=self.l)
        field_variable_t = ph.FieldVariable("eig_funcs_t", weight_label="eig_funcs", location=self.l)
        d_field_variable_t = ph.SpatialDerivedFieldVariable("eig_funcs_t", 1, weight_label="eig_funcs", location=self.l)
        # intermediate (_i) and target intermediate (_ti) transformations by z=l
        self.transform_i = lambda z: np.exp(a1/2/a2*z)         # x_i  = x   * transform_i
        self.transform_ti = lambda z: np.exp(a1_t/2/a2*z)      # x_ti = x_t * transform_ti
        print self.transform_i(self.l)
        print self.transform_i(-self.l)
        print self.transform_ti(self.l)
        # intermediate (_i) and target intermediate (_ti) field variable (list of scalar terms = sum of scalar terms)
        self.x_fem_i_at_l = [ph.ScalarTerm(fem_field_variable, self.transform_i(self.l))]
        self.x_i_at_l = [ph.ScalarTerm(field_variable, self.transform_i(self.l))]
        self.xd_i_at_l = [ph.ScalarTerm(d_field_variable, self.transform_i(self.l)),
                     ph.ScalarTerm(field_variable, self.transform_i(self.l)*a1/2/a2)]
        self.x_ti_at_l = [ph.ScalarTerm(field_variable_t, self.transform_ti(self.l))]
        self.xd_ti_at_l = [ph.ScalarTerm(d_field_variable_t, self.transform_ti(self.l)),
                      ph.ScalarTerm(field_variable_t, self.transform_ti(self.l)*a1_t/2/a2)]

        # discontinuous operator (Kx)(t) = int_kernel_zz(l)*x(l,t)
        self.int_kernel_zz = lambda z: self.alpha_ti - self.alpha_i + (a0_i-a0_ti)/2/a2*z
        print self.int_kernel_zz(self.l)

    def test_fem(self):
        controller = ut.get_parabolic_robin_backstepping_controller(state=self.x_fem_i_at_l,
                                                                    approx_state=self.x_i_at_l,
                                                                    d_approx_state=self.xd_i_at_l,
                                                                    approx_target_state=self.x_ti_at_l,
                                                                    d_approx_target_state=self.xd_ti_at_l,
                                                                    integral_kernel_zz=self.int_kernel_zz(self.l),
                                                                    original_boundary_param=(self.alpha_i, self.beta_i),
                                                                    target_boundary_param=(self.alpha_ti, self.beta_ti),
                                                                    trajectory=self.traj,
                                                                    scale=self.transform_i(-self.l))

        # determine (A,B) with modal-transfomation
        rad_pde = ut.get_parabolic_robin_weak_form("fem_funcs", "fem_funcs", controller, self.param, self.spatial_domain)
        cf = sim.parse_weak_formulation(rad_pde)
        ss_weak = cf.convert_to_state_space()

        # simulate
        t, q = sim.simulate_state_space(ss_weak, cf.input_function, np.zeros((len(self.fem_funcs))),
                                        self.temporal_domain, time_step=self.T/self.temporal_disc)

        # display results
        if show_plots:
            eval_d = ut.evaluate_approximation(q, "fem_funcs", t, self.spatial_domain, self.l/self.spatial_disc)
            win1 = vis.AnimatedPlot([eval_d], title="Test")
            win2 = vis.SurfacePlot(eval_d)
            app.exec_()

    def test_modal(self):
        a2, a1, a0, alpha, beta = self.param
        controller = ut.get_parabolic_robin_backstepping_controller(state=self.x_i_at_l,
                                                                    approx_state=self.x_i_at_l,
                                                                    d_approx_state=self.xd_i_at_l,
                                                                    approx_target_state=self.x_ti_at_l,
                                                                    d_approx_target_state=self.xd_ti_at_l,
                                                                    integral_kernel_zz=self.int_kernel_zz(self.l),
                                                                    original_boundary_param=(self.alpha_i, self.beta_i),
                                                                    target_boundary_param=(self.alpha_ti, self.beta_ti),
                                                                    trajectory=self.traj,
                                                                    scale=self.transform_i(-self.l))

        # determine (A,B) with modal-transfomation
        A = np.diag(np.real(self.eig_val))
        B = a2*np.array([self.adjoint_eig_funcs[i](self.l) for i in xrange(self.n)])
        ss_modal = sim.StateSpace("eig_funcs", A, B)

        # simulate
        t, q = sim.simulate_state_space(ss_modal, controller, np.zeros((len(self.adjoint_eig_funcs))),
                                        self.temporal_domain, time_step=self.T/self.temporal_disc)

        # display results
        if show_plots:
            eval_d = ut.evaluate_approximation(q, "eig_funcs", t, self.spatial_domain, self.l/self.spatial_disc)
            win1 = vis.AnimatedPlot([eval_d], title="Test")
            win2 = vis.SurfacePlot(eval_d)
            app.exec_()


class RadRobinSpatiallyVaryingCoefficientControlllerTest(unittest.TestCase):
    """
    """
    def test_it(self):

        # system/simulation parameters
        actuation = 'robin'
        boundary_condition = 'robin'
        self.l = 1.; self.spatial_domain = (0, self.l); self.spatial_disc = 30
        self.T = 1; self.temporal_domain = (0, self.T); self.temporal_disc = 1e2
        self.n = 10

        # original system parameters
        a2 = 1.5
        a1_z = cr.Function(lambda z: 2+np.sin(2*np.pi/self.l*z), derivative_handles=[lambda z: 2*np.pi/self.l*np.cos(2*np.pi/self.l*z)])
        a0_z = lambda z: 10+5*np.cos(5*np.pi/self.l*z)
        alpha = -2
        beta = -3
        # a2 = 1.5; a1_z = cr.Function(lambda z: 1+z, derivative_handles=[lambda z: 1]); a0_z = lambda z: 2-z; alpha = -2; beta = -3
        # a2 = 1.5; a1_z = cr.Function(lambda z: 2.5, derivative_handles=[lambda z: 0]); a0_z = lambda z: 28; alpha = -2; beta = -3
        self.param = [a2, a1_z, a0_z, alpha, beta]

        # target system parameters (controller parameters)
        a1_t = -5; a0_t = -25; alpha_t = 3; beta_t = 2
        self.param_t = [a2, a1_t, a0_t, alpha_t, beta_t]
        adjoint_param_t = ut.get_adjoint_rad_robin_evp_param(self.param_t)

        # original intermediate ("_i") and traget intermediate ("_ti") system parameters
        _, _, a0_i, alpha_i, beta_i = ut.transform2intermediate(self.param, d_end=self.l)
        self.param_i = a2, 0, a0_i, alpha_i, beta_i
        _, _, a0_ti, alpha_ti, beta_ti = ut.transform2intermediate(self.param_t)
        self.param_ti = a2, 0, a0_ti, alpha_ti, beta_ti

        # create (not normalized) target (_t) eigenfunctions
        eig_freq_t, self.eig_val_t = ut.compute_rad_robin_eigenfrequencies(self.param_t, self.l, self.n)
        init_eig_funcs_t = np.array([ut.SecondOrderRobinEigenfunction(om, self.param_t, self.spatial_domain) for om in eig_freq_t])
        init_adjoint_eig_funcs_t = np.array([ut.SecondOrderRobinEigenfunction(om, adjoint_param_t, self.spatial_domain) for om in eig_freq_t])

        # normalize eigenfunctions and adjoint eigenfunctions
        adjoint_and_eig_funcs_t = [cr.normalize_function(init_eig_funcs_t[i], init_adjoint_eig_funcs_t[i]) for i in range(self.n)]
        eig_funcs_t = np.array([f_tuple[0] for f_tuple in adjoint_and_eig_funcs_t])
        self.adjoint_eig_funcs_t = np.array([f_tuple[1] for f_tuple in adjoint_and_eig_funcs_t])

        # # transformed original eigenfunctions
        self.eig_funcs = [ef.TransformedSecondOrderEigenfunction(self.eig_val_t[i],
                                                                 [eig_funcs_t[i](0), alpha*eig_funcs_t[i](0), 0, 0],
                                                                 [a2, a1_z, a0_z],
                                                                 np.linspace(0, self.l, 1e4))
                                      for i in range(self.n) ]

        # create testfunctions
        nodes, self.fem_funcs = ut.cure_interval(cr.LagrangeFirstOrder,
                                            self.spatial_domain,
                                            node_count=self.spatial_disc)

        # register functions
        register_functions("eig_funcs_t", eig_funcs_t, overwrite=True)
        register_functions("adjoint_eig_funcs_t", self.adjoint_eig_funcs_t, overwrite=True)
        register_functions("eig_funcs", self.eig_funcs, overwrite=True)
        register_functions("fem_funcs", self.fem_funcs, overwrite=True)

        # init trajectory
        self.traj = tr.RadTrajectory(self.l, self.T, self.param_ti, boundary_condition, actuation)

        # original () and target (_t) field variable
        fem_field_variable = ph.FieldVariable("fem_funcs", location=self.l)
        field_variable_t = ph.FieldVariable("eig_funcs_t", weight_label="eig_funcs", location=self.l)
        d_field_variable_t = ph.SpatialDerivedFieldVariable("eig_funcs_t", 1, weight_label="eig_funcs", location=self.l)
        field_variable = ph.FieldVariable("eig_funcs", location=self.l)
        d_field_variable = ph.SpatialDerivedFieldVariable("eig_funcs", 1, location=self.l)
        # intermediate (_i) and target intermediate (_ti) transformations by z=l
        self.transform_i_at_l = np.exp(cr.integrate_function(lambda z: a1_z(z)/2/a2, [(0, self.l)])[0])   # x_i  = x   * transform_i
        self.inv_transform_i_at_l = np.exp(-cr.integrate_function(lambda z: a1_z(z)/2/a2, [(0, self.l)])[0])
        self.transform_ti_at_l = np.exp(a1_t/2/a2*self.l)                                                       # x_ti = x_t * transform_ti
        # intermediate (_i) and target intermediate (_ti) field variable (list of scalar terms = sum of scalar terms)
        self.x_fem_i_at_l = [ph.ScalarTerm(fem_field_variable, self.transform_i_at_l)]
        self.x_i_at_l = [ph.ScalarTerm(field_variable, self.transform_i_at_l)]
        self.xd_i_at_l = [ph.ScalarTerm(d_field_variable, self.transform_i_at_l),
                     ph.ScalarTerm(field_variable, self.transform_i_at_l*a1_z(self.l)/2/a2)]
        self.x_ti_at_l = [ph.ScalarTerm(field_variable_t, self.transform_ti_at_l)]
        self.xd_ti_at_l = [ph.ScalarTerm(d_field_variable_t, self.transform_ti_at_l),
                      ph.ScalarTerm(field_variable_t, self.transform_ti_at_l*a1_t/2/a2)]

        # discontinuous operator (Kx)(t) = int_kernel_zz(l)*x(l,t)
        self.int_kernel_zz = alpha_ti - alpha_i + cr.integrate_function(lambda z: (a0_i(z)-a0_ti)/2/a2, [(0, self.l)])[0]

        controller = ut.get_parabolic_robin_backstepping_controller(state=self.x_fem_i_at_l,
                                                                    approx_state=self.x_i_at_l,
                                                                    d_approx_state=self.xd_i_at_l,
                                                                    approx_target_state=self.x_ti_at_l,
                                                                    d_approx_target_state=self.xd_ti_at_l,
                                                                    integral_kernel_zz=self.int_kernel_zz,
                                                                    original_boundary_param=(alpha_i, beta_i),
                                                                    target_boundary_param=(alpha_ti, beta_ti),
                                                                    trajectory=self.traj,
                                                                    scale=self.inv_transform_i_at_l)

        rad_pde = ut.get_parabolic_robin_weak_form("fem_funcs", "fem_funcs", controller, self.param, self.spatial_domain)
        cf = sim.parse_weak_formulation(rad_pde)
        ss_weak = cf.convert_to_state_space()

        # simulate
        t, q = sim.simulate_state_space(ss_weak, cf.input_function, np.zeros((len(self.fem_funcs))),
                                        self.temporal_domain, time_step=self.T/self.temporal_disc)

        # display results
        if show_plots:
            eval_d = ut.evaluate_approximation(q, "fem_funcs", t, self.spatial_domain, self.l/self.spatial_disc)
            win1 = vis.AnimatedPlot([eval_d], title="Test")
            win2 = vis.SurfacePlot(eval_d)
            app.exec_()


class SimulationError(unittest.TestCase):
    """
    """
    def setUp(self):
        pass

    def test_simulate_state_space_error(self):
        pass
        # original system parameters
        a2 = 1.5; a1 = 2.5; a0 = 28; alpha = -2; beta = -3
        param = [a2, a1, a0, alpha, beta]
        adjoint_param = ut.get_adjoint_rad_robin_evp_param(param)

        # target system parameters (controller parameters)
        a1_t = -5; a0_t = -25; alpha_t = 3; beta_t = 2
        # a1_t = a1; a0_t = a0; alpha_t = alpha; beta_t = beta
        param_t = [a2, a1_t, a0_t, alpha_t, beta_t]

        # original intermediate ("_i") and traget intermediate ("_ti") system parameters
        _, _, a0_i, alpha_i, beta_i = ut.transform2intermediate(param)
        param_i = a2, 0, a0_i, alpha_i, beta_i
        _, _, a0_ti, alpha_ti, beta_ti = ut.transform2intermediate(param_t)
        param_ti = a2, 0, a0_ti, alpha_ti, beta_ti

        # system/simulation parameters
        actuation = 'robin'
        boundary_condition = 'robin'
        l = 1.; spatial_domain = (0, l); spatial_disc = 10
        T = 1; temporal_domain = (0, T); temporal_disc = 1e2
        n = 10

        # create (not normalized) eigenfunctions
        rad_eig_val = ut.RadRobinEigenvalues(param, l, n)
        eig_val = rad_eig_val.eig_values
        eig_freq = rad_eig_val.eig_freq
        init_eig_funcs = np.array([ut.SecondOrderRobinEigenfunction(om, param, spatial_domain) for om in eig_freq])
        init_adjoint_eig_funcs = np.array([ut.SecondOrderRobinEigenfunction(om, adjoint_param, spatial_domain) for om in eig_freq])

        # normalize eigenfunctions and adjoint eigenfunctions
        adjoint_and_eig_funcs = [cr.normalize_function(init_eig_funcs[i], init_adjoint_eig_funcs[i]) for i in range(n)]
        eig_funcs = np.array([f_tuple[0] for f_tuple in adjoint_and_eig_funcs])
        adjoint_eig_funcs = np.array([f_tuple[1] for f_tuple in adjoint_and_eig_funcs])

        # eigenfunctions from target system ("_t")
        eig_freq_t = np.sqrt(-a1_t**2/4/a2**2 + (a0_t - eig_val)/a2)
        eig_funcs_t = np.array([ut.SecondOrderRobinEigenfunction(eig_freq_t[i], param_t, spatial_domain).scale(eig_funcs[i](0)) for i in range(n)])

        # register eigenfunctions
        register_functions("eig_funcs", eig_funcs, overwrite=True)
        register_functions("adjoint_eig_funcs", adjoint_eig_funcs, overwrite=True)
        register_functions("eig_funcs_t", eig_funcs_t, overwrite=True)

        # derive initial field variable x(z,0) and weights
        start_state = cr.Function(lambda z: 0., domain=(0, l))
        initial_weights = cr.project_on_initial_functions(start_state, adjoint_eig_funcs)

        # init trajectory
        traj = tr.RadTrajectory(l, T, param_t, boundary_condition, actuation)

        # alternetiv controller initialization (shorter but less generic)
        x_at_l = ph.FieldVariable("eig_funcs", location=l)
        xd_at_l = ph.SpatialDerivedFieldVariable("eig_funcs", 1, location=l)
        x_t_at_l = ph.FieldVariable("eig_funcs_t", weight_label="eig_funcs", location=l)
        xd_t_at_l = ph.SpatialDerivedFieldVariable("eig_funcs_t", 1, weight_label="eig_funcs", location=l)
        combined_transform = lambda z: np.exp((a1_t-a1)/2/a2*z)
        int_kernel_zz = lambda z: alpha_ti - alpha_i + (a0_i-a0_ti)/2/a2*z
        control_law = ct.Controller(
            ct.ControlLaw([ph.ScalarTerm(x_at_l, (beta_i-beta_ti-int_kernel_zz(l))),
                           ph.ScalarTerm(x_t_at_l, -beta_ti*combined_transform(l)),
                           ph.ScalarTerm(x_at_l, beta_ti),
                           ph.ScalarTerm(xd_t_at_l, -combined_transform(l)),
                           ph.ScalarTerm(x_t_at_l, -a1_t/2/a2*combined_transform(l)),
                           ph.ScalarTerm(xd_at_l, l),
                           ph.ScalarTerm(x_at_l, a1/2/a2+int_kernel_zz(l))
                           ]))
        # input with feedback
        controller = sim.Mixer([traj, control_law])

        # determine (A,B) with modal-transfomation
        A = np.diag(np.real(eig_val))
        B = a2*np.array([adjoint_eig_funcs[i](l) for i in xrange(len(eig_freq))])
        ss_modal = sim.StateSpace("eig_funcs", A, B)

        # simulate
        # t, q = sim.simulate_state_space(ss_modal, control_law, initial_weights, temporal_domain, time_step=T/temporal_disc)

    def test_equivalent_problem(self):
        a = np.arange(1,3)
        b = np.array([1])
        # np.dot(a, b)

