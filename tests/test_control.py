from __future__ import division
import unittest
import numpy as np
from pyinduct import register_functions
from pyinduct import core as cr
from pyinduct import control as ct
from pyinduct import placeholder as ph
from pyinduct import utils as ut
from pyinduct import trajectory as tr
from pyinduct import simulation as sim
from pyinduct import visualization as vis
from numbers import Number
import pyqtgraph as pg

__author__ = 'Stefan Ecklebe'

# show_plots = False
show_plots = True

if show_plots:
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


class ReaAdvDifDirichletControlApproxTest(unittest.TestCase):

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
        eig_funcs = np.asarray([ut.ReaAdvDifDirichletEigenfunction(eig_freq[i], param, spatial_domain, norm_fac[i]) for i in range(n)])
        register_functions("eig_funcs", eig_funcs, overwrite=True)

        # eigenfunctions target system
        eig_freq_t = np.sqrt(-eig_values.astype(complex))
        norm_fac_t = norm_fac * eig_freq / eig_freq_t
        eig_funcs_t = np.asarray([ut.ReaAdvDifDirichletEigenfunction(eig_freq_t[i], param_t, spatial_domain, norm_fac_t[i]) for i in range(n)])
        register_functions("eig_funcs_t", eig_funcs_t, overwrite=True)

        # derive initial field variable x(z,0) and weights
        start_state = cr.Function(lambda z: 0., domain=(0, l))
        initial_weights = cr.project_on_initial_functions(start_state, eig_funcs)

        # init trajectory / input of target system
        traj = tr.ReaAdvDifTrajectory(l, T, param_t, boundary_condition, actuation)

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


class ReaAdvDifRobinControlApproxTest(unittest.TestCase):
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
        # a1_t = -5; a0_t = -25; alpha_t = 3; beta_t = 2
        a1_t = a1; a0_t = a0; alpha_t = alpha; beta_t = beta
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
        rad_eig_val = ut.ReaAdvDifRobinEigenvalues(param, l, n)
        eig_val = rad_eig_val.eig_values
        eig_freq = rad_eig_val.eig_freq
        init_eig_funcs = np.array([ut.ReaAdvDifRobinEigenfunction(om, param, spatial_domain) for om in eig_freq])
        init_adjoint_eig_funcs = np.array([ut.ReaAdvDifRobinEigenfunction(om, adjoint_param, spatial_domain) for om in eig_freq])

        # normalize eigenfunctions and adjoint eigenfunctions
        adjoint_and_eig_funcs = [cr.normalize_function(init_eig_funcs[i], init_adjoint_eig_funcs[i]) for i in range(n)]
        eig_funcs = np.array([f_tuple[0] for f_tuple in adjoint_and_eig_funcs])
        adjoint_eig_funcs = np.array([f_tuple[1] for f_tuple in adjoint_and_eig_funcs])

        # eigenfunctions from target system ("_t")
        eig_freq_t = np.sqrt(-a1_t**2/4/a2**2 + (a0_t - eig_val)/a2)
        eig_funcs_t = np.array([ut.ReaAdvDifRobinEigenfunction(eig_freq_t[i], param_t, spatial_domain).scale(eig_funcs[i](0)) for i in range(n)])

        # register eigenfunctions
        register_functions("eig_funcs", eig_funcs, overwrite=True)
        register_functions("adjoint_eig_funcs", adjoint_eig_funcs, overwrite=True)
        register_functions("eig_funcs_t", eig_funcs_t, overwrite=True)

        # derive initial field variable x(z,0) and weights
        start_state = cr.Function(lambda z: 0., domain=(0, l))
        initial_weights = cr.project_on_initial_functions(start_state, adjoint_eig_funcs)

        # init trajectory
        traj = tr.ReaAdvDifTrajectory(l, T, param_ti, boundary_condition, actuation)

        # intermediate and target intermediate field variable
        # original () and target (_t) field variable
        field_variable = ph.FieldVariable("eig_funcs", location=l)
        d_field_variable = ph.SpatialDerivedFieldVariable("eig_funcs", 1, location=l)
        field_variable_t = ph.FieldVariable("eig_funcs_t", weight_label="eig_funcs", location=l)
        d_field_variable_t = ph.SpatialDerivedFieldVariable("eig_funcs_t", 1, weight_label="eig_funcs", location=l)
        # wrap field variables in scalar terms
        x_at_l = [ph.ScalarTerm(field_variable)]
        xd_at_l = [ph.ScalarTerm(d_field_variable)]
        x_t_at_l = [ph.ScalarTerm(field_variable_t)]
        xd_t_at_l = [ph.ScalarTerm(d_field_variable_t)]
        # intermediate (_i) and target intermediate (_ti) transformations by z=l
        transform_i = lambda z: np.exp(a1/2/a2*z)         # x_i  = x   * transform_i
        transform_ti = lambda z: np.exp(a1_t/2/a2*z)      # x_ti = x_t * transform_ti
        # intermediate (_i) and target intermediate (_ti) field variable (scalar terms)
        x_i_at_l = [ph.ScalarTerm(field_variable, transform_i(l))]
        xd_i_at_l = [ph.ScalarTerm(d_field_variable, transform_i(l)),
                     ph.ScalarTerm(field_variable, transform_i(l)*a1/2/a2)]
        x_ti_at_l = [ph.ScalarTerm(field_variable_t, transform_ti(l))]
        xd_ti_at_l = [ph.ScalarTerm(d_field_variable_t, transform_ti(l)),
                      ph.ScalarTerm(field_variable_t, transform_ti(l)*a1_t/2/a2)]

        # discontinuous operator (Kx)(t) = int_kernel_zz(l)*x(l,t)
        int_kernel_zz = lambda z: alpha_ti - alpha_i + (a0_i-a0_ti)/2/a2*z

        # controller initialization
        control_law = ut.get_parabolic_robin_backstepping_controller(state=x_at_l,
                                                                     approx_state=x_at_l,
                                                                     d_approx_state=xd_at_l,
                                                                     approx_target_state=x_t_at_l,
                                                                     d_approx_target_state=xd_t_at_l,
                                                                     unsteady_operator_factor=int_kernel_zz(l),
                                                                     original_param=param_i,
                                                                     target_param=param_ti,
                                                                     trajectory=traj,
                                                                     scale=transform_i(-l))


        # # controller initialization
        # x_at_l = ph.FieldVariable("eig_funcs", location=l)
        # xd_at_l = ph.SpatialDerivedFieldVariable("eig_funcs", 1, location=l)
        # x_t_at_l = ph.FieldVariable("eig_funcs_t", weight_label="eig_funcs", location=l)
        # xd_t_at_l = ph.SpatialDerivedFieldVariable("eig_funcs_t", 1, weight_label="eig_funcs", location=l)
        # combined_transform = lambda z: np.exp((a1_t-a1)/2/a2*z)
        # int_kernel_zz = lambda z: alpha_ti - alpha_i + (a0_i-a0_ti)/2/a2*z
        # controller = ct.Controller(
        #     ct.ControlLaw([ph.ScalarTerm(x_at_l, (beta_i-beta_ti-int_kernel_zz(l))),
        #                    ph.ScalarTerm(x_t_at_l, -beta_ti*combined_transform(l)),
        #                    ph.ScalarTerm(x_at_l, beta_ti),
        #                    ph.ScalarTerm(xd_t_at_l, -combined_transform(l)),
        #                    ph.ScalarTerm(x_t_at_l, -a1_t/2/a2*combined_transform(l)),
        #                    ph.ScalarTerm(xd_at_l, l),
        #                    ph.ScalarTerm(x_at_l, a1/2/a2+int_kernel_zz(l))
        #                    ]))


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



class InheriateTest(unittest.TestCase):
    """
    """
    def setUp(self):
        self.traj = tr.ReaAdvDifTrajectory(1,1,[1,1,1,1,1],"robin","robin")
    def test_sim(self):
        print isinstance(self.traj, sim.SimulationInput)