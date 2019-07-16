r"""
Implementation of the approximation scheme presented in [RW2018]_.
The system

.. math::
    :nowrap:

    \begin{align*}
        \dot x(z,t) &= a_2 x''(z,t) + a_1 x'(z,t) + a_0 x(z,t) \\
        x'(0,t) &= \alpha x(0,t) \\
        x'(1,t) &= -\beta x(1,t) + u(t) \\
    \end{align*}

and the observer

.. math::
    :nowrap:

    \begin{align*}
        \dot{\hat{x}}(z,t) &= a_2 \hat x''(z,t) + a_1 \hat x'(z,t)
                        + a_0 \hat x(z,t) + l(z) \tilde y(t)\\
        \hat x'(0,t) &= \alpha \hat x(0,t) + l_0 \tilde y(t) \\
        \hat x'(1,t) &= -\beta \hat x(1,t) + u(t) \\
    \end{align*}

are approximated with :py:class:`.LagrangeFirstOrder` (FEM) shapefunctions
and the backstepping controller and observer are approximated with
the eigenfunctions respectively the adjoint eigenfunction of the system
operator, see [RW2018]_.

Note:
    For now, only :code:`a0 = 0` and :code:`a0_t_o = 0` are supported, because
    of some limitations of the automatic observer gain transformation,
    see :py:func:`.evaluate_transformations` docstring.

References:

    .. [RW2018] Marcus Riesmeier and Frank Woittennek;
          On approximation of backstepping observers for parabolic systems with
          robin boundary conditions; In: Proceedings of the 57th IEEE,
          International Conference on Decision and Control (CDC), Miami,
          Florida, USA, December 17-19, 2018.
"""
import numpy as np
import matplotlib.pyplot as plt

import pyinduct as pi
import pyinduct.parabolic as parabolic
from pyinduct.simulation import get_sim_result
from pyinduct.tests import test_examples


def approximate_observer(obs_params, sys_params, sys_domain, sys_lbl,
                         obs_sys_lbl, test_lbl, tar_test_lbl, system_input):
    a2, a1_t, a0_t, alpha_t, beta_t = obs_params
    a2, a1, a0, alpha, beta = sys_params
    bounds = sys_domain.bounds

    int_kernel_00 = beta_t - beta - (a0_t - a0) / 2 / a2 * bounds[1]
    l0 = alpha_t - alpha + int_kernel_00

    x_sys = pi.FieldVariable(sys_lbl)
    x_obs = pi.FieldVariable(obs_sys_lbl)
    psi_fem = pi.TestFunction(obs_sys_lbl)
    psi_eig = pi.TestFunction(test_lbl)
    psi_eig_t = pi.TestFunction(tar_test_lbl, approx_label=test_lbl)

    obs_rad_pde, obs_base_labels = parabolic.get_parabolic_robin_weak_form(
        obs_sys_lbl,
        obs_sys_lbl,
        system_input,
        sys_params,
        bounds)
    obs_error = pi.StateFeedback(pi.WeakFormulation(
        [pi.ScalarTerm(x_obs(0), scale=-1), pi.ScalarTerm(x_sys(0))],
        name="observer_error"))
    gain_handle = pi.ObserverFeedback(
        pi.WeakFormulation(
            [pi.ScalarTerm(psi_fem(0), scale=a2 * l0),
             pi.ScalarTerm(psi_eig_t(0), scale=a2 * alpha_t),
             pi.ScalarTerm(psi_eig(0), scale=-a2 * alpha_t),
             pi.ScalarTerm(psi_eig_t(0).derive(order=1), scale=-a2),
             pi.ScalarTerm(psi_eig(0).derive(order=1), scale=a2),
             pi.ScalarTerm(psi_eig(0), scale=-a2 * int_kernel_00)],
            name="observer_gain"),
        obs_error)

    obs_rad_pde.terms = np.hstack((
        obs_rad_pde.terms, pi.ScalarTerm(pi.ObserverGain(gain_handle))
    ))

    return obs_rad_pde, obs_base_labels


class ReversedRobinEigenfunction(pi.SecondOrderRobinEigenfunction):
    def __init__(self, om, param, l, scale=1, max_der_order=2):
        a2, a1, a0, alpha, beta = param
        _param = a2, -a1, a0, beta, alpha
        pi.SecondOrderRobinEigenfunction.__init__(self, om, _param, l,
                                                  scale, max_der_order)

        self.function_handle = self.function_handle_factory(
            self.function_handle, l)
        self.derivative_handles = [
            self.function_handle_factory(handle, l, ord + 1) for
            ord, handle in enumerate(self.derivative_handles)]

    def function_handle_factory(self, old_handle, l, der_order=0):
        def new_handle(z):
            return old_handle(l - z) * (-1) ** der_order

        return new_handle

    @staticmethod
    def eigfreq_eigval_hint(param, l, n_roots, show_plot=False):
        a2, a1, a0, alpha, beta = param
        _param = a2, -a1, a0, beta, alpha
        return pi.SecondOrderRobinEigenfunction.eigfreq_eigval_hint(
            _param, l, n_roots, show_plot=show_plot)


def main():
    # PARAMETERS TO VARY
    # number of eigenfunctions, used for control law approximation
    n_modal = 10
    # number FEM test functions, used for system approximation/simulation
    n_fem = 20

    # system/simulation parameters
    z_end = 1
    spat_bounds = (0, z_end)
    spatial_domain = pi.Domain(bounds=(0, z_end), num=n_fem)
    trans_time = 1
    temporal_domain = pi.Domain(bounds=(0, 1.5), num=2e3)
    actuation_type = 'robin'
    bound_cond_type = 'robin'
    n = n_modal

    # original system parameter
    a2 = 1
    a1 = 0
    a0 = 6
    alpha = -1
    beta = -1
    param = (a2, a1, a0, alpha, beta)
    param_a = pi.SecondOrderEigenfunction.get_adjoint_problem(param)

    # the given approach only works for self-adjoint problems
    assert param == param_a

    # controller target system parameters (controller parameters)
    a1_t_c = 0
    a0_t_c = -8
    alpha_t_c = 2
    beta_t_c = 2
    param_t_c = (a2, a1_t_c, a0_t_c, alpha_t_c, beta_t_c)

    # observer target system parameters (controller parameters)
    a1_t_o = 0
    a0_t_o = -16
    alpha_t_o = 3
    beta_t_o = 3
    param_t_o = (a2, a1_t_o, a0_t_o, alpha_t_o, beta_t_o)
    param_a_t_o = pi.SecondOrderEigenfunction.get_adjoint_problem(param_t_o)

    # the given approach only works for self-adjoint problems
    assert param_t_o == param_a_t_o

    # original intermediate ("_i") and
    # target intermediate ("_ti") system parameters
    _, _, a0_i, alpha_i, beta_i = parabolic.eliminate_advection_term(
        param, z_end)
    param_i = a2, 0, a0_i, alpha_i, beta_i
    _, _, a0_ti, alpha_ti, beta_ti = parabolic.eliminate_advection_term(
        param_t_c, z_end)
    param_ti = a2, 0, a0_ti, alpha_ti, beta_ti

    # create eigenfunctions (arbitrary f(0))
    eig_val, eig_funcs_init = pi.SecondOrderRobinEigenfunction.cure_interval(
        spatial_domain, param, n=n)

    # create adjoint eigenfunctions ("_a") (arbitrary f(l))
    scale_a = [func(z_end) for func in eig_funcs_init]
    _, eig_funcs_a_init = ReversedRobinEigenfunction.cure_interval(
        spatial_domain, param_a, eig_val=eig_val, scale=scale_a)

    # normalize eigenfunctions
    eig_funcs, eig_funcs_a = pi.normalize_base(eig_funcs_init, eig_funcs_a_init)

    # eigenfunctions from controller target system ("_t") (arbitrary f(0))
    scale_t = [func(0) for func in eig_funcs]
    _, eig_funcs_t = pi.SecondOrderRobinEigenfunction.cure_interval(
        spatial_domain, param_t_c, eig_val=eig_val, scale=scale_t)

    # adjoint eigenfunctions from observer target system ("_a_t") (arbitrary f(l))
    scale_a_t = [func(z_end) for func in eig_funcs_a]
    _, eig_funcs_a_t = ReversedRobinEigenfunction.cure_interval(
        spatial_domain, param_a_t_o, eig_val=eig_val, scale=scale_a_t)

    # create fem test functions
    fem_funcs = pi.LagrangeFirstOrder.cure_interval(spatial_domain)

    # register eigenfunctions
    sys_lbl = "sys_base"
    obs_sys_lbl = "obs_sys_base"
    tar_sys_lbl = "tar_sys_base"
    obs_tar_sys_lbl = "obs_tar_sys_base"
    pi.register_base(sys_lbl, fem_funcs)
    pi.register_base(obs_sys_lbl, fem_funcs)
    pi.register_base(obs_tar_sys_lbl, eig_funcs_a_t)
    ctrl_lbl = "ctrl_appr_base"
    ctrl_target_lbl = "ctrl_appr_target_base"
    ctrl_base = pi.Base(eig_funcs.fractions)
    ctrl_base.intermediate_base_lbls = [obs_sys_lbl]
    pi.register_base(ctrl_lbl, ctrl_base)
    pi.register_base(ctrl_target_lbl, eig_funcs_t)
    obs_lbl = "obs_appr_base"
    obs_target_lbl = "obs_appr_target_base"
    pi.register_base(obs_lbl, eig_funcs_a)
    pi.register_base(obs_target_lbl, eig_funcs_a_t)

    # original () and target (_t) field variable
    fem_field_variable = pi.FieldVariable(sys_lbl, location=z_end)
    field_variable = pi.FieldVariable(ctrl_lbl, location=z_end)
    field_variable_t = pi.FieldVariable(ctrl_target_lbl, location=z_end,
                                        weight_label=ctrl_lbl)

    # intermediate (_i) transformation at z=l
    # x_i  = x   * transform_i
    transform_i_l = np.exp(a1 / 2 / a2 * z_end)

    # target intermediate (_ti) transformation at z=l
    # x_ti = x_t * transform_ti
    transform_ti_l = np.exp(a1_t_c / 2 / a2 * z_end)

    # intermediate (_i) and target intermediate (_ti) field variable
    # (list of scalar terms = sum of scalar terms)
    x_fem_i_at_l = [pi.ScalarTerm(fem_field_variable, transform_i_l)]
    x_i_at_l = [pi.ScalarTerm(field_variable, transform_i_l)]
    xd_i_at_l = [pi.ScalarTerm(field_variable.derive(spat_order=1),
                               transform_i_l),
                 pi.ScalarTerm(field_variable, transform_i_l * a1 / 2 / a2)]
    x_ti_at_l = [pi.ScalarTerm(field_variable_t, transform_ti_l)]
    xd_ti_at_l = [pi.ScalarTerm(field_variable_t.derive(spat_order=1),
                                transform_ti_l),
                  pi.ScalarTerm(field_variable_t,
                                transform_ti_l * a1_t_c / 2 / a2)]

    # discontinuous operator (Kx)(t) = int_kernel_zz(l)*x(l,t)
    int_kernel_ll = alpha_ti - alpha_i + (a0_i - a0_ti) / 2 / a2 * z_end
    scale_factor = np.exp(-a1 / 2 / a2 * z_end)

    # trajectory initialization
    trajectory = parabolic.RadFeedForward(
        z_end, trans_time, param_ti, bound_cond_type, actuation_type,
        length_t=len(temporal_domain), scale=scale_factor)

    # controller initialization
    controller = parabolic.control.get_parabolic_robin_backstepping_controller(
        state=x_i_at_l, approx_state=x_i_at_l, d_approx_state=xd_i_at_l,
        approx_target_state=x_ti_at_l, d_approx_target_state=xd_ti_at_l,
        integral_kernel_ll=int_kernel_ll, original_beta=beta_i,
        target_beta=beta_ti, scale=scale_factor)

    # add as system input
    system_input = pi.SimulationInputSum([trajectory, controller])

    # system
    rad_pde, base_labels = parabolic.get_parabolic_robin_weak_form(
        sys_lbl,
        sys_lbl,
        system_input,
        param,
        spatial_domain.bounds)

    # observer
    obs_rad_pde, obs_base_labels = approximate_observer(param_t_o,
                                                        param,
                                                        spatial_domain,
                                                        sys_lbl,
                                                        obs_sys_lbl,
                                                        obs_lbl,
                                                        obs_target_lbl,
                                                        system_input)

    # desired observer error system
    obs_err_rad_pde, tar_obs_base_labels = parabolic.get_parabolic_robin_weak_form(
        obs_tar_sys_lbl,
        obs_tar_sys_lbl,
        pi.ConstantTrajectory(0),
        param_t_c,
        spatial_domain.bounds)

    # initial states/conditions
    def sys_ic(z): return .0
    def obs_ic(z): return .5

    ics = {rad_pde.name: [pi.Function(sys_ic, domain=spat_bounds)],
           obs_rad_pde.name: [pi.Function(obs_ic, domain=spat_bounds)]}

    # spatial domains
    spatial_domains = {rad_pde.name: spatial_domain,
                       obs_rad_pde.name: spatial_domain}

    # simulation
    sys_ed, obs_ed = pi.simulate_systems(
        [rad_pde, obs_rad_pde], ics, temporal_domain,
        spatial_domains)

    # evaluate desired output data
    y_d, t_d = pi.gevrey_tanh(trans_time, 40, length_t=len(temporal_domain))
    C = pi.coefficient_recursion(y_d, alpha * y_d, param)
    x_l = pi.power_series(np.array(spatial_domain), t_d, C)
    evald_traj = pi.EvalData([t_d, np.array(spatial_domain)], x_l,
                             name="x(z,t) desired")

    plots = list()
    # pyqtgraph visualization
    plots.append(pi.PgAnimatedPlot(
        [sys_ed, obs_ed, evald_traj], title="animation", replay_gain=.05))
    # matplotlib visualization
    plots.append(pi.MplSlicePlot([sys_ed, obs_ed], spatial_point=0,
                                 legend_label=["$x(0,t)$",
                                               "$\hat x(0,t)$"]))
    plt.legend(loc=4)
    plots.append(pi.MplSlicePlot([sys_ed, obs_ed], spatial_point=1,
                                 legend_label=["$x(1,t)$",
                                               "$\hat x(1,t)$"]))
    plt.legend(loc=1)
    pi.show()

    pi.tear_down((sys_lbl, obs_sys_lbl, ctrl_lbl, ctrl_target_lbl,
                  obs_lbl, obs_target_lbl) + \
                 base_labels + obs_base_labels + tar_obs_base_labels,
                 plots)


if __name__ == "__main__" or test_examples:
    main()


