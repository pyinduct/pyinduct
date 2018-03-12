from pyinduct.tests import test_examples

if __name__ == "__main__" or test_examples:
    import numpy as np
    import pyinduct as pi
    import pyinduct.parabolic as parabolic
    from pyinduct.simulation import get_sim_result


    def approximate_observer(sys_lbl, obs_sys_lbl, test_lbl, tar_test_lbl):
        a1_t, a0_t, alpha_t, beta_t = a1_t_o, a0_t_o, alpha_t_o, beta_t_o
        int_kernel_00 = beta_t - beta - (a0_t - a0) / 2 / a2 * l
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
            param,
            spatial_domain.bounds)
        obs_error = pi.Controller(pi.WeakFormulation(
            [pi.ScalarTerm(x_obs(0), scale=-1), pi.ScalarTerm(x_sys(0))],
            name="observer_error"))
        observer_gain = pi.ObserverFeedback(
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
            obs_rad_pde.terms, pi.ScalarTerm(pi.ObserverGain(observer_gain))
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


    # PARAMETERS TO VARY
    # number of eigenfunctions, used for control law approximation
    n_modal = 10
    # number FEM test functions, used for system approximation/simulation
    n_fem = 30
    # control law parameter
    # stabilizing: param_a0_t < 0
    # destabilizing: param_a0_t > 0
    param_a0_t = -10

    # system/simulation parameters
    l = 1
    T = 1
    actuation_type = 'robin'
    bound_cond_type = 'robin'
    spatial_domain = pi.Domain(bounds=(0, l), num=n_fem)
    temporal_domain = pi.Domain(bounds=(0, 1), num=500)
    n = n_modal

    # original system parameter
    a2 = 1
    a1 = 0
    a0 = 6
    alpha = 0
    beta = 0
    param = [a2, a1, a0, alpha, beta]
    adjoint_param = pi.SecondOrderEigenfunction.get_adjoint_problem(param)

    # controller target system parameters (controller parameters)
    a1_t = 0
    a0_t = param_a0_t
    alpha_t = 0
    beta_t = 0
    # a1_t = a1 a0_t = a0 alpha_t = alpha beta_t = beta
    param_t = [a2, a1_t, a0_t, alpha_t, beta_t]

    # observer target system parameters (controller parameters)
    a1_t_o = 0
    a0_t_o = param_a0_t
    alpha_t_o = 0
    beta_t_o = 0
    # a1_t = a1 a0_t = a0 alpha_t = alpha beta_t = beta
    param_t_o = [a2, a1_t_o, a0_t_o, alpha_t_o, beta_t_o]

    # original intermediate ("_i") and
    # target intermediate ("_ti") system parameters
    _, _, a0_i, alpha_i, beta_i = parabolic.eliminate_advection_term(
        param, l)
    param_i = a2, 0, a0_i, alpha_i, beta_i
    _, _, a0_ti, alpha_ti, beta_ti = parabolic.eliminate_advection_term(
        param_t, l)
    param_ti = a2, 0, a0_ti, alpha_ti, beta_ti

    # create eigenfunctions (arbitrary f(0))
    eig_val, eig_funcs_init = pi.SecondOrderRobinEigenfunction.solve_evp_hint(
        param, l, n=n)
    eig_funcs = pi.normalize_base(eig_funcs_init)

    # eigenfunctions from target system ("_t") (arbitrary f(0))
    scale_t = [func(0) for func in eig_funcs]
    _, eig_funcs_t = pi.SecondOrderRobinEigenfunction.solve_evp_hint(
        param_t, l, eig_val=eig_val, scale=scale_t)

    # create reversed eigenfunctions ("_r") (arbitrary f(l))
    # TODO: Reversed... durch Adjoint... ersetzen
    # scale_r = [func(l) for func in eig_funcs]
    # _, eig_funcs_r = ReversedRobinEigenfunction.solve_evp_hint(
    #     param, l, eig_val=eig_val, scale=scale_r)
    # plot_funcs(spatial_domain, eig_funcs_r, "dual")
    eig_funcs_r = eig_funcs

    # reversed eigenfunctions from target system ("_r_t") (arbitrary f(l))
    scale_r_t = [func(l) for func in eig_funcs_r]
    _, eig_funcs_r_t = ReversedRobinEigenfunction.solve_evp_hint(
        param_t_o, l, eig_val=eig_val, scale=scale_r_t)

    # create fem test functions
    nodes, fem_funcs = pi.cure_interval(pi.LagrangeNthOrder,
                                        spatial_domain.bounds,
                                        node_count=len(spatial_domain),
                                        order=1)

    # register eigenfunctions
    sys_lbl = "sys_base"
    obs_sys_lbl = "obs_sys_base"
    tar_sys_lbl = "tar_sys_base"
    pi.register_base(sys_lbl, fem_funcs)
    pi.register_base(obs_sys_lbl, fem_funcs)
    pi.register_base(tar_sys_lbl, eig_funcs_r_t)
    ctrl_lbl = "ctrl_appr_base"
    ctrl_target_lbl = "ctrl_appr_target_base"
    pi.register_base(ctrl_lbl, eig_funcs.set_sb_source(obs_sys_lbl))
    pi.register_base(ctrl_target_lbl, eig_funcs_t)
    obs_lbl = "obs_appr_base"
    obs_target_lbl = "obs_appr_target_base"
    pi.register_base(obs_lbl, eig_funcs_r)
    pi.register_base(obs_target_lbl, eig_funcs_r_t)

    # original () and target (_t) field variable
    fem_field_variable = pi.FieldVariable(sys_lbl, location=l)
    field_variable = pi.FieldVariable(ctrl_lbl, location=l)
    field_variable_t = pi.FieldVariable(ctrl_target_lbl, location=l,
                                        weight_label=ctrl_lbl)

    # intermediate (_i) transformation at z=l
    # x_i  = x   * transform_i
    transform_i_l = np.exp(a1 / 2 / a2 * l)

    # target intermediate (_ti) transformation at z=l
    # x_ti = x_t * transform_ti
    transform_ti_l = np.exp(a1_t / 2 / a2 * l)

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
                                transform_ti_l * a1_t / 2 / a2)]

    # discontinuous operator (Kx)(t) = int_kernel_zz(l)*x(l,t)
    int_kernel_ll = alpha_ti - alpha_i + (a0_i - a0_ti) / 2 / a2 * l
    scale_factor = np.exp(-a1 / 2 / a2 * l)

    # trajectory initialization
    trajectory = parabolic.RadFeedForward(
        l, T, param_ti, bound_cond_type, actuation_type,
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
    rad_pde.name = "system state x(z,t)"

    # observer
    obs_rad_pde, obs_base_labels = approximate_observer(
        sys_lbl, obs_sys_lbl, obs_lbl, obs_target_lbl)
    obs_rad_pde.name = "observer state \hat x(z,t)"

    # desired observer error system
    obs_err_rad_pde, tar_obs_base_labels = parabolic.get_parabolic_robin_weak_form(
        tar_sys_lbl,
        tar_sys_lbl,
        pi.ConstantTrajectory(0),
        param_t,
        spatial_domain.bounds)


    # initial states/conditions
    def sys_ic(z): return .2
    def obs_ic(z): return -.2

    ics = {rad_pde.name: [pi.Function(sys_ic)],
           obs_rad_pde.name: [pi.Function(obs_ic)]}

    # spatial domains
    spatial_domains = {rad_pde.name: spatial_domain,
                       obs_rad_pde.name: spatial_domain}

    # simulation
    weights = dict()
    sys_ed, obs_ed = pi.simulate_systems(
        [rad_pde, obs_rad_pde], ics, temporal_domain,
        spatial_domains, out=weights)

    # evaluate desired output data
    y_d, t_d = pi.gevrey_tanh(T, 40, length_t=len(temporal_domain))
    C = pi.coefficient_recursion(y_d, alpha * y_d, param)
    x_l = pi.power_series(np.array(spatial_domain), t_d, C)
    evald_traj = pi.EvalData([t_d, np.array(spatial_domain)], x_l,
                             name="desired system state x_d(z,t)")
    sys_err_ed = pi.EvalData(sys_ed.input_data,
                         sys_ed.output_data - evald_traj.output_data,
                         name="tracking error state x(z,t) - x_d(z,t)")

    # error system
    err_ed = pi.EvalData(sys_ed.input_data,
                         obs_ed.output_data - sys_ed.output_data,
                         name="observer error state \\hat x(z,t) - x(z,t)")

    # simulate coefficients of the target (error) system
    def obs_err_ic(z): return obs_ic(z) - sys_ic(z)
    if 1:
        obs_err_ic_weights = pi.project_on_base(
            [pi.Function(obs_err_ic)], pi.get_base(obs_lbl))
        err_t_ss = pi.create_state_space(pi.parse_weak_formulation(obs_err_rad_pde))
        err_t_ss.A[1] = np.transpose(err_t_ss.A[1])
        print(pi.SecondOrderRobinEigenfunction.eigfreq_eigval_hint(
            param_t, l, n_modal)[1])
        print(np.flipud(np.sort(np.linalg.eig(err_t_ss.A[1])[0])))
        _, err_t_weights = pi.simulate_state_space(
            err_t_ss, obs_err_ic_weights, temporal_domain)
        err_t_ed = get_sim_result(
            obs_lbl, err_t_weights, temporal_domain, spatial_domain,
            0, 0, name="observer error target system \hat x(z,t)")[0]
    else:
        obs_err_ic_weights = pi.project_on_base(
            [pi.Function(obs_err_ic)], eig_funcs_r)
        l = np.matrix([[f.derive(1)(0)] for f in eig_funcs_r_t])
        c = np.matrix([[f(0) for f in eig_funcs_r]])
        lam = np.matrix(np.diag(np.real_if_close(eig_val)))
        A = np.asarray(lam + l @ c)
        err_t_ss = pi.create_state_space(pi.parse_weak_formulation(obs_err_rad_pde))
        err_t_ss.A[1] = A
        # err_t_ss = pi.StateSpace({0: np.zeros(A.shape[0]), 1: A}, {0: {1: np.zeros((A.shape[0],1))}})
        _, err_t_weights = pi.simulate_state_space(
            err_t_ss, obs_err_ic_weights, temporal_domain)
        err_t_ed = get_sim_result(
            obs_lbl, err_t_weights, temporal_domain, spatial_domain,
            0, 0, name="error target system")[0]

    plots = list()
    # pyqtgraph visualization
    plots.append(pi.PgAnimatedPlot(
        [sys_ed, obs_ed, evald_traj], title="animation", replay_gain=.05))
    plots.append(pi.PgAnimatedPlot(
        [err_ed, sys_err_ed], title="animation", replay_gain=.05))
    # matplotlib visualization
    plots.append(pi.MplSlicePlot([evald_traj, sys_ed, obs_ed], spatial_point=0,
                                 legend_label=["$x_d(0,t)$",
                                               "$x(0,t)$",
                                               "$\hat x(0,t)$"]))
    plots.append(pi.MplSlicePlot([evald_traj, sys_ed, obs_ed], spatial_point=1,
                                 legend_label=["$x_d(1,t)$",
                                               "$x(1,t)$",
                                               "$\hat x(1,t)$"]))
    pi.show()

    pi.tear_down((sys_lbl, obs_sys_lbl, ctrl_lbl, ctrl_target_lbl,
                  obs_lbl, obs_target_lbl) + \
                 base_labels + obs_base_labels + tar_obs_base_labels,
                 plots)
