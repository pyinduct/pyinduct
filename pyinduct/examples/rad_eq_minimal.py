from pyinduct.tests import test_examples

if __name__ == "__main__" or test_examples:
    import pyinduct as pi
    import pyinduct.parabolic as parabolic
    import numpy as np

    # PARAMETERS TO VARY
    # number of eigenfunctions, used for control law approximation
    n_modal = 10
    # control law parameter, stabilizing: param_a0_t < 0, destabilizing:
    # param_a0_t > 0
    param_a0 = 7
    # initial profile x(z,0) (desired x(z,0)=0)
    init_profile = 1

    # original system parameters
    a2 = 1
    a1 = 3
    a0 = param_a0
    param = [a2, a1, a0, None, None]

    # target system parameters (controller parameters)
    a1_t = -3
    a0_t = -6
    param_t = [a2, a1_t, a0_t, None, None]

    # system/simulation parameters
    actuation_type = 'dirichlet'
    bound_cond_type = 'dirichlet'
    l = 1
    T = 1
    spatial_domain = pi.Domain(bounds=(0, l), num=30)
    temporal_domain = pi.Domain(bounds=(0, T), num=100)
    n = n_modal

    # eigenvalues /-functions original system
    scale = np.ones(n) * np.sqrt(2)
    eig_values, eig_funcs = pi.SecondOrderDirichletEigenfunction.cure_interval(
        spatial_domain, param=param, n=n, scale=scale)
    pi.register_base("eig_funcs", eig_funcs)

    # eigenfunctions target system
    eig_freq = pi.SecondOrderDirichletEigenfunction.eigval_tf_eigfreq(
        param, eig_val=eig_values)
    eig_freq_t = pi.SecondOrderDirichletEigenfunction.eigval_tf_eigfreq(
        param_t, eig_val=eig_values)
    scale_t = scale * eig_freq / eig_freq_t
    _, eig_funcs_t = pi.SecondOrderDirichletEigenfunction.cure_interval(
        spatial_domain, param=param_t, eig_freq=eig_freq_t, scale=scale_t)
    pi.register_base("eig_funcs_t", eig_funcs_t)

    # init controller
    x_at_1 = pi.FieldVariable("eig_funcs", location=1)
    xt_at_1 = pi.FieldVariable("eig_funcs_t",
                               weight_label="eig_funcs",
                               location=1)
    controller = pi.StateFeedback(pi.WeakFormulation([pi.ScalarTerm(x_at_1, 1),
                                                      pi.ScalarTerm(xt_at_1, -1)],
                                                     name="backstepping_controller"))

    # derive initial field variable x(z,0) and weights
    start_state = pi.Function(lambda z: init_profile, domain=spatial_domain.bounds)
    initial_weights = pi.project_on_base(start_state, eig_funcs)

    # init trajectory
    traj = parabolic.RadFeedForward(l,
                                    T,
                                    param_t,
                                    bound_cond_type,
                                    actuation_type)

    # input with feedback
    control_law = pi.SimulationInputSum([traj, controller])

    # determine (A,B) with modal-transformation
    A = np.diag(np.real_if_close(eig_values))
    B = -a2 * np.array([eig_funcs[i].derive()(l) for i in range(n)])
    B = np.reshape(B, (B.size, 1))
    ss = pi.StateSpace(A, B, base_lbl="eig_funcs", input_handles=control_law)

    # evaluate desired output data
    z_d = np.linspace(0, l, len(spatial_domain))
    y_d, t_d = pi.gevrey_tanh(T, 40)
    C = pi.coefficient_recursion(np.zeros(y_d.shape), y_d, param)
    x_l = pi.power_series(z_d, t_d, C)
    evald_traj = pi.EvalData([t_d, z_d], x_l, name="x(z,t) desired")

    # simulate
    t, q = pi.simulate_state_space(ss, initial_weights, temporal_domain)

    # visualization
    plots = list()
    evald_x = pi.evaluate_approximation(
        "eig_funcs",
        q,
        t, spatial_domain,
        name="x(z,t) with x(z,0)=" + str(init_profile))

    # pyqtgraph visualization
    plots.append(pi.PgAnimatedPlot([evald_x, evald_traj], title="animation"))
    plots.append(pi.PgSurfacePlot(evald_x, title=evald_x.name))
    plots.append(pi.PgSurfacePlot(evald_traj, title=evald_traj.name))
    # matplotlib visualization
    plots.append(pi.MplSlicePlot(
        [evald_x, evald_traj], time_point=1,
        legend_label=["$x(z,1)$", "$x_d(z,1)$"], legend_location=2))
    pi.show()

    pi.tear_down(("eig_funcs", "eig_funcs_t"), plots)
