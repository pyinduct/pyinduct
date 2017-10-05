from pyinduct.tests import test_examples

if __name__ == "__main__" or test_examples:
    import pyinduct as pi
    import pyinduct.parabolic as parabolic
    import numpy as np

    # PARAMETERS TO VARY
    # number of eigenfunctions, used for control law approximation
    n_modal = 10
    # number FEM test functions, used for system approximation/simulation
    n_fem = 30
    # control law parameter, stabilizing: param_a0_t < 0, destabilizing: param_a0_t > 0
    param_a0_t = -6
    # initial profile x(z,0) (desired x(z,0)=0)
    init_profile = 0.2

    # system/simulation parameters
    actuation_type = 'robin'
    bound_cond_type = 'robin'
    l = 1
    T = 1
    spatial_domain = pi.Domain(bounds=(0, l), num=n_fem)
    temporal_domain = pi.Domain(bounds=(0, T), num=1e2)
    n = n_modal
    show_plots = False

    # original system parameters
    a2 = .5
    a1 = 1
    a0 = 2
    alpha = -0.5
    beta = -1
    param = [a2, a1, a0, alpha, beta]
    adjoint_param = pi.SecondOrderEigenfunction.get_adjoint_problem(param)

    # target system parameters (controller parameters)
    a1_t = -1
    a0_t = param_a0_t
    alpha_t = 3
    beta_t = 2
    param_t = [a2, a1_t, a0_t, alpha_t, beta_t]

    # actuation_type by b which is close to b_desired on a k times subdivided spatial domain
    b_desired = 0.4
    k = 5  # = k1 + k2
    k1, k2, b = parabolic.control.split_domain(k, b_desired, l, mode='coprime')[0:3]
    M = np.linalg.inv(parabolic.general.get_in_domain_transformation_matrix(k1, k2, mode="2n"))

    # original intermediate ("_i") and target intermediate ("_ti") system parameters
    _, _, a0_i, alpha_i, beta_i = parabolic.eliminate_advection_term(param, l)
    param_i = a2, 0, a0_i, alpha_i, beta_i
    _, _, a0_ti, alpha_ti, beta_ti = parabolic.eliminate_advection_term(param_t, l)
    param_ti = a2, 0, a0_ti, alpha_ti, beta_ti

    # COMPUTE DESIRED FIELDVARIABLE
    # THE NAMING OF THE POWER SERIES COEFFICIENTS IS BASED ON THE PUBLICATION:
    #      - WANG; WOITTENNEK:BACKSTEPPING-METHODE FUER PARABOLISCHE SYSTEM MIT PUNKTFOERMIGEM INNEREN EINGRIFF
    # compute input u_i of the boundary-controlled intermediate (_i) system with n_y/2 temporal derivatives
    n_y = 40
    y, t_x = pi.gevrey_tanh(T, n_y, 1.1, 2)
    B = pi.coefficient_recursion(y, alpha_i * y, param_i)
    x_i_at_l = pi.temporal_derived_power_series(l, B, int(n_y / 2) - 1, n_y, spatial_der_order=0)
    dx_i_at_l = pi.temporal_derived_power_series(l, B, int(n_y / 2) - 1, n_y, spatial_der_order=1)
    u_i = dx_i_at_l + beta_i * x_i_at_l

    # compute coefficients C, D and E for the power series
    E = pi.coefficient_recursion(y, beta_i * y, param_i)
    q = pi.temporal_derived_power_series(l - b, E, int(n_y / 2) - 1, n_y)
    C = pi.coefficient_recursion(q, alpha_i * q, param_i)
    D = pi.coefficient_recursion(np.zeros(u_i.shape), u_i, param_i)

    # compute power series for the desired in-domain intermediate (_id) fieldvariable (subdivided in x1_i & x2_i)
    z_x1 = np.linspace(0, b, len(spatial_domain) * k1 / k)
    x1_id_desired = pi.power_series(z_x1, t_x, C)
    z_x2 = np.linspace(b, l, len(spatial_domain) * k2 / k)[1:]
    x2_id_desired = pi.power_series(z_x2, t_x, C) - pi.power_series(z_x2 - b, t_x, D)
    z_x = np.array(list(z_x1) + list(z_x2))
    x_id = np.concatenate((x1_id_desired, x2_id_desired), axis=1)

    # get the original system field variable: x = e^(-a1/2/a2*z)*x_id
    x_desired = np.nan * np.zeros(x_id.shape)
    for i in range(x_id.shape[0]):
        x_desired[i, :] = x_id[i, :] * np.exp(-a1 / 2 / a2 * z_x)

    evald_xd = pi.EvalData([t_x, z_x], x_desired, name="x(z,t) power series")

    # compute desired intermediate (_i) fieldvariable
    C_i = pi.coefficient_recursion(y, alpha_i * y, param_i)
    xi_desired = pi.power_series(z_x, t_x, C_i)
    evald_xi_desired = pi.EvalData([t_x, z_x], xi_desired, name="x_i(z,t) power series")

    # THE TOOLBOX OFFERS TWO WAYS TO GENERATE A TRAJECTORY FOR THE TARGET SYSTEM
    if False:
        # First way: simply instantiate pi.RadTrajectory
        traj = parabolic.RadFeedForward(l, T, param_ti, bound_cond_type, actuation_type, show_plot=show_plots)
    else:
        # Second (and more general) way:
        #   - calculate the power series coefficients with pi.coefficient_recursion
        #   - calculate the power series with pi.power_series
        #   - instantiate pi.InterpTrajectory with the calculated data
        C_ti = pi.coefficient_recursion(y, alpha_ti * y, param_ti)
        x_ti_desired = pi.power_series(np.array([l]), t_x, C_ti)
        dx_ti_desired = pi.power_series(np.array([l]), t_x, C_ti, spatial_der_order=1)
        v_i = dx_ti_desired + beta_ti * x_ti_desired
        traj = pi.InterpolationTrajectory(t_x, v_i, show_plot=show_plots)

    # scale trajectory that x(0,T)=1 instead of x_i(0,T)=1
    # traj.scale /= x1_id_desired[-1, 0]

    # create (not normalized) eigenfunctions
    eig_val, init_eig_base = pi.SecondOrderRobinEigenfunction.solve_evp_hint(param, l, n=n)
    _, init_adjoint_eig_base = pi.SecondOrderRobinEigenfunction.solve_evp_hint(adjoint_param, l, eig_val=eig_val)

    # normalize eigenfunctions and adjoint eigenfunctions
    eig_base, adjoint_eig_base = pi.normalize_base(init_eig_base, init_adjoint_eig_base)

    # eigenfunctions of the in-domain intermediate (_id) and the intermediate (_i) system
    scale_id = [f(0) for f in eig_base]
    eig_val_i, eig_base_id = pi.SecondOrderRobinEigenfunction.solve_evp_hint(param_i, l, eig_val=eig_val, scale=scale_id)
    scale_i = [e_f(0) * e_f_id(l) / e_f_id(b) for e_f, e_f_id in zip(eig_base, eig_base_id)]
    _, eig_base_i = pi.SecondOrderRobinEigenfunction.solve_evp_hint(param_i, l, eig_val=eig_val, scale=scale_i)

    # eigenfunctions from target intermediate system ("_ti")
    scale_ti = [f(0) for f in eig_base_i]
    _, eig_base_ti = pi.SecondOrderRobinEigenfunction.solve_evp_hint(param_ti, l, eig_val=eig_val, scale=scale_ti)

    # create test-functions
    nodes, fem_base = pi.cure_interval(pi.LagrangeFirstOrder,
                                       spatial_domain.bounds,
                                       node_count=len(spatial_domain))

    # register functions
    pi.register_base("adjoint_eig_funcs", adjoint_eig_base)
    pi.register_base("eig_funcs", eig_base)
    pi.register_base("eig_funcs_i", eig_base_i)
    pi.register_base("eig_funcs_ti", eig_base_ti)
    pi.register_base("fem_funcs", fem_base)

    # original intermediate (_i), target intermediate (_ti) and fem field variable
    fem_field_variable = pi.FieldVariable("fem_funcs", location=l)
    field_variable_i = pi.FieldVariable("eig_funcs_i", weight_label="eig_funcs", location=l)
    d_field_variable_i = field_variable_i.derive(spat_order=1)
    field_variable_ti = pi.FieldVariable("eig_funcs_ti", weight_label="eig_funcs", location=l)
    d_field_variable_ti = field_variable_ti.derive(spat_order=1)

    # intermediate (_i) and target intermediate (_ti) field variable (list of scalar terms = sum of scalar terms)
    x_i_at_l = [pi.ScalarTerm(field_variable_i)]
    xd_i_at_l = [pi.ScalarTerm(d_field_variable_i)]
    x_ti_at_l = [pi.ScalarTerm(field_variable_ti)]
    xd_ti_at_l = [pi.ScalarTerm(d_field_variable_ti)]

    # shift transformation
    shifted_fem_funcs_i = pi.Base(np.array(
        [pi.FiniteTransformFunction(func, M, l, scale_func=lambda z: np.exp(a1 / 2 / a2 * z))
         for func in fem_base]))
    shifted_eig_funcs_id = pi.Base(np.array([pi.FiniteTransformFunction(func, M, l)
                                             for func in eig_base_id]))
    pi.register_base("sh_fem_funcs_i", shifted_fem_funcs_i, overwrite=True)
    pi.register_base("sh_eig_funcs_id", shifted_eig_funcs_id, overwrite=True)
    sh_fem_field_variable_i = pi.FieldVariable("sh_fem_funcs_i", weight_label="fem_funcs", location=l)
    sh_field_variable_id = pi.FieldVariable("sh_eig_funcs_id", weight_label="eig_funcs", location=l)
    sh_x_fem_i_at_l = [pi.ScalarTerm(sh_fem_field_variable_i),
                       pi.ScalarTerm(field_variable_i),
                       pi.ScalarTerm(sh_field_variable_id, -1)]


    # controller initialization
    def int_kernel_zz(z):
        return alpha_ti - alpha_i + (a0_i - a0_ti) / 2 / a2 * z


    scale = np.exp(-a1 / 2 / a2 * b)
    controller = parabolic.control.get_parabolic_robin_backstepping_controller(state=sh_x_fem_i_at_l, approx_state=x_i_at_l,
                                                                               d_approx_state=xd_i_at_l,
                                                                               approx_target_state=x_ti_at_l,
                                                                               d_approx_target_state=xd_ti_at_l,
                                                                               integral_kernel_ll=int_kernel_zz(l),
                                                                               original_beta=beta_i, target_beta=beta_ti,
                                                                               scale=scale)
    traj.scale(scale)
    input = pi.SimulationInputSum([traj, controller])

    # determine (A,B) with modal transformation
    rad_pde, base_labels = parabolic.general.get_parabolic_robin_weak_form(
        "fem_funcs", "fem_funcs", input, param, spatial_domain.bounds, b)
    ce = pi.parse_weak_formulation(rad_pde)
    ss_weak = pi.create_state_space(ce)

    # simulate (t: time vector, q: weights matrix)
    t, q = pi.simulate_state_space(ss_weak, init_profile * np.ones((len(fem_base))), temporal_domain)

    # compute modal weights (for the intermediate system: evald_modal_xi)
    mat = pi.calculate_base_transformation_matrix(fem_base, eig_base)
    q_i = np.zeros((q.shape[0], len(eig_base_i)))
    for i in range(q.shape[0]):
        q_i[i, :] = np.dot(q[i, :], np.transpose(mat))

    # evaluate approximation of xi
    evald_modal_xi = pi.evaluate_approximation("eig_funcs_i", q_i, t, spatial_domain, name="x_i(z,t) modal simulation")
    evald_modal_T0_xid = pi.evaluate_approximation("sh_eig_funcs_id", q_i, t, spatial_domain,
                                                    name="T0*x_i(z,t) modal simulation")
    evald_shifted_x = pi.evaluate_approximation("sh_fem_funcs_i", q, t, spatial_domain,
                                                 name="T0*e^(-a1/a2/2*z)*x_(z,t) fem simulation")
    evald_appr_xi = pi.EvalData(evald_modal_xi.input_data,
                                 evald_shifted_x.output_data + evald_modal_xi.output_data - evald_modal_T0_xid.output_data,
                                 name="x_i(t) approximated")

    # evaluate approximation of x
    evald_fem_x = pi.evaluate_approximation("fem_funcs", q, t, spatial_domain, name="x(z,t) simulation")

    plots = list()
    # pyqtgraph visualisations
    plots.append(pi.PgAnimatedPlot(
        [evald_fem_x, evald_modal_xi, evald_appr_xi, evald_xd, evald_xi_desired]))
    plots.append(pi.PgSurfacePlot(evald_xd, title=evald_xd.name))
    plots.append(pi.PgSurfacePlot(evald_fem_x, title=evald_fem_x.name))
    # matplotlib visualization
    plots.append(pi.MplSurfacePlot(evald_fem_x))
    plots.append(pi.MplSlicePlot([evald_xd, evald_fem_x], spatial_point=0,
                                 legend_label=[evald_xd.name, evald_fem_x.name]))
    pi.show()

    pi.tear_down(("adjoint_eig_funcs",
                  "eig_funcs",
                  "eig_funcs_i",
                  "eig_funcs_ti",
                  "fem_funcs") + base_labels,
                 plots)
