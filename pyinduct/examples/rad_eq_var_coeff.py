import numpy as np
import scipy.integrate as si
import pyinduct as pi
import pyinduct.parabolic as parabolic


def run():
    # system/simulation parameters
    actuation_type = 'robin'
    bound_cond_type = 'robin'
    l = 1.
    T = 1
    spatial_domain = pi.Domain(bounds=(0, l), num=15)
    temporal_domain = pi.Domain(bounds=(0, T), num=100)
    n = 10

    # original system parameters
    a2 = .5
    a2_z = pi.ConstantFunction(a2)
    a1_z = pi.Function(lambda z: 0.1 * np.exp(4 * z),
                       derivative_handles=[lambda z: 0.4 * np.exp(4 * z)])
    a0_z = lambda z: 1 + 10 * z + 2 * np.sin(4 * np.pi / l * z)
    alpha = -1
    beta = -1
    param = [a2, a1_z, a0_z, alpha, beta]

    # target system parameters (controller parameters)
    a1_t = -0
    a0_t = -6
    alpha_t = 3
    beta_t = 3
    param_t = [a2, a1_t, a0_t, alpha_t, beta_t]
    adjoint_param_t = pi.SecondOrderEigenfunction.get_adjoint_problem(param_t)

    # original intermediate ("_i") and target intermediate ("_ti") system
    # parameters
    _, _, a0_i, alpha_i, beta_i = parabolic.general.eliminate_advection_term(
        param, l)
    param_i = a2, 0, a0_i, alpha_i, beta_i
    _, _, a0_ti, alpha_ti, beta_ti = parabolic.general.eliminate_advection_term(
        param_t, l)
    param_ti = a2, 0, a0_ti, alpha_ti, beta_ti

    # create (not normalized) target (_t) eigenfunctions
    eig_val_t, init_eig_base_t = \
        pi.SecondOrderRobinEigenfunction.cure_interval(
            spatial_domain, param=param_t, n=n)
    _, init_adjoint_eig_base_t = \
        pi.SecondOrderRobinEigenfunction.cure_interval(
            spatial_domain, param=adjoint_param_t, eig_val=eig_val_t)

    # normalize eigenfunctions and adjoint eigenfunctions
    eig_base_t, adjoint_eig_base_t = pi.normalize_base(init_eig_base_t,
                                                       init_adjoint_eig_base_t)

    # transformed original eigenfunctions
    eig_base = pi.Base([pi.TransformedSecondOrderEigenfunction(
        eig_val_t[i],
        [eig_base_t[i](0), alpha * eig_base_t[i](0), 0, 0],
        [a2_z, a1_z, a0_z],
        pi.Domain((0, l), 100))
        for i in range(n)]
    )

    # create testfunctions
    fem_base = pi.LagrangeFirstOrder.cure_interval(spatial_domain)

    # register functions
    pi.register_base("eig_funcs_t", eig_base_t)
    pi.register_base("adjoint_eig_funcs_t", adjoint_eig_base_t)
    pi.register_base("eig_funcs", eig_base)
    pi.register_base("fem_funcs", fem_base)

    # init trajectory
    traj = parabolic.RadFeedForward(l,
                                    T,
                                    param_ti,
                                    bound_cond_type,
                                    actuation_type)

    # original () and target (_t) field variable
    fem_field_variable = pi.FieldVariable("fem_funcs", location=l)
    field_variable_t = pi.FieldVariable("eig_funcs_t",
                                        weight_label="eig_funcs",
                                        location=l)
    d_field_variable_t = field_variable_t.derive(spat_order=1)
    field_variable = pi.FieldVariable("eig_funcs", location=l)
    d_field_variable = field_variable.derive(spat_order=1)

    # intermediate (_i) and target intermediate (_ti) transformations by z=l

    # x_i  = x   * transform_i_at_l
    transform_i_at_l = np.exp(si.quad(lambda z: a1_z(z) / 2 / a2, 0, l)[0])

    # x  = x_i   * inv_transform_i_at_l
    inv_transform_i_at_l = np.exp(-si.quad(lambda z: a1_z(z) / 2 / a2, 0, l)[0])

    # x_ti = x_t * transform_ti_at_l
    transform_ti_at_l = np.exp(a1_t / 2 / a2 * l)

    # intermediate (_i) and target intermediate (_ti) field variable (list of
    #  scalar terms = sum of scalar terms)
    x_fem_i_at_l = [pi.ScalarTerm(fem_field_variable, transform_i_at_l)]
    x_i_at_l = [pi.ScalarTerm(field_variable, transform_i_at_l)]
    xd_i_at_l = [pi.ScalarTerm(d_field_variable, transform_i_at_l),
                 pi.ScalarTerm(field_variable,
                               transform_i_at_l * a1_z(l) / 2 / a2)]
    x_ti_at_l = [pi.ScalarTerm(field_variable_t, transform_ti_at_l)]
    xd_ti_at_l = [pi.ScalarTerm(d_field_variable_t, transform_ti_at_l),
                  pi.ScalarTerm(field_variable_t,
                                transform_ti_at_l * a1_t / 2 / a2)]

    # discontinuous operator (Kx)(t) = int_kernel_zz(l)*x(l,t)
    int_kernel_ll = (alpha_ti
                     - alpha_i
                     + si.quad(lambda z: (a0_i(z) - a0_ti) / 2 / a2, 0, l)[0])

    # init controller
    controller = parabolic.control.get_parabolic_robin_backstepping_controller(
        state=x_fem_i_at_l, approx_state=x_i_at_l,
        d_approx_state=xd_i_at_l,
        approx_target_state=x_ti_at_l,
        d_approx_target_state=xd_ti_at_l,
        integral_kernel_ll=int_kernel_ll,
        original_beta=beta_i, target_beta=beta_ti,
        scale=inv_transform_i_at_l)

    traj.scale(inv_transform_i_at_l)
    input = pi.SimulationInputSum([traj, controller])

    rad_pde, base_labels = parabolic.general.get_parabolic_robin_weak_form(
        "fem_funcs", "fem_funcs", input, param, spatial_domain.bounds)
    ce = pi.parse_weak_formulation(rad_pde)
    ss_weak = pi.create_state_space(ce)

    # simulate
    t, q = pi.simulate_state_space(ss_weak,
                                   np.zeros((len(fem_base))),
                                   temporal_domain)

    # visualization
    plots = list()
    evald_x = pi.evaluate_approximation("fem_funcs", q, t, spatial_domain,
                                        name="x(z,t)")

    # pyqtgraph visualization
    plots.append(pi.evaluate_approximation("fem_funcs", q, t, spatial_domain,
                                           name="x(z,t)"))
    plots.append(pi.PgAnimatedPlot([evald_x],
                                   title="animation",
                                   replay_gain=.25))
    plots.append(pi.PgSurfacePlot(evald_x, title=evald_x.name))

    # matplotlib visualization
    plots.append(pi.MplSlicePlot([evald_x], time_point=1,
                                 legend_label=["$x(z,1)$"], legend_location=1))
    plots.append(pi.MplSurfacePlot(evald_x))
    pi.show()

    pi.tear_down(("eig_funcs_t",
                  "adjoint_eig_funcs_t",
                  "eig_funcs",
                  "fem_funcs") + base_labels,
                 plots)


if __name__ == "__main__":
    run()
