"""
Main script file for the simulation of the string with mass example.
"""
from pyinduct.tests import test_examples
from pyinduct.examples.string_with_mass.control import *
from pyinduct.hyperbolic.feedforward import FlatString
import pyinduct as pi


def main():

    # domains
    z_end = 1
    spatial_discretization = 100
    spatial_domain = pi.Domain((0, z_end), spatial_discretization)
    spat_domain_cf = pi.Domain((-z_end, z_end), spatial_discretization)
    t_end = 10
    temporal_discretization = 300
    temporal_domain = pi.Domain((0, t_end), temporal_discretization)

    # planning input trajectories
    smooth_transition = pi.SmoothTransition(
        (0, 1), (2, 4), method="poly", differential_order=2)
    closed_loop_traj = SecondOrderFeedForward(smooth_transition)
    open_loop_traj = FlatString(
        y0=0, y1=1, z0=spatial_domain.bounds[0], z1=spatial_domain.bounds[1],
        t0=1, dt=3, params=param)

    # set up bases
    sys_fem_lbl = "fem_system"
    sys_modal_lbl = "modal_system"
    obs_fem_lbl = "fem_observer"
    obs_modal_lbl = "modal_observer"
    n1 = 6
    n2 = 6
    n_obs_fem = 7
    n_obs_modal = 10
    build_fem_bases(sys_fem_lbl, n1, n2, obs_fem_lbl, n_obs_fem, sys_modal_lbl)
    build_modal_bases(sys_modal_lbl, n_obs_modal, obs_modal_lbl, n_obs_modal)

    # controller
    controller = build_controller(sys_fem_lbl)
    input_ = pi.SimulationInputSum([closed_loop_traj, controller])

    # system approximation
    sys_wf = build_original_weak_formulation(
        sys_fem_lbl, spatial_domain, input_, sys_fem_lbl)
    obs_fem_wf = build_canonical_weak_formulation(
        obs_fem_lbl, spat_domain_cf, input_, obs_fem_lbl)
    obs_modal_wf = build_canonical_weak_formulation(
        obs_modal_lbl, spat_domain_cf, input_, obs_modal_lbl)

    # simulation
    init_cond = {
        sys_wf.name: [SwmBaseFraction(
            [pi.Function.from_constant(0), pi.Function.from_constant(0)],
            [0, 0])],
        obs_fem_wf.name: [SwmBaseCanonicalFraction(
            [pi.Function.from_constant(0)], [0, 0])],
        obs_modal_wf.name: [SwmBaseCanonicalFraction(
            [pi.Function.from_constant(0)], [0, 0])]
    }
    spatial_domains = {sys_wf.name: spatial_domain,
                       obs_fem_wf.name: spat_domain_cf,
                       obs_modal_wf.name: spat_domain_cf}
    intermediate_results = list()
    _ = pi.simulate_systems(
        [sys_wf, obs_fem_wf, obs_modal_wf],
        init_cond, temporal_domain, spatial_domains, out=intermediate_results
    )
    ceq, ss, init_weights, weights = intermediate_results

    # check eigenvalues of the approximation
    A_sys = (-ceq[0].dynamic_forms[sys_fem_lbl].e_n_pb_inv @
             ceq[0].dynamic_forms[sys_fem_lbl].matrices["E"][0][1])
    A_obs = (-ceq[1].dynamic_forms[obs_fem_lbl].e_n_pb_inv @
             ceq[1].dynamic_forms[obs_fem_lbl].matrices["E"][0][1])
    A_modal_obs = (-ceq[2].dynamic_forms[obs_modal_lbl].e_n_pb_inv @
             ceq[2].dynamic_forms[obs_modal_lbl].matrices["E"][0][1])
    pprint()
    pprint("Eigenvalues [{}, {}, {}]".format(sys_fem_lbl, obs_fem_lbl, obs_modal_lbl))
    pprint([np.linalg.eigvals(A_) for A_ in (A_sys, A_obs, A_modal_obs)])

    # visualization data
    split_indizes = [n1 + n2 + 1,
                     n1 + n2 + 1 + n_obs_fem + 2,
                     n1 + n2 + 1 + n_obs_fem + 2 + n_obs_modal]
    ## system
    weights_sys = weights[:, :split_indizes[0]]
    eval_data1 = pi.get_sim_result(sys_fem_lbl + "_1_visu", weights_sys, temporal_domain, spatial_domain, 0, 0, name="x1(z,t)")[0]
    ## fem observer
    weights_fem_obs = weights[:, split_indizes[0]: split_indizes[1]]
    eta1_data = pi.get_sim_result(obs_fem_lbl + "_1_visu", weights_fem_obs, temporal_domain, pi.Domain((0, 1), num=spatial_discretization), 0, 0)[0]
    dz_et3_m1_0 = pi.get_sim_result(obs_fem_lbl + "_3_visu", weights_fem_obs, temporal_domain, pi.Domain((-1, 0), num=spatial_discretization), 0, 1)[1]
    dz_et3_0_p1 = pi.get_sim_result(obs_fem_lbl + "_3_visu", weights_fem_obs, temporal_domain, pi.Domain((0, 1), num=spatial_discretization), 0, 1)[1]
    fem_obs_ed = pi.EvalData(eta1_data.input_data, -param.m / 2 * (
        dz_et3_m1_0.output_data + np.fliplr(dz_et3_0_p1.output_data) + eta1_data.output_data),
        name="\hat x1_fem(z,t)"
    )
    ## modal observer
    weights_modal_obs = weights[:, split_indizes[1]: split_indizes[2]]
    eta1_data_m = pi.get_sim_result(obs_modal_lbl + "_1_visu", weights_modal_obs, temporal_domain, pi.Domain((0, 1), num=spatial_discretization), 0, 0)[0]
    dz_et3_m1_0_m = pi.get_sim_result(obs_modal_lbl + "_3_visu", weights_modal_obs, temporal_domain, pi.Domain((-1, 0), num=spatial_discretization), 0, 1)[1]
    dz_et3_0_p1_m = pi.get_sim_result(obs_modal_lbl + "_3_visu", weights_modal_obs, temporal_domain, pi.Domain((0, 1), num=spatial_discretization), 0, 1)[1]
    modal_obs_ed = pi.EvalData(eta1_data.input_data, -param.m / 2 * (
        dz_et3_m1_0_m.output_data + np.fliplr(dz_et3_0_p1_m.output_data) + eta1_data_m.output_data),
        name="\hat x1_modal(z,t)"
    )

    plots = list()
    plots.append(pi.PgAnimatedPlot([eval_data1, fem_obs_ed, modal_obs_ed]))
    pi.show()


if __name__ == "__main__" or test_examples:
    main()
