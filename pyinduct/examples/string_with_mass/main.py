"""
Main script file for the simulation of the string with mass example.
"""
from pyinduct.tests import test_examples
from pyinduct.examples.string_with_mass.control import *
from pyinduct.hyperbolic.feedforward import FlatString
import pyinduct as pi
import pickle
import time


def main():

    # control mode
    control_mode = ["open_loop",
                    "closed_loop",
                    "modal_observer",
                    "fem_observer"][2]

    # constant observer initial error
    ie = 0.2

    # domains
    z_end = 1
    spatial_discretization = 100
    spatial_domain = pi.Domain((0, z_end), spatial_discretization)
    spat_domain_cf = pi.Domain((-z_end, z_end), spatial_discretization)
    t_end = 30
    temporal_discretization = int(30 * t_end)
    temporal_domain = pi.Domain((0, t_end), temporal_discretization)

    # planning input trajectories
    smooth_transition1 = pi.SmoothTransition(
        (0, 1), (2, 4), method="poly", differential_order=2)
    smooth_transition2 = pi.SmoothTransition(
        (0, -1.5), (23, 25), method="poly", differential_order=2)
    not_too_smooth_transition = pi.SmoothTransition(
        (0, -.5), (14, 14.2), method="poly", differential_order=2)
    closed_loop_traj1 = SecondOrderFeedForward(smooth_transition1)
    closed_loop_traj2 = SecondOrderFeedForward(smooth_transition2)
    disturbance = SecondOrderFeedForward(not_too_smooth_transition)
    open_loop_traj = FlatString(
        y0=0, y1=1, z0=spatial_domain.bounds[0], z1=spatial_domain.bounds[1],
        t0=1, dt=3, params=param)

    # set up bases
    sys_fem_lbl = "fem_system"
    sys_modal_lbl = "modal_system"
    obs_fem_lbl = "fem_observer"
    obs_modal_lbl = "modal_observer"
    n1 = 11
    n2 = 11
    n_obs_fem = 11
    n_obs_modal = 16
    build_fem_bases(sys_fem_lbl, n1, n2, obs_fem_lbl, n_obs_fem, sys_modal_lbl)
    build_modal_bases(sys_modal_lbl, n_obs_modal, obs_modal_lbl, n_obs_modal)

    # controller
    controller = build_controller(sys_fem_lbl, sys_modal_lbl)
    if control_mode == "open_loop":
        input_ = pi.SimulationInputSum([open_loop_traj])
    else:
        input_ = pi.SimulationInputSum(
            [closed_loop_traj1, controller, disturbance, closed_loop_traj2])

    # observer error
    obs_fem_error, obs_modal_error = init_observer_gain(
        sys_fem_lbl, sys_modal_lbl, obs_fem_lbl, obs_modal_lbl)

    # input / observer error vector
    input_vector = pi.SimulationInputVector([input_, obs_fem_error, obs_modal_error])
    control = pi.Input(input_vector, index=0)
    yt_fem = pi.Input(input_vector, index=1)
    yt_modal = pi.Input(input_vector, index=2)

    # system approximation
    sys_wf = build_original_weak_formulation(
        sys_fem_lbl, spatial_domain, control, sys_fem_lbl)
    obs_fem_wf = build_canonical_weak_formulation(
        obs_fem_lbl, spat_domain_cf, control, yt_fem, obs_fem_lbl)
    obs_modal_wf = build_canonical_weak_formulation(
        obs_modal_lbl, spat_domain_cf, control, yt_modal, obs_modal_lbl)

    # set control mode
    apply_control_mode(sys_fem_lbl, sys_modal_lbl, obs_fem_lbl, obs_modal_lbl,
                       control_mode)

    # define initial conditions
    init_cond = {
        sys_wf.name: [SwmBaseFraction(
            [pi.ConstantFunction(0, domain=spatial_domain.bounds),
             pi.ConstantFunction(0, domain=spatial_domain.bounds)],
            [0, 0])],
        obs_fem_wf.name: [SwmBaseCanonicalFraction(
            [pi.Function(lambda th: ie * (2 - th), (-1, 1))], [0, ie * 4])],
        obs_modal_wf.name: [SwmBaseCanonicalFraction(
            [pi.Function(lambda th: ie * (2 - th), (-1, 1))], [0, ie * 4])]
    }

    # simulation
    spatial_domains = {sys_wf.name: spatial_domain,
                       obs_fem_wf.name: spat_domain_cf,
                       obs_modal_wf.name: spat_domain_cf}
    intermediate_results = list()
    _ = pi.simulate_systems(
        [sys_wf, obs_fem_wf, obs_modal_wf],
        init_cond, temporal_domain, spatial_domains, out=intermediate_results
    )
    ceq, ss, init_weights, weights = intermediate_results

    # print some stuff for debugging
    check_eigenvalues(sys_fem_lbl, obs_fem_lbl, obs_modal_lbl, ceq, ss)

    # visualization data
    split_indizes = [n1 + n2 ,
                     n1 + n2 + n_obs_fem,
                     n1 + n2 + n_obs_fem + n_obs_modal]
    #  system
    weights_sys = weights[:, :split_indizes[0]]
    eval_data1 = pi.get_sim_result(sys_fem_lbl + "_1_visu", weights_sys,
                                   temporal_domain, spatial_domain, 0, 0,
                                   name="x1(z,t)")[0]

    #  fem observer
    weights_fem_obs = weights[:, split_indizes[0]: split_indizes[1]]
    fem_obs_ed = pi.get_sim_result(sys_fem_lbl + "_1_trafo_visu", weights_fem_obs,
                                     temporal_domain, spatial_domain, 0, 0,
                                     name="\hat x1_fem(z,t)")[0]
    #  modal observer
    weights_modal_obs = weights[:, split_indizes[1]: split_indizes[2]]
    modal_obs_ed = pi.get_sim_result(sys_modal_lbl + "_1_trafo_visu", weights_modal_obs,
                                     temporal_domain, spatial_domain, 0, 0,
                                     name="\hat x1_modal(z,t)")[0]

    # create plots
    plots = list()
    plots.append(pi.PgSurfacePlot([eval_data1, modal_obs_ed]))
    plots.append(SwmPgAnimatedPlot([eval_data1, modal_obs_ed]))
    pi.show()

    # save results
    if 0:
        timestamp = time.strftime("%Y-%m-%d__%H-%M-%S__")
        path = "results/"
        conf = "{}__({}-{}-{})__".format(
            control_mode, n1 + n2, n_obs_fem, n_obs_modal)
        description = input("result description:").replace(" ", "-")
        file = open(path + timestamp + conf + description + ".pkl", "wb")
        pickle.dump([eval_data1, fem_obs_ed, modal_obs_ed], file)
        file.close()


if __name__ == "__main__" or test_examples:
    main()
