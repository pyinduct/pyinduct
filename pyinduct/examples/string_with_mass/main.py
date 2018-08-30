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
    sys_lbl = "fem_system"
    obs_lbl = "fem_observer"
    n1 = 6
    n2 = 6
    ncf = 8
    build_fem_bases(sys_lbl, n1, n2, obs_lbl, ncf)

    # controller
    controller = build_controller(sys_lbl)
    input_ = pi.SimulationInputSum([closed_loop_traj, controller])

    # system approximation
    sys_wf = build_original_weak_formulation(
        sys_lbl, spatial_domain, input_, sys_lbl)
    obs_wf = build_canonical_weak_formulation(
        obs_lbl, spat_domain_cf, input_, obs_lbl)

    # simulation
    init_cond = {
        sys_wf.name: [SwmBaseFraction(
        [pi.Function.from_constant(0), pi.Function.from_constant(0)], [0, 0])],
        obs_wf.name: [SwmBaseCanonicalFraction(
        [pi.Function.from_constant(0)], [0, 0])]}
    spatial_domains = {sys_wf.name: spatial_domain, obs_wf.name: spat_domain_cf}
    ceq, ss, init_weights, weights, evald = pi.simulate_systems(
        [sys_wf, obs_wf], init_cond, temporal_domain, spatial_domains,
    )

    # check eigenvalues of the approximation
    pprint()
    pprint("Eigenvalues of the system:")
    pprint(sort_eigenvalues(np.linalg.eigvals(ceq[0].dynamic_forms["fem_system"].e_n_pb_inv @
                                              ceq[0].dynamic_forms["fem_system"].matrices["E"][0][1])))
    pprint("Eigenvalues of the observer:")
    pprint(sort_eigenvalues(np.linalg.eigvals(ceq[1].dynamic_forms["fem_observer"].e_n_pb_inv @
                                              ceq[1].dynamic_forms["fem_observer"].matrices["E"][0][1])))


    # visualization
    eval_data1 = pi.get_sim_result(sys_lbl + "_1_visu", weights, temporal_domain, spatial_domain, 0, 0, name="x1(z,t)")
    # eval_data2 = pi.get_sim_result(sys_lbl + "_2_visu", weights, temporal_domain, spatial_domain, 0, 0, name="x2(z,t)")
    eval_data3 = pi.get_sim_result(sys_lbl + "_3_visu", weights, temporal_domain, spatial_domain, 0, 0, name="xi1(t)")
    # eval_data4 = pi.get_sim_result(sys_lbl + "_4_visu", weights, temporal_domain, spatial_domain, 0, 0, name="xi2(t)")
    plots = list()
    plots.append(pi.PgAnimatedPlot(eval_data1 + eval_data3))
    # plots.append(pi.PgAnimatedPlot(eval_data2 + eval_data4))
    pi.show()


if __name__ == "__main__" or test_examples:
    main()
