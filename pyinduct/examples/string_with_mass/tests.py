from pyinduct.examples.string_with_mass.control import *
from pyinduct.registry import clear_registry
from pyinduct.tests import show_plots
import pickle
import unittest



class StringWithMassTest(unittest.TestCase):
    """
    Its not a real test case class.
    Just a few code snippets for tests during development of this example.
    From the most python ide's each code snippet/test method
    can easily executed via a keyboard shortcut .
    """
    def setUp(self) -> None:
        file_path = os.path.dirname(os.path.abspath(__file__))
        res_dir = os.path.join(file_path, "results")
        if not os.path.isdir(res_dir):
            os.mkdir(res_dir)
        self.res_file = os.path.join(res_dir, "swm_results.pkl")

    def test_find_eigenvalues(self):
        eig_om, eig_vals = find_eigenvalues(4)
        f = plot_eigenvalues(eig_vals, return_figure=True)
        pprint(eig_vals)
        if show_plots:
            f.show()
            pi.show(show_pg=False)

    @unittest.skip("Test is broken")
    def test_observer_evp_scripts(self):
        from pyinduct.examples.string_with_mass.observer_evp_scripts.modal_approximation \
            import build_bases_for_modal_observer_approximation, validate_modal_bases
        pb, db, pbn, dbn, eig_vals = build_bases_for_modal_observer_approximation(10)
        a, b, c, l, l_ub = validate_modal_bases(pb, db, pbn, dbn, eig_vals)
        pprint(l)

    def test_modal_base(self):
        from pyinduct.examples.string_with_mass.observer_evp_scripts.modal_approximation \
            import build_bases_for_modal_observer_approximation
        from pyinduct.examples.string_with_mass.utils import sym

        primal_base, primal_base_nf, dual_base, dual_base_nf, eig_vals = (
            build_bases_for_modal_observer_approximation(6))

        register_evp_base("primal_base", primal_base, sym.z, (0, 1))
        register_evp_base("dual_base", dual_base, sym.z, (0, 1))
        register_evp_base("primal_base_can", primal_base_nf, sym.theta, (-1, 1))
        register_evp_base("dual_base_can", dual_base_nf, sym.theta, (-1, 1))

        pprint(pi.calculate_scalar_product_matrix(
            pi.get_base("dual_base_can"),
            pi.get_base("primal_base_can"),
            SwmBaseCanonicalFraction.scalar_product))

        SwmBaseFraction.l2_scalar_product = False
        pprint(pi.calculate_scalar_product_matrix(
            pi.get_base("primal_base"),
            pi.get_base("dual_base"),
            SwmBaseFraction.scalar_product))
        SwmBaseFraction.l2_scalar_product = True

        plots = list()
        plots.append(pi.visualize_functions([
            frac.members["funcs"][0]
            for frac in pi.get_base("primal_base")],
            return_window=True))
        plots.append(pi.visualize_functions([
            frac.members["funcs"][1]
            for frac in pi.get_base("primal_base")],
            return_window=True))
        if show_plots:
            [p.show() for p in plots]
            pi.show()

    def test_modal_cf_wf(self):
        # domains
        z_end = 1
        spatial_discretization = 100
        spatial_domain = pi.Domain((0, z_end), spatial_discretization)
        spat_domain_cf = pi.Domain((-z_end, z_end), spatial_discretization)

        # set up bases
        sys_fem_lbl = "fem_system"
        sys_modal_lbl = "modal_system"
        obs_fem_lbl = "fem_observer"
        obs_modal_lbl = "modal_observer"
        n1 = 11
        n2 = 11
        n_obs_fem = 23
        n_obs_modal = 12
        build_fem_bases(sys_fem_lbl, n1, n2, obs_fem_lbl, n_obs_fem,
                        sys_modal_lbl)
        build_modal_bases(sys_modal_lbl, n_obs_modal, obs_modal_lbl,
                          n_obs_modal)

        # controller
        input_ = pi.SimulationInputSum([pi.ConstantTrajectory(0)])

        # observer error
        obs_fem_error, obs_modal_error = init_observer_gain(
            sys_fem_lbl, sys_modal_lbl, obs_fem_lbl, obs_modal_lbl)

        # input / observer error vector
        input_vector = pi.SimulationInputVector(
            [input_, obs_fem_error, obs_modal_error])
        control = pi.Input(input_vector, index=0)
        yt_fem = pi.Input(input_vector, index=1)
        yt_modal = pi.Input(input_vector, index=2)

        # system approximation
        wf = build_original_weak_formulation(
            sys_fem_lbl, spatial_domain, control, sys_fem_lbl)
        obs_fem_wf = build_canonical_weak_formulation(
            obs_fem_lbl, spat_domain_cf, control, yt_fem, obs_fem_lbl)
        obs_modal_wf = build_canonical_weak_formulation(
            obs_modal_lbl, spat_domain_cf, control, yt_modal, obs_modal_lbl)

        def calc_eigvals(wf, label):
            ce = pi.parse_weak_formulation(wf)
            dyn_matrix = (
                -np.linalg.inv(
                    ce.dynamic_forms[label].matrices["E"][1][1]) @
                ce.dynamic_forms[label].matrices["E"][0][1])
            return np.sort(
                [np.imag(ev) for ev in np.linalg.eigvals(dyn_matrix)
                 if (0.1 < np.imag(ev) < 15) and np.abs(np.real(ev)) < 0.001])

        imag_ev_sys = calc_eigvals(wf, sys_fem_lbl)
        imag_ev_obs = calc_eigvals(obs_fem_wf, obs_fem_lbl)
        imag_ev_obs_modal = calc_eigvals(obs_modal_wf, obs_modal_lbl)

        self.assertTrue(np.linalg.norm(imag_ev_sys - imag_ev_obs) < 1)
        self.assertTrue(np.linalg.norm(imag_ev_sys - imag_ev_obs_modal) < 1)

    def test_state_transform(self):
        ie = 1
        org_state = SwmBaseFraction(
            [pi.ConstantFunction(ie, domain=(0, 1)),
             pi.ConstantFunction(0, domain=(0, 1))],
            [ie, 0])
        ocf_state = ocf_inverse_state_transform(org_state)
        pprint(ocf_state.members["scalars"])

        f = pi.visualize_functions(ocf_state.members["funcs"],
                                   return_window=True)
        if show_plots:
            f.show()
            pi.show()

    def test_save_results(self):
        data = [pi.EvalData([[0, 1], [0, 1]], np.eye(2))] * 3

        with open(self.res_file, "wb") as f:
           pickle.dump(data, f)

    def test_xtract_and_plot_results(self):
        # the name is needed to make sure that the test date is already created
        # from tkinter import Tk
        # Tk().withdraw()

        with open(self.res_file, "rb") as f:
            raw_data = pickle.load(f)

        data = list()
        for i, item in enumerate(raw_data):
            if i == 0:
                item.name = "System"
            elif i == 2:
                item.name = "Beobachter"
            else:
                continue
            data.append(item)

        if show_plots:
            # _ = SwmPgAnimatedPlot(data, save_pics=True, create_video=True, labels={'bottom': ("z")})
            _ = SwmPgAnimatedPlot(data, labels={'bottom': ("z")})
            pi.show()

    def test_modal_ctrl_bases(self):
        sys_modal_lbl = "modal_system"
        obs_modal_lbl = "modal_observer"
        n_obs_modal = 10
        build_modal_bases(sys_modal_lbl, n_obs_modal, obs_modal_lbl, n_obs_modal)
        get_modal_base_for_ctrl_approximation()

        # controller
        controller = build_controller(sys_modal_lbl, sys_modal_lbl)
        approx_ctrl = approximate_controller(sys_modal_lbl, sys_modal_lbl)
