import os
import time
import pickle
import unittest
import pyqtgraph as pg

from pyinduct.tests import show
from pyinduct.examples.string_with_mass.control import *


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
        show(show_pg=False)

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
        show()

    @unittest.skip("Test case is incomplete and broken")
    def test_modal_cf_wf(self):
        # TODO fix these calls and add an actual test

        n = 4
        n_cf = 4  # HACK added n_cf since interface of build_modal_base_changed
        base_label = "base"
        base_label_cf = "base_cf"
        build_modal_bases(base_label, n,  base_label_cf, n_cf)
        # It seems like a call to the function below is missing
        # init_observer_gain(sys_fem_lbl, sys_modal_lbl, obs_fem_lbl, obs_modal_lbl)
        wf = build_canonical_weak_formulation(base_label_cf,
                                              pi.Domain((-1, 1), 2),
                                              pi.ConstantTrajectory(0), "")
        ce = pi.parse_weak_formulation(wf)
        pprint(ce.dynamic_forms[base_label_cf].matrices["E"][1][1])
        pprint(ce.dynamic_forms[base_label_cf].matrices["E"][0][1])

    def test_state_transform(self):
        ie = 1
        org_state = SwmBaseFraction(
            [pi.ConstantFunction(ie, domain=(0, 1)),
             pi.ConstantFunction(0, domain=(0, 1))],
            [ie, 0])
        ocf_state = ocf_inverse_state_transform(org_state)
        pprint(ocf_state.members["scalars"])

        _ = pi.visualize_functions(ocf_state.members["funcs"],
                                   return_window=True)
        show()

    def test_save_results(self):
        data = [pi.EvalData([[0, 1], [0, 1]], np.eye(2))] * 3

        with open(self.res_file, "wb") as f:
           pickle.dump(data, f)

    def test_xtract_and_plot_results(self):
        # the name is needed to make sure that the test date is already created
        from tkinter import Tk
        Tk().withdraw()

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

        # _ = SwmPgAnimatedPlot(data, save_pics=True, create_video=True, labels={'bottom': ("z")})
        _ = SwmPgAnimatedPlot(data, labels={'bottom': ("z")})
        show()

    def test_modal_ctrl_bases(self):
        sys_modal_lbl = "modal_system"
        obs_modal_lbl = "modal_observer"
        n_obs_modal = 10
        build_modal_bases(sys_modal_lbl, n_obs_modal, obs_modal_lbl, n_obs_modal)
        get_modal_base_for_ctrl_approximation()

        # controller
        controller = build_controller(sys_modal_lbl, sys_modal_lbl)
        approx_ctrl = approximate_controller(sys_modal_lbl, sys_modal_lbl)
