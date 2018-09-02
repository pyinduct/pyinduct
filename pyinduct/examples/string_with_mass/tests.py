from pyinduct.examples.string_with_mass.control import *
import pyqtgraph as pg
import  unittest


class StringWithMassTest(unittest.TestCase):
    """
    Its not a real test case class.
    Just a few code snippets for tests during development of this example.
    From the most python ide's each code snippet/test method
    can easily executed via a keyboard shortcut .
    """

    def test_find_eigenvalues(self):
        eig_om, eig_vals = find_eigenvalues(4)
        plot_eigenvalues(eig_vals)
        pprint(eig_vals)

    def test_observer_evp_scripts(self):
        from pyinduct.examples.string_with_mass.observer_evp_scripts.modal_approximation \
            import build_bases_for_modal_observer_approximation, validate_modal_bases
        pb, db, pbn, dbn, eig_vals = build_bases_for_modal_observer_approximation(10)
        a, b, c, l, l_ub = validate_modal_bases(pb, db, pbn, dbn, eig_vals)


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
            SwmBaseCanonicalFraction.scalar_product,
            pi.get_base("dual_base_can"), pi.get_base("primal_base_can")))
        SwmBaseFraction.l2_scalar_product = False
        pprint(pi.calculate_scalar_product_matrix(
            SwmBaseFraction.scalar_product,
            pi.get_base("primal_base"), pi.get_base("dual_base")))
        SwmBaseFraction.l2_scalar_product = True

        plot = 0
        if plot:
            plots = list()
            plots.append(pi.visualize_functions([frac.members["funcs"][0]
                                    for frac in pi.get_base("primal_base")],
                                   return_window=True))
            plots.append(pi.visualize_functions([frac.members["funcs"][1]
                                    for frac in pi.get_base("primal_base")],
                               return_window=True))
            pg.QAPP.exec_()

    def test_modal_cf_wf(self):
        n = 4
        base_label = "base"
        base_label_cf = "base_cf"
        build_modal_bases(base_label, None, None, base_label_cf, n)
        wf = build_canonical_weak_formulation(base_label_cf, pi.Domain((-1, 1), 2), pi.ConstantTrajectory(0), "")
        ce = pi.parse_weak_formulation(wf)
        pprint(ce.dynamic_forms[base_label_cf].matrices["E"][1][1])
        pprint(ce.dynamic_forms[base_label_cf].matrices["E"][0][1])
