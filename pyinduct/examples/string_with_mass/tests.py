from pyinduct.examples.string_with_mass.utils import *
from pyinduct.examples.string_with_mass.system import *
import pyqtgraph as pg
import  unittest


class StringWithMassTest(unittest.TestCase):

    def test_find_eigenvalues(self):
        eig_om, eig_vals = find_eigenvalues(4)
        plot_eigenvalues(eig_vals)
        pprint(eig_vals)

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

