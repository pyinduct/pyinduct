from pyinduct.examples.string_with_mass.utils import *
from pyinduct.examples.string_with_mass.system import *
import  unittest


class PneumaticSystemTests(unittest.TestCase):

    def test_find_eigenvalues(self):
        eig_om, eig_vals = find_eigenvalues(4)
        plot_eigenvalues(eig_vals)
        pprint(eig_vals)

    def test_modal_base(self):
        modal_base_lbl = "modal_primal_base"
        build_primal_modal_bases(modal_base_lbl, self.eigenvalues[:13])

        pi.visualize_functions([frac.members["funcs"][0]
                                for frac in pi.get_base(modal_base_lbl)])
