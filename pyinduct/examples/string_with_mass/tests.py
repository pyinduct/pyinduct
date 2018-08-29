from pyinduct.examples.string_with_mass.utils import *
from pyinduct.examples.string_with_mass.system import *
import  unittest


class PneumaticSystemTests(unittest.TestCase):

    def test_sort_eigenvalues(self):
        eigenvalues = self.eigenvalues_
        sorted_eigenvalues = sort_eigenvalues(eigenvalues)
        pprint(sorted_eigenvalues)
        plot_eigenvalues(sorted_eigenvalues)

    def test_find_eigenvalues(self):
        # print characteristic equation
        phi = get_primal_eigenvector(according_paper=True)
        char_eq = phi[1].subs(sym.z, 1)
        sp.pprint(char_eq.subs(subs_list), num_columns=108)

        # compute and show eigenvalues
        phi = get_primal_eigenvector()
        phi = sp.simplify(sp.expand(phi.subs(subs_list)))
        char_eq = phi[1].subs(sym.z, 1)
        grid = [np.linspace(-10, 5, 200), np.linspace(-1000, 1000, 200)]
        eig_vals = find_eigenvalues(char_eq, None, grid)
        plot_eigenvalues(eig_vals)
        pprint(eig_vals)

    def test_modal_base(self):
        modal_base_lbl = "modal_primal_base"
        build_primal_modal_bases(modal_base_lbl, self.eigenvalues[:13])

        pi.visualize_functions([frac.members["funcs"][0]
                                for frac in pi.get_base(modal_base_lbl)])

    def test_complex_modal_base(self):
        modal_base_lbl = "modal_primal_base"
        build_primal_modal_bases(modal_base_lbl, self.eigenvalues[:13], complex_=True)

        pi.visualize_functions([frac.members["funcs"][0]
                                for frac in pi.get_base(modal_base_lbl)])

    # to save time during test cases
    eigenvalues_ = np.array([
        -3.76033058e+00 - 9.52074726e+02j, -3.76033058e+00 - 8.97726688e+02j,
        -3.76033058e+00 - 8.43382371e+02j, -3.76033058e+00 - 7.89042542e+02j,
        -3.76033058e+00 - 7.34708193e+02j, -3.76033058e+00 - 6.80380633e+02j,
        -3.76033058e+00 - 6.26061622e+02j, -3.76033058e+00 - 5.71753582e+02j,
        -3.76033058e+00 - 5.17459947e+02j, -3.76033058e+00 - 4.63185745e+02j,
        -3.76033058e+00 - 4.08938639e+02j, -3.76033058e+00 - 3.54730904e+02j,
        -3.76033058e+00 - 3.00583486e+02j, -3.76033058e+00 - 2.46535175e+02j,
        -3.76033058e+00 - 1.92666315e+02j, -3.76033058e+00 - 1.39172067e+02j,
        -3.76033058e+00 - 8.66540079e+01j, -3.76033058e+00 - 3.76414628e+01j,
        -3.76033058e+00 + 3.76414628e+01j, -3.76033058e+00 + 8.66540079e+01j,
        -3.76033058e+00 + 1.39172067e+02j, -3.76033058e+00 + 1.92666315e+02j,
        -3.76033058e+00 + 2.46535175e+02j, -3.76033058e+00 + 3.00583486e+02j,
        -3.76033058e+00 + 3.54730904e+02j, -3.76033058e+00 + 4.08938639e+02j,
        -3.76033058e+00 + 4.63185745e+02j, -3.76033058e+00 + 5.17459947e+02j,
        -3.76033058e+00 + 5.71753582e+02j, -3.76033058e+00 + 6.26061622e+02j,
        -3.76033058e+00 + 6.80380633e+02j, -3.76033058e+00 + 7.34708193e+02j,
        -3.76033058e+00 + 7.89042542e+02j, -3.76033058e+00 + 8.43382371e+02j,
        -3.76033058e+00 + 8.97726688e+02j, -3.76033058e+00 + 9.52074726e+02j,
        +2.65479223e-33 - 4.31972942e-32j])
    eigenvalues = sort_eigenvalues(eigenvalues_)
