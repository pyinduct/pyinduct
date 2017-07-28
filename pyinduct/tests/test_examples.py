import unittest

from pyinduct.tests import test_all_examples

skip_msg = "Quick test of all examples, must be started manually."
class TestAllExamples(unittest.TestCase):
    """
    Here you can check if all examples run fine again.
    By unittest discover or setup this test case
    will be skipped (see __init__.py).
    """
    @unittest.skipIf(not test_all_examples, skip_msg)
    def test_parabolic_in_domain(self):
        import pyinduct.examples.rad_eq_in_domain

    @unittest.skip
    def test_parabolic_var_coeff(self):
        import pyinduct.examples.rad_eq_var_coeff

    @unittest.skipIf(not test_all_examples, skip_msg)
    def test_parabolic_const_coeff(self):
        import pyinduct.examples.rad_eq_const_coeff

    @unittest.skipIf(not test_all_examples, skip_msg)
    def test_parabolic_minimal(self):
        import pyinduct.examples.rad_eq_minimal

    @unittest.skipIf(not test_all_examples, skip_msg)
    def test_parabolic_dirichlet_fem(self):
        import pyinduct.examples.rad_eq_diri_fem

    @unittest.skipIf(not test_all_examples, skip_msg)
    def test_transport_system(self):
        import pyinduct.examples.transport_system

