import unittest

from pyinduct.tests import test_examples

skip_msg = "Test of examples was deactivated."


class TestAllExamples(unittest.TestCase):
    """
    Here you can check if all examples run fine again.
    By unittest discover or setup this test case
    will be skipped (see __init__.py).
    """
    @unittest.skipIf(not test_examples, skip_msg)
    def test_parabolic_in_domain(self):
        import pyinduct.examples.rad_eq_in_domain as exp
        exp.run(False)

    @unittest.skipIf(not test_examples, skip_msg)
    def test_parabolic_var_coeff(self):
        import pyinduct.examples.rad_eq_var_coeff as exp
        exp.run(False)

    @unittest.skipIf(not test_examples, skip_msg)
    def test_parabolic_const_coeff(self):
        import pyinduct.examples.rad_eq_const_coeff as exp
        exp.run(False)

    @unittest.skipIf(not test_examples, skip_msg)
    def test_parabolic_minimal(self):
        import pyinduct.examples.rad_eq_minimal as exp
        exp.run(False)

    @unittest.skipIf(not test_examples, skip_msg)
    def test_parabolic_dirichlet_fem(self):
        import pyinduct.examples.rad_eq_diri_fem as exp
        exp.run(False)

    @unittest.skipIf(not test_examples, skip_msg)
    def test_transport_system(self):
        import pyinduct.examples.transport_system as exp
        exp.run(False)

    @unittest.skipIf(not test_examples, skip_msg)
    def test_diff_eq_multiple_inputs(self):
        import pyinduct.examples.diff_eq_multiple_inputs as exp
        exp.run(False)

    @unittest.skipIf(not test_examples, skip_msg)
    def test_euler_bernoulli_beam(self):
        import pyinduct.examples.euler_bernoulli_beam as exp
        exp.run(False)

    @unittest.skipIf(not test_examples, skip_msg)
    def test_pipe_flow(self):
        import pyinduct.examples.pipe_flow as exp
        exp.run(False)

    @unittest.skipIf(not test_examples, skip_msg)
    def test_string_with_mass(self):
        import pyinduct.examples.string_with_mass.main as exp
        exp.run(False)

