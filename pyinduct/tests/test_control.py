import unittest

import numpy as np
import pyinduct as pi


class CollocatedTestCase(unittest.TestCase):
    def setUp(self):
        spat_dom = pi.Domain((0, 1), 10)
        nodes = pi.Domain(spat_dom.bounds, num=3)
        base = pi.LagrangeFirstOrder.cure_interval(nodes)
        pi.register_base("base", base)

        x = pi.FieldVariable("base")
        x_dt_at1 = x.derive(temp_order=1)(1)
        x_dz_at0 = x.derive(spat_order=1)(0)

        exp_base = pi.Base(pi.Function(np.exp))
        pi.register_base("exp_base", exp_base, overwrite=True)
        exp_at1 = pi.ScalarFunction("exp_base")(1)

        alpha = 2
        self.term1 = pi.ScalarTerm(x_dt_at1, 1 + alpha)
        self.term2 = pi.ScalarTerm(x_dz_at0, 2)
        self.term3 = pi.ScalarTerm(pi.Product(x(0), exp_at1))

        self.weight_label = "base"
        self.weights = np.array([1, 1, 1, 2, 2, 2])

    def _build_case(self, term):
        ce = pi.parse_weak_formulation(pi.WeakFormulation([term], name="test"),
                                       finalize=False)
        evaluator = pi.LawEvaluator(ce)
        return evaluator(self.weights, self.weight_label)["output"]

    def test_temp_term(self):
        res = self._build_case(self.term1)
        self.assertAlmostEqual(res, 6)

    def test_spat_term(self):
        res = self._build_case(self.term2)
        self.assertAlmostEqual(res, 0)

    def test_product_term(self):
        res = self._build_case(self.term3)
        self.assertAlmostEqual(res, 1 * np.exp(1))

    def tearDown(self):
        pi.deregister_base("base")
        pi.deregister_base("exp_base")


class ContinuousTestCase(unittest.TestCase):
    def setUp(self):
        self.weight_label = "base"

        interval = (0, 1)
        nodes = pi.Domain(interval, num=3)
        funcs = pi.LagrangeFirstOrder.cure_interval(nodes)
        pi.register_base(self.weight_label, funcs)

        x = pi.FieldVariable(self.weight_label)
        x_dt = x.derive(temp_order=1)
        x_dz = x.derive(spat_order=1)

        pi.register_base("scalar_func", pi.Base(pi.Function(np.exp)))
        exp = pi.ScalarFunction("scalar_func")

        alpha = 2
        self.term1 = pi.IntegralTerm(x_dt, interval, 1 + alpha)
        self.term2 = pi.IntegralTerm(x_dz, interval, 2)
        self.term3 = pi.IntegralTerm(pi.Product(x, exp), interval)

        self.weights = np.array([1, 1, 1, 2, 2, 2])

    def _build_case(self, term):
        ce = pi.parse_weak_formulation(pi.WeakFormulation([term], name="test"),
                                       finalize=False)
        evaluator = pi.LawEvaluator(ce)
        return evaluator(self.weights, self.weight_label)["output"]

    def test_temp_term(self):
        res = self._build_case(self.term1)
        self.assertTrue(np.equal(res, 6))

    def test_spat_term(self):
        res = self._build_case(self.term2)
        self.assertAlmostEqual(res, 0)

    def test_product_term(self):
        res = self._build_case(self.term3)
        # TODO calculate expected result
        # self.assertAlmostEqual(res, 1*np.exp(1))

    def tearDown(self):
        pi.deregister_base(self.weight_label)
        pi.deregister_base("scalar_func")


class TestControllerInExamplesModule(unittest.TestCase):
    def test_dirichlet(self):
        import pyinduct.examples.rad_eq_minimal

    def test_robin(self):
        import pyinduct.examples.rad_eq_const_coeff

    @unittest.skip("This takes about 22 minutes on my machine, Travis will"
                   "quit after 10 minutes.")
    def test_robin_spatially_varying_coefficients(self):
        import pyinduct.examples.rad_eq_var_coeff

    def test_robin_in_domain(self):
        import pyinduct.examples.rad_eq_in_domain

