import unittest

import numpy as np
import pyinduct as pi


class ControllerObserverTestCase(unittest.TestCase):
    def setUp(self):
        self.interval = (0, 1)
        spat_dom = pi.Domain(self.interval, 10)
        nodes = pi.Domain(spat_dom.bounds, num=3)
        base = pi.LagrangeFirstOrder.cure_interval(nodes)
        self.weight_label = "base"
        pi.register_base(self.weight_label, base)

        exp_base = pi.Base(pi.Function(np.exp))
        self.func_label = "exp_base"
        pi.register_base(self.func_label, exp_base, overwrite=True)

        self.alpha = 2
        self.weights = np.array([1, 1, 1, 2, 2, 2])
        self.x = pi.FieldVariable(self.weight_label)
        self.psi = pi.TestFunction(self.weight_label)
        self.exp = pi.ScalarFunction(self.func_label)
        self.out_err = pi.StateFeedback(pi.WeakFormulation(
            [pi.ScalarTerm(self.x(0))], name="observer_error"))

    def _build_ctrl(self, term):
        ctrl = pi.StateFeedback(pi.WeakFormulation([term], name="test_ctrl"))

        return ctrl._calc_output(weights=self.weights,
                                 weight_lbl=self.weight_label)["output"]

    def _build_obs(self, term):
        obs = pi.ObserverFeedback(pi.WeakFormulation([term], name="test_obs"),
                                  self.out_err)

        return obs._calc_output(time=0,
                                weights=self.weights,
                                weight_lbl=self.weight_label,
                                obs_weight_lbl=self.weight_label)["output"]

    def test_collocated_ctrl(self):
        term1c = pi.ScalarTerm(self.x.derive(temp_order=1)(1), 1 + self.alpha)
        term2c = pi.ScalarTerm(self.x.derive(spat_order=1)(0), 2)
        term3c = pi.ScalarTerm(pi.Product(self.x(1), self.exp(1)))

        res = self._build_ctrl(term1c)
        self.assertAlmostEqual(res, 6)

        res = self._build_ctrl(term2c)
        self.assertAlmostEqual(res, 0)

        res = self._build_ctrl(term3c)
        self.assertAlmostEqual(res, 1 * np.exp(1))

    def test_collocated_obs(self):
        term1o = pi.ScalarTerm(self.psi(1), 1 + self.alpha)
        term2o = pi.ScalarTerm(self.psi.derive(1)(0), 2)
        term3o = pi.ScalarTerm(pi.Product(self.psi(0), self.exp(1)))

        res = self._build_obs(term1o)
        np.testing.assert_array_almost_equal(res, np.array([[0], [0], [3]]))

        res = self._build_obs(term2o)
        np.testing.assert_array_almost_equal(res, np.array([[-4], [4], [0]]))

        res = self._build_obs(term3o)
        np.testing.assert_array_almost_equal(res, np.eye(3)[:, 0, None] * np.e)

    def test_continuous_ctrl(self):
        term1c = pi.IntegralTerm(
            self.x.derive(temp_order=1), self.interval, 1 + self.alpha)
        term2c = pi.IntegralTerm(self.x.derive(spat_order=1), self.interval, 2)
        term3c = pi.IntegralTerm(pi.Product(self.x, self.exp), self.interval)

        res = self._build_ctrl(term1c)
        self.assertAlmostEqual(res, 6)

        res = self._build_ctrl(term2c)
        self.assertAlmostEqual(res, 0)

        res = self._build_ctrl(term3c)
        self.assertAlmostEqual(
            float(res), np.array([0.2974425414002563, 0.8416785741175778,
                                  0.5791607129412111, 0, 0, 0]) @ self.weights)

    def test_continuous_obs(self):
        term1c = pi.IntegralTerm(self.psi, self.interval, 1 + self.alpha)
        term2c = pi.IntegralTerm(self.psi.derive(order=1), self.interval, 2)
        term3c = pi.IntegralTerm(pi.Product(self.psi, self.exp), self.interval)

        res = self._build_obs(term1c)
        np.testing.assert_array_almost_equal(
            res, np.array([[0.75], [1.5], [0.75]]))

        res = self._build_obs(term2c)
        np.testing.assert_array_almost_equal(
            res, np.array([[-2], [0], [2]]))

        res = self._build_obs(term3c)
        np.testing.assert_array_almost_equal(
            res, np.array([[0.2974425414002563],
                           [0.8416785741175778],
                           [0.5791607129412111]]))

    def test_observer_errors(self):
        test1o = pi.WeakFormulation([pi.IntegralTerm(
            self.x, limits=self.interval)], name="test")
        test2o = pi.WeakFormulation([pi.IntegralTerm(
            pi.Product(self.x, self.exp), limits=self.interval)], name="test")
        test3o = pi.WeakFormulation([pi.ScalarTerm(pi.ObserverGain(
            pi.ObserverFeedback(pi.WeakFormulation([pi.IntegralTerm(
                self.psi, limits=self.interval)], name="test"),
                self.out_err)))], name="test")

        self.assertRaises(ValueError, pi.ObserverFeedback, test1o, self.out_err)
        self.assertRaises(ValueError, pi.ObserverFeedback, test2o, self.out_err)
        self.assertRaises(ValueError, pi.ObserverFeedback, test3o, self.out_err)

    def tearDown(self):
        pi.deregister_base(self.weight_label)
        pi.deregister_base(self.func_label)


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


