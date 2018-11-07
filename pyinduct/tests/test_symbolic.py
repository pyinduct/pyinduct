import unittest
import sympy as sp
import numpy as np
import pyinduct.symbolic as sy
import pyinduct as pi


class SymbolicTests(unittest.TestCase):

    def test_get_variable(self):

        inp_rlm = "inputs"
        coef_rlm = "coefficients"
        func_rlm = "functions"
        impl_func_rlm = "implemented_functions"
        pool = sy.VariablePool()

        a = pool.new_symbol("a", inp_rlm)
        c, d = pool.new_symbols(["c", "d"], coef_rlm)
        sp.pprint(a*c*d)
        sp.pprint(pool.categories[coef_rlm])

        f1 = pool.new_function("f1", (a,), func_rlm)
        f2, f3 = pool.new_functions(["f2", "f3"], [(c,), (d,)], func_rlm)
        sp.pprint(f1 * f2 * f3)
        sp.pprint(pool.categories[func_rlm])

        f1_imp = lambda z: 1
        f2_imp = lambda z: 2
        f3_imp = lambda z: 3
        f1 = pool.new_implemented_function("fi1", (a,), f1_imp, impl_func_rlm)
        f2, f3 = pool.new_implemented_functions(["fi2", "fi3"], [(c,), (d,)], [f2_imp, f3_imp], impl_func_rlm)
        sp.pprint(f1 * f2 * f3)
        sp.pprint((f1 * f2 * f3).subs([(a, 1), (c, 1), (d, 1)]))
        sp.pprint(sp.lambdify((a,c,d), (f1 * f2 * f3))(1,1,1))
        sp.pprint(sp.lambdify([], (f1 * f2 * f3).subs([(a, 1), (c, 1), (d, 1)]))())
        sp.pprint(pool.categories[impl_func_rlm])


        expr = sp.diff(f1, a)
        sp.pprint(expr.atoms(sp.Derivative))
        for der in expr.atoms(sp.Derivative):
            sp.pprint(der.atoms(sp.UndefinedFunction)._imp_)
        # lam_f1 = sp.lambdify(a, expr, modules=Derivative)
        # sp.pprint(lam_f1(1))

class EvaluateIntegralTestCase(unittest.TestCase):

    def setUp(self):
        def func1(z):
            return 2 - z
        def d_func1(z):
            return -1

        def func2(z):
            return z
        def d_func2(z):
            return 1

        self.var_pool = sy.VariablePool("integral test case")
        self.z, self.t = self.var_pool.new_symbols(["z", "t"], "test")
        self.f1 = self.var_pool.new_implemented_function(
            "f1", (self.z,), func1, "test")
        self.f2 = self.var_pool.new_implemented_function(
            "f2", (self.z,), func2, "test")

        func1_pi = pi.Function(func1, derivative_handles=[d_func1])
        func2_pi = pi.Function(func2, derivative_handles=[d_func2])
        self.f1_pi = self.var_pool.new_implemented_function(
            "f1_p", (self.z,), func1_pi, "test")
        self.f2_pi = self.var_pool.new_implemented_function(
            "f2_p", (self.z,), func2_pi, "test")

    def test_python_functions(self):
        expr1 = sp.Integral(self.f1, (self.z, 0, 2))
        expr2 = sp.Integral(self.f2, (self.z, 0, 2))
        expr12 = sp.Integral(self.f1 * self.f2, (self.z, 0, 2))
        expr_a = expr1 * expr2 * expr12

        self.assertAlmostEqual(sy.evaluate_integrals(expr1), 2)
        self.assertAlmostEqual(sy.evaluate_integrals(expr2), 2)
        self.assertAlmostEqual(sy.evaluate_integrals(expr12), 4 / 3)
        self.assertAlmostEqual(sy.evaluate_integrals(expr_a), 16 / 3)

    def test_pyinduct_functions(self):
        expr1 = sp.Integral(self.f1_pi, (self.z, 0, 2))
        expr2 = sp.Integral(self.f2_pi, (self.z, 0, 2))
        expr12 = sp.Integral(self.f1_pi * self.f2_pi, (self.z, 0, 2))
        expr_a = expr1 * expr2 * expr12

        self.assertAlmostEqual(sy.evaluate_integrals(expr1), 2)
        self.assertAlmostEqual(sy.evaluate_integrals(expr2), 2)
        self.assertAlmostEqual(sy.evaluate_integrals(expr12), 4 / 3)
        self.assertAlmostEqual(sy.evaluate_integrals(expr_a), 16 / 3)

        d_expr1 = sp.Integral(sp.diff(self.f1_pi, self.z), (self.z, 0, 2))
        d_expr2 = sp.Integral(sp.diff(self.f2_pi, self.z), (self.z, 0, 2))
        d_expr12 = sp.Integral(self.f1_pi * sp.diff(self.f2_pi, self.z), (self.z, 0, 2))
        d_expr_a = d_expr1 * d_expr2 * d_expr12

        self.assertAlmostEqual(sy.evaluate_integrals(d_expr1), -2)
        self.assertAlmostEqual(sy.evaluate_integrals(d_expr2), 2)
        self.assertAlmostEqual(sy.evaluate_integrals(d_expr12), 2)
        self.assertAlmostEqual(sy.evaluate_integrals(d_expr_a), -8)
