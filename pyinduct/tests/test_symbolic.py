import unittest
import sympy as sp
import pyinduct.symbolic as sy
import pyinduct as pi


class VarPoolTests(unittest.TestCase):

    def test_get_variable(self):

        inp_rlm = "inputs"
        coef_rlm = "coefficients"
        func_rlm = "functions"
        impl_func_rlm = "implemented_functions"
        pool = sy.VariablePool("test variable pool")

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
        f2, f3 = pool.new_implemented_functions(["fi2", "fi3"], [(c,), (d,)],
                                                [f2_imp, f3_imp], impl_func_rlm)
        sp.pprint(f1 * f2 * f3)
        sp.pprint((f1 * f2 * f3).subs([(a, 1), (c, 1), (d, 1)]))
        sp.pprint(sp.lambdify((a,c,d), (f1 * f2 * f3))(1,1,1))
        sp.pprint(
            sp.lambdify([], (f1 * f2 * f3).subs([(a, 1), (c, 1), (d, 1)]))())
        sp.pprint(pool.categories[impl_func_rlm])


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
        self.f1_py = self.var_pool.new_implemented_function(
            "f1", (self.z,), func1, "test")
        self.f2_py = self.var_pool.new_implemented_function(
            "f2", (self.z,), func2, "test")

        func1_pi = pi.Function(func1, derivative_handles=[d_func1])
        func2_pi = pi.Function(func2, derivative_handles=[d_func2])
        self.f1_pi = self.var_pool.new_implemented_function(
            "f1_p", (self.z,), func1_pi, "test")
        self.f2_pi = self.var_pool.new_implemented_function(
            "f2_p", (self.z,), func2_pi, "test")

        self.f1_sp = 2 - self.z
        self.f2_sp = self.z

    def test_python_functions(self):
        pair1 = (self.f1_py,)
        pair2 = (self.f2_py,)
        self.check_all_combinations(pair1, pair2)

    def test_pyinduct_functions(self):
        pair1 = (self.f1_pi,)
        pair2 = (self.f2_pi,)
        self.check_all_combinations(pair1, pair2)

    def test_sympy_functions(self):
        pair1 = (self.f1_sp,)
        pair2 = (self.f2_sp,)
        self.check_all_combinations(pair1, pair2)

    def check_all_combinations(self, pair1, pair2):
        combs = [(fu1, fu2) for fu1 in pair1 for fu2 in pair2]

        for comb in combs:
            fu1, fu2 = comb
            expr1 = sp.Integral(fu1, (self.z, 0, 2))
            expr2 = sp.Integral(fu2, (self.z, 0, 2))
            expr12 = sp.Integral(fu1 * fu2, (self.z, 0, 2))
            expr_p = expr1 + expr2 + expr12
            expr_m = expr1 * expr2 * expr12
            self.check_funcs(expr1, expr2, expr12, expr_p, expr_m)

            d_expr1 = sp.Integral(sp.diff(fu1, self.z), (self.z, 0, 2))
            d_expr2 = sp.Integral(sp.diff(fu2, self.z), (self.z, 0, 2))
            d_expr12 = sp.Integral(fu1 * sp.diff(fu2, self.z), (self.z, 0, 2))
            d_expr_p = d_expr1 + d_expr2 + d_expr12
            d_expr_m = d_expr1 * d_expr2 * d_expr12
            # since we can not derive a plain callable
            # we have to take care of this case
            ni_1 = (hasattr(fu1, "_imp_") and
                     not isinstance(fu1._imp_, pi.Function))
            ni_2 = (hasattr(fu2, "_imp_") and
                     not isinstance(fu2._imp_, pi.Function))
            self.check_derived_funcs(d_expr1, d_expr2, d_expr12, d_expr_p,
                                     d_expr_m, ni_1, ni_2)

    def test_combinations(self):
        pair1 = (self.f1_pi, self.f1_sp)
        pair2 = (self.f2_pi, self.f2_sp)
        self.check_all_combinations(pair1, pair2)

        pair1 = (self.f1_py, self.f1_sp)
        pair2 = (self.f2_py, self.f2_sp)
        self.check_all_combinations(pair1, pair2)

        pair1 = (self.f1_pi, self.f1_py)
        pair2 = (self.f2_pi, self.f2_py)
        self.check_all_combinations(pair1, pair2)


    def check_funcs(self, expr1, expr2, expr12, expr_p, expr_m):
        self.assertAlmostEqual(sy.evaluate_integrals(expr1), 2)
        self.assertAlmostEqual(sy.evaluate_integrals(expr2), 2)
        self.assertAlmostEqual(sy.evaluate_integrals(expr12), 4 / 3)
        self.assertAlmostEqual(sy.evaluate_integrals(expr_p), 16 / 3)
        self.assertAlmostEqual(sy.evaluate_integrals(expr_m), 16 / 3)

    def check_derived_funcs(self, d_expr1, d_expr2, d_expr12, d_expr_p,
                            d_expr_m, ni_1=False, ni_2=False):

        if ni_1:
            with self.assertRaises(NotImplementedError):
                sy.evaluate_integrals(d_expr1)

        else:
            self.assertAlmostEqual(sy.evaluate_integrals(d_expr1), -2)

        if ni_2:
            with self.assertRaises(NotImplementedError):
                sy.evaluate_integrals(d_expr2)
            with self.assertRaises(NotImplementedError):
                sy.evaluate_integrals(d_expr12)

        else:
            self.assertAlmostEqual(sy.evaluate_integrals(d_expr2), 2)
            self.assertAlmostEqual(sy.evaluate_integrals(d_expr12), 2)

        if ni_1 or ni_2:
            with self.assertRaises(NotImplementedError):
                sy.evaluate_integrals(d_expr_m)
            with self.assertRaises(NotImplementedError):
                sy.evaluate_integrals(d_expr_p)

        else:
            self.assertAlmostEqual(sy.evaluate_integrals(d_expr_p), 2)
            self.assertAlmostEqual(sy.evaluate_integrals(d_expr_m), -8)


    def tearDown(self):
        sy.VariablePool.variable_pool_registry.clear()

