import unittest
import sympy as sp
from pyinduct.symbolic import *
from sympy.utilities.lambdify import implemented_function


class SymbolicTests(unittest.TestCase):

    def test_get_variable(self):

        inp_rlm = "inputs"
        coef_rlm = "coefficients"
        func_rlm = "functions"
        impl_func_rlm = "implemented_functions"
        pool = VariablePool()

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
