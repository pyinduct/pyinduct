import sys
import collections
import unittest
from numbers import Number

import numpy as np

from pyinduct import register_base, get_base, core, shapefunctions

if any([arg in {'discover', 'setup.py', 'test'} for arg in sys.argv]):
    show_plots = False
else:
    # show_plots = True
    show_plots = False

if show_plots:
    import pyqtgraph as pg

    app = pg.QtGui.QApplication([])


class SanitizeInputTestCase(unittest.TestCase):
    def test_scalar(self):
        self.assertRaises(TypeError, core.sanitize_input, 1.0, int)
        core.sanitize_input(1, int)
        core.sanitize_input(1.0, float)


class FunctionTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_init(self):
        self.assertRaises(TypeError, core.Function, 42)
        p = core.Function(np.sin)
        # default kwargs
        self.assertEqual(p.domain, [(-np.inf, np.inf)])
        self.assertEqual(p.nonzero, [(-np.inf, np.inf)])

        for kwarg in ["domain", "nonzero"]:
            # some nice but wrong variants
            for val in ["4-2", dict(start=1, stop=2), [1, 2]]:
                self.assertRaises(TypeError, core.Function, np.sin, **{kwarg: val})

            # a correct one
            core.Function(np.sin, **{kwarg: (0, 10)})
            core.Function(np.sin, **{kwarg: [(0, 3), (5, 10)]})

            # check sorting
            p = core.Function(np.sin, **{kwarg: (0, -10)})
            self.assertEqual(getattr(p, kwarg), [(-10, 0)])
            p = core.Function(np.sin, **{kwarg: [(5, 0), (-10, -5)]})
            self.assertEqual(getattr(p, kwarg), [(-10, -5), (0, 5)])

            if kwarg == "domain":
                # check domain check
                self.assertRaises(ValueError, p, -3)
                self.assertRaises(ValueError, p, 10)
            else:
                # TODO check if nonzero check generates warning
                pass

        # test stupid handle
        def wrong_handle(x):
            return np.array([x, x])

        self.assertRaises(TypeError, core.Function, wrong_handle)

    def test_derivation(self):
        f = core.Function(np.sin, derivative_handles=[np.cos, np.sin])

        # be robust to meaningless input
        self.assertRaises(ValueError, f.derive, -1)  # stupid derivative
        self.assertRaises(ValueError, f.derive, 3)  # unknown derivative
        self.assertRaises(ValueError, f.derive, 100)  # unknown derivative

        # zeroth derivative should return the function itself
        d0 = f.derive(0)
        self.assertEqual(f, d0)

        d_default = f.derive()

        # default arg should be one
        d1 = f.derive()
        d_default.__dict__.pop("members")
        d1.__dict__.pop("members")
        self.assertEqual(d_default.__dict__, d1.__dict__)

        # derivatives should change
        p_func = d1.__dict__.pop("_function_handle")
        self.assertEqual(p_func, np.cos)

        # list of handles should get shorter
        p_deriv = d1.__dict__.pop("_derivative_handles")
        self.assertEqual(p_deriv, [np.sin])

        # rest should stay the same
        f.__dict__.pop("_function_handle")
        f.__dict__.pop("_derivative_handles")
        f.__dict__.pop("members")
        self.assertEqual(d1.__dict__, f.__dict__)

        f_2 = core.Function(np.sin, derivative_handles=[np.cos, np.sin])
        d2 = f_2.derive(2)

        # derivatives should change
        p_func = d2.__dict__.pop("_function_handle")
        self.assertEqual(p_func, np.sin)

        # list of handles should get shorter
        p_deriv = d2.__dict__.pop("_derivative_handles")
        self.assertEqual(p_deriv, [])

        # rest should stay the same
        f_2.__dict__.pop("_function_handle")
        f_2.__dict__.pop("_derivative_handles")
        f_2.__dict__.pop("members")
        d2.__dict__.pop("members")
        self.assertEqual(d2.__dict__, f_2.__dict__)

    def test_scale(self):
        f = core.Function(np.sin, derivative_handles=[np.cos, np.sin])

        # no new object since trivial scaling occurred
        g1 = f.scale(1)
        self.assertEqual(f, g1)

        # after scaling, return scalars and vectors like normal
        g2 = f.scale(10)

        self.assertIsInstance(g2(5), Number)
        self.assertNotIsInstance(g2(5), np.ndarray)
        self.assertTrue(np.array_equal(10 * np.sin(list(range(100))), g2(list(range(100)))))

        # scale with function
        g3 = f.scale(lambda z: z)

        def check_handle(z):
            return z * f(z)

        self.assertIsInstance(g3(5), Number)
        self.assertNotIsInstance(g3(5), np.ndarray)
        self.assertTrue(np.array_equal(g3(list(range(10))), check_handle(list(range(10)))))
        self.assertRaises(ValueError, g3.derive, 1)  # derivatives should be removed when scaled by function

    def test_raise(self):
        f = core.Function(np.sin, derivative_handles=[np.cos, np.sin])

        # no new object since trivial scaling occurred
        g1 = f.raise_to(1)
        self.assertEqual(f, g1)

        # after scaling, return scalars and vectors like normal
        g2 = f.raise_to(2)

        self.assertIsInstance(g2(5), Number)
        self.assertNotIsInstance(g2(5), np.ndarray)
        self.assertTrue(np.array_equal(np.sin(np.array(range(100))) ** 2,
                                       g2(np.array(range(100)))))
        self.assertRaises(ValueError, g2.derive, 1)  # derivatives should be removed when scaled by function

    def test_call(self):

        def func(x):
            if isinstance(x, collections.Iterable):
                raise TypeError("no vectorial stuff allowed!")
            return 2 ** x

        f = core.Function(func, domain=(0, 10))
        self.assertEqual(f._vectorial, False)  # function handle should be recognized as non-vectorial

        # call with scalar should return scalar with correct value
        self.assertIsInstance(f(10), Number)
        self.assertNotIsInstance(f(10), np.ndarray)
        self.assertEqual(f(10), func(10))

        # vectorial arguments should be understood and an np.ndarray shall be returned
        self.assertIsInstance(f(np.array(range(10))), np.ndarray)
        self.assertTrue(np.array_equal(f(np.array(range(10))), [func(val) for val in range(10)]))

    def test_vector_call(self):

        def vector_func(x):
            return 2 * x

        f = core.Function(vector_func, domain=(0, 10))
        self.assertEqual(f._vectorial, True)  # function handle should be recognized as vectorial

        # call with scalar should return scalar with correct value
        self.assertIsInstance(f(10), Number)
        self.assertNotIsInstance(f(10), np.ndarray)
        self.assertEqual(f(10), vector_func(10))

        # vectorial arguments should be understood and an np.ndarray shall be returned
        self.assertIsInstance(f(np.array(range(10))), np.ndarray)
        self.assertTrue(np.array_equal(f(np.array(range(10))), [vector_func(val) for val in range(10)]))


class IntersectionTestCase(unittest.TestCase):
    def test_wrong_arguments(self):
        # interval bounds not sorted
        self.assertRaises(ValueError, core.domain_intersection, (3, 2), (1, 3))
        # intervals not sorted
        self.assertRaises(ValueError, core.domain_intersection, [(4, 5), (1, 2)], (1, 3))
        # intervals useless
        self.assertRaises(ValueError, core.domain_intersection, [(4, 5), (5, 6)], (1, 3))

    def test_easy_intersections(self):
        self.assertEqual(core.domain_intersection((0, 2), (1, 3)), [(1, 2)])
        self.assertEqual(core.domain_intersection((0, 1), (1, 3)), [])
        self.assertEqual(core.domain_intersection((3, 5), (1, 3)), [])
        self.assertEqual(core.domain_intersection((3, 5), (1, 4)), [(3, 4)])
        self.assertEqual(core.domain_intersection((3, 5), (1, 6)), [(3, 5)])
        self.assertEqual(core.domain_intersection((3, 5), (6, 7)), [])

    def test_complex_intersections(self):
        self.assertEqual(core.domain_intersection([(0, 2), (3, 5)], (3, 4)), [(3, 4)])
        self.assertEqual(core.domain_intersection([(0, 2), (3, 5)], (1, 4)), [(1, 2), (3, 4)])
        self.assertEqual(core.domain_intersection((1, 4), [(0, 2), (3, 5)]), [(1, 2), (3, 4)])
        self.assertEqual(core.domain_intersection([(1, 3), (4, 6)], [(0, 2), (3, 5)]), [(1, 2), (4, 5)])
        self.assertEqual(core.domain_intersection([(-10, -4), (2, 5), (10, 17)], [(-20, -5), (3, 5), (7, 23)]),
                         [(-10, -5), (3, 5)], (10, 17))


class ScalarDotProductL2TestCase(unittest.TestCase):
    def setUp(self):
        self.f1 = core.Function(lambda x: 1, domain=(0, 10))
        self.f2 = core.Function(lambda x: 2, domain=(0, 5))
        self.f3 = core.Function(lambda x: 2, domain=(0, 5), nonzero=(2, 3))
        self.f4 = core.Function(lambda x: 2, domain=(0, 5), nonzero=(2, 2 + 1e-1))

        self.f5 = shapefunctions.LagrangeFirstOrder(0, 1, 2)
        self.f6 = shapefunctions.LagrangeFirstOrder(1, 2, 3)
        self.f7 = shapefunctions.LagrangeFirstOrder(2, 3, 4)

    def test_domain(self):
        self.assertAlmostEqual(core._dot_product_l2(self.f1, self.f2), 10)
        self.assertAlmostEqual(core._dot_product_l2(self.f1, self.f3), 2)

    def test_nonzero(self):
        self.assertAlmostEqual(core._dot_product_l2(self.f1, self.f4), 2e-1)

    def test_lagrange(self):
        self.assertAlmostEqual(core._dot_product_l2(self.f5, self.f7), 0)
        self.assertAlmostEqual(core._dot_product_l2(self.f5, self.f6), 1 / 6)
        self.assertAlmostEqual(core._dot_product_l2(self.f7, self.f6), 1 / 6)
        self.assertAlmostEqual(core._dot_product_l2(self.f5, self.f5), 2 / 3)


# TODO tests for dot_product_l2 (vectorial case)


class CalculateScalarProductMatrixTestCase(unittest.TestCase):
    def setUp(self):
        interval = (0, 10)
        nodes = 5
        self.nodes1, self.initial_functions1 = shapefunctions.cure_interval(shapefunctions.LagrangeFirstOrder, interval,
                                                                            node_count=nodes)
        self.nodes2, self.initial_functions2 = shapefunctions.cure_interval(shapefunctions.LagrangeFirstOrder, interval,
                                                                            node_count=2 * nodes - 1)
        self.optimization = None
        print(np.array(self.nodes1), np.array(self.nodes2))

    def run_benchmark(self):
        """
        # run the non optimized code
        """
        # symmetrical
        mat = core.calculate_scalar_product_matrix(core.dot_product_l2,
                                                   self.initial_functions1, self.initial_functions1,
                                                   optimize=self.optimization)
        # print(mat)
        # print()

        # rect1
        mat = core.calculate_scalar_product_matrix(core.dot_product_l2,
                                                   self.initial_functions2, self.initial_functions1,
                                                   optimize=self.optimization)
        # print(mat)
        # print()

        # rect2
        mat = core.calculate_scalar_product_matrix(core.dot_product_l2,
                                                   self.initial_functions1, self.initial_functions2,
                                                   optimize=self.optimization)
        # print(mat)
        # print()

    @unittest.skip
    def test_optimized(self):
        # run the non optimized code
        self.optimization = True
        self.run_benchmark()

    def test_unoptimized(self):
        # run the non optimized code
        self.optimization = False
        self.run_benchmark()


class ProjectionTest(unittest.TestCase):
    def setUp(self):
        interval = (0, 10)
        node_cnt = 11
        self.nodes, self.initial_functions = shapefunctions.cure_interval(shapefunctions.LagrangeFirstOrder, interval,
                                                                          node_count=node_cnt)
        register_base("ini_funcs", self.initial_functions, overwrite=True)

        # "real" functions
        self.z_values = np.linspace(interval[0], interval[1], 100 * node_cnt)  # because we are smarter
        self.funcs = [core.Function(lambda x: 2),
                      core.Function(lambda x: 2 * x),
                      core.Function(lambda x: x ** 2),
                      core.Function(lambda x: np.sin(x))
                      ]
        self.funcs[1](10)
        self.real_values = [func(self.z_values) for func in self.funcs]

    def test_types_projection(self):
        self.assertRaises(TypeError, core.project_on_base, 1, 2)
        self.assertRaises(TypeError, core.project_on_base, np.sin, np.sin)

    def test_projection_on_lag1st(self):
        weights = []

        # convenience wrapper for non array input -> constant function
        weight = core.project_on_base(self.funcs[0], self.initial_functions[1])
        self.assertTrue(np.allclose(weight, 1.5 * self.funcs[0](self.nodes[1])))

        # linear function -> should be fitted exactly
        weights.append(core.project_on_base(self.funcs[1], self.initial_functions))
        self.assertTrue(np.allclose(weights[-1], self.funcs[1](self.nodes)))

        # quadratic function -> should be fitted somehow close
        weights.append(core.project_on_base(self.funcs[2], self.initial_functions))
        self.assertTrue(np.allclose(weights[-1], self.funcs[2](self.nodes), atol=.5))

        # trig function -> will be crappy
        weights.append(core.project_on_base(self.funcs[3], self.initial_functions))

        if show_plots:
            # since test function are lagrange1st order, plotting the results is fairly easy
            for idx, w in enumerate(weights):
                pw = pg.plot(title="Weights {0}".format(idx))
                pw.plot(x=self.z_values, y=self.real_values[idx + 1], pen="r")
                pw.plot(x=self.nodes, y=w, pen="b")
                app.exec_()

    def test_types_back_projection(self):
        self.assertRaises(TypeError, core.back_project_from_base, 1, 2)
        self.assertRaises(TypeError, core.back_project_from_base, 1.0, np.sin)

    def test_back_projection_from_lagrange_1st(self):
        vec_real_func = np.vectorize(self.funcs[1])
        real_weights = vec_real_func(self.nodes)
        approx_func = core.back_project_from_base(real_weights, self.initial_functions)
        approx_func_dz = core.back_project_from_base(real_weights, get_base("ini_funcs", 1))
        self.assertTrue(np.allclose(approx_func(self.z_values), vec_real_func(self.z_values)))

        if show_plots:
            # lines should match exactly
            pw = pg.plot(title="back projected linear function")
            pw.plot(x=self.z_values, y=vec_real_func(self.z_values), pen="r")
            pw.plot(x=self.z_values, y=approx_func(self.z_values), pen="g")
            pw.plot(x=self.z_values, y=approx_func_dz(self.z_values), pen="b")
            app.exec_()

    def tearDown(self):
        pass


class ChangeProjectionBaseTest(unittest.TestCase):
    def setUp(self):
        # real function
        self.z_values = np.linspace(0, 1, 1000)
        self.real_func = core.Function(lambda x: x)
        self.real_func_handle = np.vectorize(self.real_func)

        # approximation by lag1st
        self.nodes, self.src_test_funcs = shapefunctions.cure_interval(shapefunctions.LagrangeFirstOrder, (0, 1),
                                                                       node_count=2)
        register_base("test_funcs", self.src_test_funcs, overwrite=True)
        self.src_weights = core.project_on_base(self.real_func, self.src_test_funcs)
        self.assertTrue(np.allclose(self.src_weights, [0, 1]))  # just to be sure
        self.src_approx_handle = core.back_project_from_base(self.src_weights, self.src_test_funcs)

        # approximation by sin(w*x)
        def trig_factory(freq):
            def func(x):
                return np.sin(freq * x)

            return func

        self.trig_test_funcs = np.array([core.Function(trig_factory(w), domain=(0, 1)) for w in range(1, 3)])

    def test_types_change_projection_base(self):
        self.assertRaises(TypeError, core.change_projection_base, 1, np.sin, np.cos)

    def test_lag1st_to_trig(self):
        # scalar case
        dest_weight = core.change_projection_base(self.src_weights, self.src_test_funcs, self.trig_test_funcs[0])
        dest_approx_handle_s = core.back_project_from_base(dest_weight, self.trig_test_funcs[0])

        # standard case
        dest_weights = core.change_projection_base(self.src_weights, self.src_test_funcs, self.trig_test_funcs)
        dest_approx_handle = core.back_project_from_base(dest_weights, self.trig_test_funcs)
        error = np.sum(
            np.power(np.subtract(self.real_func_handle(self.z_values), dest_approx_handle(self.z_values)), 2))

        if show_plots:
            pw = pg.plot(title="change projection base")
            i1 = pw.plot(x=self.z_values, y=self.real_func_handle(self.z_values), pen="r")
            i2 = pw.plot(x=self.z_values, y=self.src_approx_handle(self.z_values),
                         pen=pg.mkPen("g", style=pg.QtCore.Qt.DashLine))
            i3 = pw.plot(x=self.z_values, y=dest_approx_handle_s(self.z_values), pen="b")
            i4 = pw.plot(x=self.z_values, y=dest_approx_handle(self.z_values), pen="c")
            legend = pw.addLegend()
            legend.addItem(i1, "f(x) = x")
            legend.addItem(i2, "2x Lagrange1st")
            legend.addItem(i3, "sin(x)")
            legend.addItem(i4, "sin(wx) with w in [1, {0}]".format(dest_weights.shape[0]))
            app.exec_()

        # should fit pretty nice
        self.assertLess(error, 1e-2)

    def tearDown(self):
        pass


class NormalizeFunctionsTestCase(unittest.TestCase):
    def setUp(self):
        self.f = core.Function(np.sin, domain=(0, np.pi * 2))
        self.g = core.Function(np.cos, domain=(0, np.pi * 2))
        self.l = core.Function(np.log, domain=(0, np.exp(1)))

        self.base_f = np.array([self.f])
        self.base_g = np.array([self.g])
        self.base_l = np.array([self.l])

    def test_self_scale(self):
        f = core.normalize_base(self.base_f)
        prod = core.dot_product_l2(f, f)[0]
        self.assertAlmostEqual(prod, 1)

    def test_scale(self):
        f, l = core.normalize_base(self.base_f, self.base_l)
        prod = core.dot_product_l2(f, l)[0]
        self.assertAlmostEqual(prod, 1)

    def test_culprits(self):
        # not possible
        self.assertRaises(ValueError, core.normalize_base, self.base_g, self.base_l)

        # orthogonal
        self.assertRaises(ValueError, core.normalize_base, self.base_f, self.base_g)
