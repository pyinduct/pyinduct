import collections
import unittest
from numbers import Number

import numpy as np
import pyinduct as pi
import pyinduct.core as core
from pyinduct.tests import show_plots
import pyqtgraph as pg



class SanitizeInputTestCase(unittest.TestCase):
    def test_scalar(self):
        self.assertRaises(TypeError, core.sanitize_input, 1.0, int)
        pi.sanitize_input(1, int)
        pi.sanitize_input(1.0, float)


class BaseFractionTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_init(self):
        f = pi.BaseFraction(np.sin)
        self.assertEqual(f.members, np.sin)

    def test_derive(self):
        f = pi.BaseFraction(np.sin)
        self.assertEqual(f.members, np.sin)

        f_d0 = f.derive(0)
        self.assertEqual(f, f_d0)


class FunctionTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_init(self):
        self.assertRaises(TypeError, pi.Function, 42)
        p = pi.Function(np.sin)

        # default kwargs
        self.assertEqual(p.domain, [(-np.inf, np.inf)])
        self.assertEqual(p.nonzero, [(-np.inf, np.inf)])

        for kwarg in ["domain", "nonzero"]:
            # some nice but wrong variants
            for val in ["4-2", dict(start=1, stop=2), [1, 2]]:
                self.assertRaises(TypeError, pi.Function, np.sin, **{kwarg: val})

            # a correct one
            pi.Function(np.sin, **{kwarg: (0, 10)})
            pi.Function(np.sin, **{kwarg: [(0, 3), (5, 10)]})

            # check sorting
            p = pi.Function(np.sin, **{kwarg: (0, -10)})
            self.assertEqual(getattr(p, kwarg), [(-10, 0)])
            p = pi.Function(np.sin, **{kwarg: [(5, 0), (-10, -5)]})
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

        self.assertRaises(TypeError, pi.Function, wrong_handle)

    def test_derivation(self):
        f = pi.Function(np.sin, derivative_handles=[np.cos, np.sin])

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

        f_2 = pi.Function(np.sin, derivative_handles=[np.cos, np.sin])
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
        f = pi.Function(np.sin, derivative_handles=[np.cos, np.sin])

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
        f = pi.Function(np.sin, derivative_handles=[np.cos, np.sin])

        # no new object since trivial scaling occurred
        g1 = f.raise_to(1)
        self.assertEqual(f, g1)

        # after scaling, return scalars and vectors like normal
        g2 = f.raise_to(2)

        self.assertIsInstance(g2(5), Number)
        self.assertNotIsInstance(g2(5), np.ndarray)
        self.assertTrue(np.array_equal(np.sin(np.array(range(100))) ** 2, g2(np.array(range(100)))))
        self.assertRaises(ValueError, g2.derive, 1)  # derivatives should be removed when scaled by function

    def test_call(self):

        def func(x):
            if isinstance(x, collections.Iterable):
                raise TypeError("no vectorial stuff allowed!")
            return 2 ** x

        f = pi.Function(func, domain=(0, 10))
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

        f = pi.Function(vector_func, domain=(0, 10))
        self.assertEqual(f._vectorial, True)  # function handle should be recognized as vectorial

        # call with scalar should return scalar with correct value
        self.assertIsInstance(f(10), Number)
        self.assertNotIsInstance(f(10), np.ndarray)
        self.assertEqual(f(10), vector_func(10))

        # vectorial arguments should be understood and an np.ndarray shall be returned
        self.assertIsInstance(f(np.array(range(10))), np.ndarray)
        self.assertTrue(np.array_equal(f(np.array(range(10))), [vector_func(val) for val in range(10)]))


class BaseTestCase(unittest.TestCase):
    def setUp(self):
        self.fractions = [pi.Function(lambda x: 2),
                          pi.Function(lambda x: 2 * x),
                          pi.Function(lambda x: x ** 2),
                          pi.Function(lambda x: np.sin(x))
                          ]

    def test_init(self):
        # single and iterable arguments should be taken
        b1 = pi.Base(self.fractions[0])
        b2 = pi.Base(self.fractions)

    def test_scale(self):
        f = pi.Base([pi.Function(np.sin,
                                 derivative_handles=[np.cos, np.sin])
                     for i in range(5)])

        # no new fractions should be created since trivial scaling occurred
        g1 = f.scale(1)
        np.testing.assert_array_equal(f.fractions, g1.fractions)

        # all fractions should be scaled the same way
        factor = 10
        values = np.linspace(1, 1e2)
        g2 = f.scale(factor)
        for a, b in zip(f.fractions, g2.fractions):
            np.testing.assert_array_equal(10 * a(values), b(values))

    def test_transformation_hint(self):
        f = pi.Base([pi.Function(np.sin, domain=(0, np.pi)),
                     pi.Function(np.cos, domain=(0, np.pi))])

        info = core.TransformationInfo()
        info.src_lbl = "me"
        info.dst_lbl = "me again"
        info.src_base = f
        info.dst_base = f
        info.src_order = 1
        info.dst_order = 1

        # test defaults
        func, extra = f.transformation_hint(info)
        weights = np.array(range(4))
        t_weights = func(weights)
        np.testing.assert_array_almost_equal(weights, t_weights)
        self.assertIsNone(extra)

        # no transformation hint known
        info.dst_base = pi.StackedBase(self.fractions, None)
        func, extra = f.transformation_hint(info)
        self.assertIsNone(func)
        self.assertIsNone(extra)

    def test_scalar_product_hint(self):
        f = pi.Base(self.fractions)

        # test defaults
        self.assertEqual(f.scalar_product_hint(),
                         self.fractions[0].scalar_product_hint())

    def test_raise_to(self):
        sin_func = pi.Function(np.sin, derivative_handles=[np.cos])
        cos_func = pi.Function(np.cos,
                               derivative_handles=[lambda z: -np.sin(z)])
        f = pi.Base([sin_func, cos_func])
        self.assertEqual(f.fractions[0], sin_func)
        self.assertEqual(f.fractions[1], cos_func)

        numbers = np.random.random_sample((100, ))

        # power 1
        np.testing.assert_array_equal(f.raise_to(1).fractions[0](numbers),
                                      sin_func.raise_to(1)(numbers))
        np.testing.assert_array_equal(f.raise_to(1).fractions[1](numbers),
                                      cos_func.raise_to(1)(numbers))

        # power 4
        np.testing.assert_array_equal(f.raise_to(4).fractions[0](numbers),
                                      sin_func.raise_to(4)(numbers))
        np.testing.assert_array_equal(f.raise_to(4).fractions[1](numbers),
                                      cos_func.raise_to(4)(numbers))

    def test_derive(self):
        sin_func = pi.Function(np.sin, derivative_handles=[np.cos])
        cos_func = pi.Function(np.cos,
                               derivative_handles=[lambda z: -np.sin(z)])
        f = pi.Base([sin_func, cos_func])
        self.assertEqual(f.fractions[0], sin_func)
        self.assertEqual(f.fractions[1], cos_func)

        numbers = np.random.random_sample((100, ))

        # derivative 0
        np.testing.assert_array_equal(f.derive(0).fractions[0](numbers),
                                      sin_func.derive(0)(numbers))
        np.testing.assert_array_equal(f.derive(0).fractions[1](numbers),
                                      cos_func.derive(0)(numbers))

        # derivative 1
        np.testing.assert_array_equal(f.derive(1).fractions[0](numbers),
                                      sin_func.derive(1)(numbers))
        np.testing.assert_array_equal(f.derive(1).fractions[1](numbers),
                                      cos_func.derive(1)(numbers))


class StackedBaseTestCase(unittest.TestCase):
    def setUp(self):
        self.b1 = pi.Base([pi.Function(lambda x: np.sin(2)),
                           pi.Function(lambda x: np.sin(2*x)),
                           pi.Function(lambda x: np.sin(2 ** 2 * x)),
                           ])
        pi.register_base("b1", self.b1)
        self.b2 = pi.Base([pi.Function(lambda x: np.cos(4)),
                           pi.Function(lambda x: np.cos(4 * x)),
                           pi.Function(lambda x: np.cos(4 ** 2 * x)),
                           ])
        pi.register_base("b2", self.b2)

    @unittest.skip  # WIP
    def test_init(self):
        fractions = np.hstack([self.b1, self.b2])
        info = None
        b = pi.StackedBase(fractions, info)
        self.assertEqual(b.fractions.size, 6)


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
        self.f1 = pi.Function(lambda x: 1, domain=(0, 10))
        self.f2 = pi.Function(lambda x: 2, domain=(0, 5))
        self.f3 = pi.Function(lambda x: 2, domain=(0, 5), nonzero=(2, 3))
        self.f4 = pi.Function(lambda x: 2, domain=(0, 5), nonzero=(2, 2 + 1e-1))

        self.f5 = pi.LagrangeFirstOrder(0, 1, 2)
        self.f6 = pi.LagrangeFirstOrder(1, 2, 3)
        self.f7 = pi.LagrangeFirstOrder(2, 3, 4)

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
        self.nodes1, self.initial_functions1 = pi.cure_interval(pi.LagrangeFirstOrder, interval, node_count=nodes)
        self.nodes2, self.initial_functions2 = pi.cure_interval(pi.LagrangeFirstOrder, interval, node_count=2*nodes-1)
        self.optimization = None
        # print(np.array(self.nodes1), np.array(self.nodes2))

    def run_benchmark(self):
        """
        # run the non optimized code
        """
        # symmetrical
        mat = pi.calculate_scalar_product_matrix(pi.dot_product_l2,
                                                 self.initial_functions1, self.initial_functions1,
                                                 optimize=self.optimization)
        # print(mat)
        # print()

        # rect1
        mat = pi.calculate_scalar_product_matrix(pi.dot_product_l2,
                                                 self.initial_functions2, self.initial_functions1,
                                                 optimize=self.optimization)
        # print(mat)
        # print()

        # rect2
        mat = pi.calculate_scalar_product_matrix(pi.dot_product_l2,
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
        self.nodes, self.lag_base = pi.cure_interval(pi.LagrangeFirstOrder, interval, node_count=node_cnt)
        pi.register_base("lag_base", self.lag_base, overwrite=True)

        # "real" functions
        self.z_values = np.linspace(interval[0], interval[1], 100 * node_cnt)  # because we are smarter
        self.functions = [pi.Function(lambda x: 2),
                          pi.Function(lambda x: 2 * x),
                          pi.Function(lambda x: x ** 2),
                          pi.Function(lambda x: np.sin(x))
                          ]
        self.real_values = [func(self.z_values) for func in self.functions]

    def test_types_projection(self):
        self.assertRaises(TypeError, pi.project_on_base, 1, 2)
        self.assertRaises(TypeError, pi.project_on_base, np.sin, np.sin)

    def test_projection_on_lag1st(self):
        weights = [pi.project_on_base(self.functions[1], self.lag_base),
                   pi.project_on_base(self.functions[2], self.lag_base),
                   pi.project_on_base(self.functions[3], self.lag_base)]

        # linear function -> should be fitted exactly
        np.testing.assert_array_almost_equal(weights[0], self.functions[1](self.nodes))

        # quadratic function -> should be fitted somehow close
        np.testing.assert_array_almost_equal(weights[1], self.functions[2](self.nodes), decimal=0)

        # trig function -> will be crappy
        np.testing.assert_array_almost_equal(weights[2], self.functions[3](self.nodes), decimal=1)

        if show_plots:
            # since test function are lagrange1st order, plotting the results is fairly easy
            for idx, w in enumerate(weights):
                pw = pg.plot(title="Weights {0}".format(idx))
                pw.plot(x=self.z_values, y=self.real_values[idx + 1], pen="r")
                pw.plot(x=self.nodes.points, y=w, pen="b")
                pi.show(show_mpl=False)

    def test_back_projection_from_lagrange_1st(self):
        vec_real_func = np.vectorize(self.functions[1])
        real_weights = vec_real_func(self.nodes)
        approx_func = pi.back_project_from_base(real_weights, self.lag_base)
        approx_func_dz = pi.back_project_from_base(real_weights, pi.get_base("lag_base").derive(1))
        np.testing.assert_array_almost_equal(approx_func(self.z_values), vec_real_func(self.z_values))

        if show_plots:
            # lines should match exactly
            pw = pg.plot(title="back projected linear function")
            pw.plot(x=self.z_values, y=vec_real_func(self.z_values), pen="r")
            pw.plot(x=self.z_values, y=approx_func(self.z_values), pen="g")
            pw.plot(x=self.z_values, y=approx_func_dz(self.z_values), pen="b")
            pi.show(show_mpl=False)

    def tearDown(self):
        pi.deregister_base("lag_base")


class ChangeProjectionBaseTest(unittest.TestCase):
    def setUp(self):
        # real function
        self.z_values = np.linspace(0, 1, 1000)
        self.real_func = pi.Function(lambda x: x)
        self.real_func_handle = np.vectorize(self.real_func)

        # approximation by lag1st
        self.nodes, self.lag_base = pi.cure_interval(pi.LagrangeFirstOrder, (0, 1), node_count=2)
        pi.register_base("lag_base", self.lag_base)
        self.src_weights = pi.project_on_base(self.real_func, self.lag_base)
        np.testing.assert_array_almost_equal(self.src_weights, [0, 1])  # just to be sure
        self.src_approx_handle = pi.back_project_from_base(self.src_weights, self.lag_base)

        # approximation by sin(w*x)
        def trig_factory(freq):
            def func(x):
                return np.sin(freq * x)

            return func

        self.trig_base = pi.Base([pi.Function(trig_factory(w), domain=(0, 1)) for w in range(1, 3)])

    def test_types_change_projection_base(self):
        with self.assertRaises(TypeError):
            pi.change_projection_base(1, pi.Base(np.sin), pi.Base(np.cos))

    def test_lag1st_to_trig(self):
        destination_weights = pi.change_projection_base(self.src_weights, self.lag_base, self.trig_base)
        destination_approx_handle = pi.back_project_from_base(destination_weights, self.trig_base)
        error = np.sum(np.power(
            np.subtract(self.real_func_handle(self.z_values), destination_approx_handle(self.z_values)),
            2))

        if show_plots:
            pw = pg.plot(title="change projection base")
            i1 = pw.plot(x=self.z_values, y=self.real_func_handle(self.z_values), pen="r")
            i2 = pw.plot(x=self.z_values, y=self.src_approx_handle(self.z_values),
                         pen=pg.mkPen("g", style=pg.QtCore.Qt.DashLine))
            i3 = pw.plot(x=self.z_values, y=destination_approx_handle(self.z_values), pen="c")
            legend = pw.addLegend()
            legend.addItem(i1, "f(x) = x")
            legend.addItem(i2, "2x Lagrange1st")
            legend.addItem(i3, "sin(wx) with w in [1, {0}]".format(destination_weights.shape[0]))
            pi.show(show_mpl=False)

        # should fit pretty nice
        self.assertLess(error, 1e-2)

    def tearDown(self):
        pi.deregister_base("lag_base")


class NormalizeFunctionsTestCase(unittest.TestCase):
    def setUp(self):
        self.f = pi.Function(np.sin, domain=(0, np.pi * 2))
        self.g = pi.Function(np.cos, domain=(0, np.pi * 2))
        self.l = pi.Function(np.log, domain=(0, np.exp(1)))

        self.base_f = pi.Base(self.f)
        self.base_g = pi.Base(self.g)
        self.base_l = pi.Base(self.l)

    def test_self_scale(self):
        f = pi.normalize_base(self.base_f)
        prod = pi.dot_product_l2(f.fractions, f.fractions)[0]
        self.assertAlmostEqual(prod, 1)

    def test_scale(self):
        f, l = pi.normalize_base(self.base_f, self.base_l)
        prod = pi.dot_product_l2(f.fractions, l.fractions)[0]
        self.assertAlmostEqual(prod, 1)

    def test_culprits(self):
        # not possible
        self.assertRaises(ValueError, pi.normalize_base, self.base_g, self.base_l)

        # orthogonal
        self.assertRaises(ValueError, pi.normalize_base, self.base_f, self.base_g)


class FindRootsTestCase(unittest.TestCase):
    def setUp(self):
        def _frequent_equation(omega):
            return np.cos(10 * omega)

        def _char_equation(omega):
            return omega * (np.sin(omega) + omega * np.cos(omega))

        def _univariate_equation(x):
            return [np.cos(x[0]), np.cos(4 * x[1])]

        def _complex_equation(lamda):
            """
            Five roots on the unit circle.
            """
            if lamda == 0:
                return 0
            return lamda**5 - 1

        self.frequent_eq = _frequent_equation
        self.char_eq = _char_equation
        self.univariate_eq = _univariate_equation
        self.complex_eq = _complex_equation

        self.n_roots = 10
        self.small_grid = np.arange(0, 1, 1)
        self.grid = np.arange(0, 50, 1)
        self.rtol = .1

    def test_all_roots(self):
        grid = np.linspace(np.pi/20, 3*np.pi/2, num=20)
        roots = pi.find_roots(function=self.frequent_eq, grid=grid,
                              n_roots=self.n_roots, rtol=self.rtol/100)

        # if show_plots:
        #     pi.visualize_roots(roots,
        #                        [np.linspace(np.pi/20, 3*np.pi/2, num=1000)],
        #                        self.frequent_eq)

        real_roots = [(2*k - 1)*np.pi/2/10 for k in range(1, self.n_roots+1)]
        np.testing.assert_array_almost_equal(roots, real_roots)

    def test_in_fact_roots(self):
        roots = pi.find_roots(function=self.char_eq, grid=self.grid,
                              n_roots=self.n_roots, rtol=self.rtol)
        # pi.visualize_roots(roots, self.grid, self.char_eq)

        for root in roots:
            self.assertAlmostEqual(self.char_eq(root), 0)

    def test_enough_roots(self):
        # small area -> not enough roots -> Exception
        with self.assertRaises(ValueError):
            pi.find_roots(self.char_eq, self.small_grid, self.n_roots,
                          self.rtol)

        # bigger area, check good amount
        roots = pi.find_roots(self.char_eq, self.grid, self.n_roots, self.rtol)
        self.assertEqual(len(roots), self.n_roots)

    def test_rtol(self):
        roots = pi.find_roots(self.char_eq, self.grid, self.n_roots, self.rtol)
        self.assertGreaterEqual(np.log10(
            min(np.abs(np.diff(roots)))), self.rtol)

    def test_in_area(self):
        roots = pi.find_roots(self.char_eq, self.grid, self.n_roots, self.rtol)
        for root in roots:
            self.assertTrue(root >= 0.)

    def test_complex_func(self):
        grid = [np.linspace(-2, 2), np.linspace(-2, 2)]
        roots = pi.find_roots(function=self.complex_eq, grid=grid, n_roots=5,
                              rtol=self.rtol, cmplx=True)
        np.testing.assert_array_almost_equal(
            [self.complex_eq(root) for root in roots],
            [0] * len(roots))

        # pi.visualize_roots(roots,
        #                    grid,
        #                    self.complex_eq,
        #                    cmplx=True)

    def test_n_dim_func(self):
        grid = [np.linspace(0, 10),
                np.linspace(0, 2)]
        roots = pi.find_roots(function=self.univariate_eq, grid=grid, n_roots=6,
                              rtol=self.rtol)
        grid = [np.arange(0, 10, .1), np.arange(0, 10, .1)]

        # pi.visualize_roots(roots, grid, self.univariate_eq)

    def tearDown(self):
        pass


class RealTestCase(unittest.TestCase):

    def test_call(self):
        self.assertTrue(np.isreal(pi.real(1)))
        self.assertTrue(np.isreal(pi.real(1 + 0j)))
        self.assertTrue(np.isreal(pi.real(1 + 1e-20j)))
        self.assertRaises(TypeError, pi.real, None)
        # self.assertRaises(TypeError, pi.real, [None, 2., 2 + 2j])
        self.assertRaises(ValueError, pi.real, (1, 2., 2 + 2j))
        self.assertRaises(ValueError, pi.real, [1, 2., 2 + 2j])
        self.assertRaises(ValueError, pi.real, 1 + 1e-10j)
        self.assertRaises(ValueError, pi.real, 1j)


class ParamsTestCase(unittest.TestCase):
    def test_init(self):
        p = pi.Parameters(a=10, b=12, c="high")
        self.assertTrue(p.a == 10)
        self.assertTrue(p.b == 12)
        self.assertTrue(p.c == "high")
