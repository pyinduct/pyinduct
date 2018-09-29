import collections
import time
import unittest
import warnings

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
        self.assertEqual(p.domain, {(-np.inf, np.inf)})
        self.assertEqual(p.nonzero, {(-np.inf, np.inf)})

        for kwarg in ["domain", "nonzero"]:
            # some nice but wrong variants
            for val in ["4-2", dict(start=1, stop=2), [1, 2]]:
                self.assertRaises(TypeError,
                                  pi.Function,
                                  np.sin,
                                  **{kwarg: val})

            # a correct one
            pi.Function(np.sin, **{kwarg: (0, 10)})
            pi.Function(np.sin, **{kwarg: {(0, 3), (5, 10)}})

            # check sorting
            p = pi.Function(np.sin, **{kwarg: (0, -10)})
            self.assertEqual(getattr(p, kwarg), {(-10, 0)})
            p = pi.Function(np.sin, **{kwarg: {(5, 0), (-10, -5)}})
            self.assertEqual(getattr(p, kwarg), {(-10, -5), (0, 5)})

            # check simplification
            p = pi.Function(np.sin, **{kwarg: {(0, 5), (3, 10)}})
            self.assertEqual(getattr(p, kwarg), {(0, 10)})

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
        self.assertTrue(np.array_equal(10 * np.sin(list(range(100))),
                                       g2(list(range(100)))))

        # scale with function
        g3 = f.scale(lambda z: z)

        def check_handle(z):
            return z * f(z)

        self.assertIsInstance(g3(5), Number)
        self.assertNotIsInstance(g3(5), np.ndarray)
        self.assertTrue(np.array_equal(g3(list(range(10))),
                                       check_handle(list(range(10)))))

        # derivatives should be removed when scaled by function
        self.assertRaises(ValueError, g3.derive, 1)

    def test_raise(self):
        f = pi.Function(np.sin, derivative_handles=[np.cos, np.sin])

        # no new object since trivial scaling occurred
        g1 = f.raise_to(1)
        self.assertEqual(f, g1)

        # after scaling, return scalars and vectors like normal
        g2 = f.raise_to(2)

        self.assertIsInstance(g2(5), Number)
        self.assertNotIsInstance(g2(5), np.ndarray)
        self.assertTrue(np.array_equal(np.sin(np.array(range(100))) ** 2,
                                       g2(np.array(range(100)))))
        # derivatives should be removed when scaled by function
        self.assertRaises(ValueError, g2.derive, 1)

    def test_call(self):

        def func(x):
            if isinstance(x, collections.Iterable):
                raise TypeError("no vectorial stuff allowed!")
            return 2 ** x

        f = pi.Function(func, domain=(0, 10))
        # function handle should be recognized as non-vectorial
        self.assertEqual(f._vectorial, False)

        # domain check must be checked
        with self.assertRaises(ValueError):
            f(-4)
        with self.assertRaises(ValueError):
            f(11)

        # call with scalar should return scalar with correct value
        self.assertIsInstance(f(10), Number)
        self.assertNotIsInstance(f(10), np.ndarray)
        self.assertEqual(f(10), func(10))

        # vectorial arguments should be understood and an np.ndarray shall be
        #  returned
        self.assertIsInstance(f(np.array(range(10))), np.ndarray)
        self.assertTrue(np.array_equal(f(np.array(range(10))),
                                       [func(val) for val in range(10)]))

    def test_vector_call(self):

        def vector_func(x):
            return 2 * x

        f = pi.Function(vector_func, domain=(0, 10))
        # function handle should be recognized as vectorial
        self.assertEqual(f._vectorial, True)

        # call with scalar should return scalar with correct value
        self.assertIsInstance(f(10), Number)
        self.assertNotIsInstance(f(10), np.ndarray)
        self.assertEqual(f(10), vector_func(10))

        # vectorial arguments should be understood and an np.ndarray shall be
        #  returned
        self.assertIsInstance(f(np.array(range(10))), np.ndarray)
        self.assertTrue(np.array_equal(f(np.array(range(10))),
                                       [vector_func(val) for val in range(10)]))


class BaseTestCase(unittest.TestCase):
    def setUp(self):
        self.fractions = [pi.Function(lambda x: 2),
                          pi.Function(lambda x: 2 * x),
                          pi.Function(lambda x: x ** 2),
                          pi.Function(lambda x: np.sin(x))
                          ]
        self.other_fractions = [pi.ComposedFunctionVector(frac, 7)
                                for frac in self.fractions]
        self.completely_other_fractions = [
            pi.ComposedFunctionVector(self.fractions, i)
            for i in range(10)]

    def test_init(self):
        # single and iterable arguments should be taken
        b1 = pi.Base(self.fractions[0])
        b2 = pi.Base(self.fractions)
        b3 = pi.Base(self.other_fractions)
        b4 = pi.Base(self.completely_other_fractions)

        # the provided scalar product hints should be compatible
        with self.assertRaises(ValueError):
            pi.Base([self.fractions[0], self.other_fractions[2]])

        with self.assertRaises(ValueError):
            pi.Base([self.other_fractions[0],
                     self.completely_other_fractions[2]])

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

    def test_get_attribute(self):
        sin_func = pi.Function(np.sin, derivative_handles=[np.cos])
        cos_func = pi.Function(np.cos,
                               derivative_handles=[lambda z: -np.sin(z)])
        f = pi.Base([sin_func, cos_func])

        domains = f.get_attribute("domain")
        self.assertIsInstance(domains, np.ndarray)
        self.assertEqual(len(domains), 2)

        # query something that is not there
        res = f.get_attribute("Answer to the ultimate question of life, "
                              "The universe, and Everything")
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(len(res), 2)
        self.assertIsNone(res[0])
        self.assertIsNone(res[1])


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

    @unittest.skip  # TODO complete this
    def test_init(self):
        fractions = np.hstack([self.b1, self.b2])
        info = None
        b = pi.StackedBase(fractions, info)
        self.assertEqual(b.fractions.size, 6)


class TransformationTestCase(unittest.TestCase):

    def setUp(self):
        dom1 = pi.Domain((0, 1), num=11)
        dom2 = pi.Domain((0, 1), num=21)
        self.base1 = pi.LagrangeFirstOrder.cure_interval(dom1)
        pi.register_base("fem1", self.base1)
        self.base2 = pi.LagrangeSecondOrder.cure_interval(dom2)
        pi.register_base("fem2", self.base2)

    def test_transformation_info(self):
        info = core.get_transformation_info("fem1", "fem2", 1, 7)
        self.assertEqual(info.src_lbl, "fem1")
        self.assertEqual(info.src_base, self.base1)
        self.assertEqual(info.src_order, 1)
        self.assertEqual(info.dst_lbl, "fem2")
        self.assertEqual(info.dst_base, self.base2)
        self.assertEqual(info.dst_order, 7)

    def test_get_trafo_simple(self):
        """ Transformation between to standard bases"""
        info = core.get_transformation_info("fem1", "fem2", 0, 0)
        trafo = core.get_weight_transformation(info)
        trafo_mat = trafo(np.eye(11))
        src_weights = np.random.rand(11)
        dst_weights = trafo(src_weights)
        self.assertEqual(dst_weights.shape, (21,))

        # now with different orders
        info = core.get_transformation_info("fem1", "fem2", 1, 1)
        trafo = core.get_weight_transformation(info)
        src_weights = np.random.rand(22)
        dst_weights = trafo(src_weights)
        self.assertEqual(dst_weights.shape, (42,))

    def tearDown(self):
        pi.deregister_base("fem1")
        pi.deregister_base("fem2")


class SimplificationTestCase(unittest.TestCase):

    def test_easy_simplifications(self):
        self.assertEqual(core.domain_simplification({(0, 2), (1, 3)}),
                         {(0, 3)})
        self.assertEqual(core.domain_simplification({(0, 2), (3, 1)}),
                         {(0, 3)})
        self.assertEqual(core.domain_simplification({(3, 1), (0, 2)}),
                         {(0, 3)})
        self.assertEqual(core.domain_simplification({(3, 1), (2, 0)}),
                         {(0, 3)})

    def test_sophisticated_simplifications(self):
        self.assertEqual(core.domain_simplification({(0, 2), (1, 3), (1, 5)}),
                         {(0, 5)})
        self.assertEqual(core.domain_simplification({(0, 2), (1, 5), (4, 6)}),
                         {(0, 6)})


class IntersectionTestCase(unittest.TestCase):
    def test_wrong_arguments(self):
        # interval bounds not sorted
        self.assertRaises(ValueError, core.domain_intersection, (3, 2), (1, 3))
        # intervals not sorted
        self.assertRaises(ValueError, core.domain_intersection, [(4, 5), (1, 2)], (1, 3))
        # intervals useless
        self.assertRaises(ValueError, core.domain_intersection, [(4, 5), (5, 6)], (1, 3))

    def test_easy_intersections(self):
        self.assertEqual(core.domain_intersection((0, 2), (1, 3)), {(1, 2)})
        self.assertEqual(core.domain_intersection((0, 1), (1, 3)), set())
        self.assertEqual(core.domain_intersection((3, 5), (1, 3)), set())
        self.assertEqual(core.domain_intersection((3, 5), (1, 4)), {(3, 4)})
        self.assertEqual(core.domain_intersection((3, 5), (1, 6)), {(3, 5)})
        self.assertEqual(core.domain_intersection((3, 5), (6, 7)), set())

    def test_complex_intersections(self):
        self.assertEqual(core.domain_intersection([(0, 2), (3, 5)], (3, 4)),
                         {(3, 4)})
        self.assertEqual(core.domain_intersection([(0, 2), (3, 5)], (1, 4)),
                         {(1, 2), (3, 4)})
        self.assertEqual(core.domain_intersection((1, 4), [(0, 2), (3, 5)]),
                         {(1, 2), (3, 4)})
        self.assertEqual(core.domain_intersection([(1, 3), (4, 6)],
                                                  [(0, 2), (3, 5)]),
                         {(1, 2), (4, 5)})
        self.assertEqual(core.domain_intersection([(-10, -4), (2, 5), (10, 17)],
                                                  [(-20, -5), (3, 5), (7, 23)]),
                         {(-10, -5), (3, 5)}, (10, 17))


class IntegrateFunctionTestCase(unittest.TestCase):
    def setUp(self):
        self.int1 = [(0, 10)]
        self.int2 = [(0, 10), (20, 30)]
        self.int3 = [(0, 20), (20, 30)]
        self.int4 = [(0, 20), (20, 30)]

        self.func1 = lambda x: 2*x
        self.func2 = lambda z: 3*z**2 + 2j*z
        self.func2_int = lambda z: z**3 + 1j*z**2

    def test_real_integration(self):
        # real integrals
        res, err = core.integrate_function(self.func1, self.int1)
        self.assertFalse(np.iscomplexobj(res))
        np.testing.assert_almost_equal(res, 100)

        res, err = core.integrate_function(self.func1, self.int2)
        self.assertFalse(np.iscomplexobj(res))
        np.testing.assert_almost_equal(res, 10**2-0**2 + 30**2-20**2)

        # multiple regions
        res3, err = core.integrate_function(self.func1, self.int3)
        res4, err = core.integrate_function(self.func1, self.int4)
        self.assertFalse(np.iscomplexobj(res3))
        self.assertFalse(np.iscomplexobj(res4))
        np.testing.assert_almost_equal(res3, 30**2)
        np.testing.assert_almost_equal(res3, res4)

    def test_complex_integration(self):
        res, err = core.integrate_function(self.func2, self.int1)
        self.assertTrue(np.iscomplexobj(res))
        np.testing.assert_almost_equal(res, self.func2_int(10))


class ScalarDotProductL2TestCase(unittest.TestCase):
    def setUp(self):
        self.f1 = pi.Function(lambda x: 1, domain=(0, 10))
        self.f2 = pi.Function(lambda x: 2, domain=(0, 5))
        self.f3 = pi.Function(lambda x: 2, domain=(0, 5), nonzero=(2, 3))
        self.f4 = pi.Function(lambda x: 2, domain=(0, 5), nonzero=(2, 2 + 1e-1))

        self.f5 = pi.LagrangeFirstOrder(0, 1, 2)
        self.f6 = pi.LagrangeFirstOrder(1, 2, 3)
        self.f7 = pi.LagrangeFirstOrder(2, 3, 4)

        self.g1 = pi.Function(lambda x: 2 + 2j, domain=(0, 5))
        self.g2 = pi.Function(lambda x: 2 - 2j, domain=(0, 5))

    def test_domain(self):
        self.assertAlmostEqual(core._dot_product_l2(self.f1, self.f2), 10)
        # swap arguments
        self.assertAlmostEqual(core._dot_product_l2(self.f2, self.f1),
                               np.conjugate(10))

        self.assertAlmostEqual(core._dot_product_l2(self.f1, self.f3), 2)
        # swap arguments
        self.assertAlmostEqual(core._dot_product_l2(self.f3, self.f1),
                               np.conjugate(2))

    def test_nonzero(self):
        self.assertAlmostEqual(core._dot_product_l2(self.f1, self.f4), 2e-1)
        self.assertAlmostEqual(core._dot_product_l2(self.f4, self.f1),
                               np.conjugate(2e-1))

    def test_lagrange(self):
        self.assertAlmostEqual(core._dot_product_l2(self.f5, self.f7), 0)
        self.assertAlmostEqual(core._dot_product_l2(self.f5, self.f6), 1 / 6)
        self.assertAlmostEqual(core._dot_product_l2(self.f7, self.f6), 1 / 6)
        self.assertAlmostEqual(core._dot_product_l2(self.f5, self.f5), 2 / 3)

    def test_complex(self):
        self.assertAlmostEqual(core._dot_product_l2(self.g1, self.g2), -40j)
        # swapping of args will return the conjugated expression
        self.assertAlmostEqual(core._dot_product_l2(self.g2, self.g1),
                               np.conj(-40j))

    def test_linearity(self):
        factor = 2+1j
        s = self.g1.scale(factor)
        res = core._dot_product_l2(s, self.g2)
        part = core._dot_product_l2(self.g1, self.g2)
        np.testing.assert_almost_equal(np.conjugate(factor)*part, res)


class DotProductL2TestCase(unittest.TestCase):
    def setUp(self):
        self.dom = pi.Domain((0, 10), num=11)
        self.fem_base = pi.LagrangeFirstOrder.cure_interval(self.dom, num=10)

    def test_length(self):
        with self.assertRaises(ValueError):
            pi.dot_product_l2(self.fem_base[2:4], self.fem_base[4:8])

    def test_output(self):
        res = pi.dot_product_l2(self.fem_base.fractions,
                                self.fem_base.fractions)
        np.testing.assert_almost_equal(res, [1/3] + [2/3]*9 + [1/3])


class CalculateScalarProductMatrixTestCase(unittest.TestCase):
    def setUp(self, dim1=10, dim2=20):
        interval = (0, 10)
        self.nodes1 = pi.Domain(interval, num=dim1)
        self.nodes2 = pi.Domain(interval, num=dim2)
        self.initial_functions1 = pi.LagrangeFirstOrder.cure_interval(
            self.nodes1)
        self.initial_functions2 = pi.LagrangeFirstOrder.cure_interval(
            self.nodes2)

        self.optimization = False

    def test_quadratic(self):
        res_quad1 = np.zeros([len(self.initial_functions1)]*2)
        for idx1, frac1 in enumerate(self.initial_functions1):
            for idx2, frac2 in enumerate(self.initial_functions1):
                res_quad1[idx1, idx2] = core._dot_product_l2(frac1, frac2)

        r, t = self.quadratic_case1()
        self.assertFalse(np.iscomplexobj(r))
        self.assertEqual(r.shape, (len(self.initial_functions1),
                                   len(self.initial_functions1)))
        np.testing.assert_almost_equal(r, res_quad1)

        res_quad2 = np.zeros([len(self.initial_functions2)]*2)
        for idx1, frac1 in enumerate(self.initial_functions2):
            for idx2, frac2 in enumerate(self.initial_functions2):
                res_quad2[idx1, idx2] = core._dot_product_l2(frac1, frac2)

        r, t = self.quadratic_case2()
        self.assertFalse(np.iscomplexobj(r))
        self.assertEqual(r.shape, (len(self.initial_functions2),
                                   len(self.initial_functions2)))
        np.testing.assert_almost_equal(r, res_quad2)

    def test_rectangular(self):
        res = np.zeros((len(self.initial_functions1),
                        len(self.initial_functions2)))
        for idx1, frac1 in enumerate(self.initial_functions1):
            for idx2, frac2 in enumerate(self.initial_functions2):
                res[idx1, idx2] = core._dot_product_l2(frac1, frac2)

        res_rect1 = res.copy()
        res_rect2 = np.conjugate(res).T

        r, t = self.rectangular_case_1()
        self.assertFalse(np.iscomplexobj(r))
        self.assertEqual(r.shape, (len(self.initial_functions1),
                                   len(self.initial_functions2)))
        np.testing.assert_almost_equal(r, res_rect1)

        r, t = self.rectangular_case_2()
        self.assertFalse(np.iscomplexobj(r))
        self.assertEqual(r.shape, (len(self.initial_functions2),
                                   len(self.initial_functions1)))
        np.testing.assert_almost_equal(r, res_rect2)

    def quadratic_case1(self):
        # result is quadratic
        t0 = time.clock()
        mat = pi.calculate_scalar_product_matrix(pi.dot_product_l2,
                                                 self.initial_functions1,
                                                 self.initial_functions1,
                                                 optimize=self.optimization)
        t_calc = time.clock() - t0
        return mat, t_calc

    def quadratic_case2(self):
        # result is quadratic
        t0 = time.clock()
        mat = pi.calculate_scalar_product_matrix(pi.dot_product_l2,
                                                 self.initial_functions2,
                                                 self.initial_functions2,
                                                 optimize=self.optimization)
        t_calc = time.clock() - t0
        return mat, t_calc

    def rectangular_case_1(self):
        # rect1
        t0 = time.clock()
        mat = pi.calculate_scalar_product_matrix(pi.dot_product_l2,
                                                 self.initial_functions1,
                                                 self.initial_functions2,
                                                 optimize=self.optimization)
        t_calc = time.clock() - t0
        return mat, t_calc

    def rectangular_case_2(self):
        # rect2
        t0 = time.clock()
        mat = pi.calculate_scalar_product_matrix(pi.dot_product_l2,
                                                 self.initial_functions2,
                                                 self.initial_functions1,
                                                 optimize=self.optimization)
        t_calc = time.clock() - t0
        return mat, t_calc

    def compare_timings(self):
        """
        # run different versions of the code
        """
        test_cases = [
            self.quadratic_case1,
            self.quadratic_case2,
            self.rectangular_case_1,
            self.rectangular_case_2
        ]
        variants = [False, True]
        run_cnt = 5

        timings = np.zeros((len(test_cases), len(variants)))
        for t_idx, test in enumerate(test_cases):
            results = []
            print(">>> Running: {}".format(test))
            for o_idx, optimize in enumerate(variants):
                t_sum = 0
                self.optimization = optimize
                for n in range(run_cnt):
                    _m, _t = test()
                    t_sum += _t
                results.append(_m)
                timings[t_idx, o_idx] = t_sum / run_cnt
            # they should compute the same results
            np.testing.assert_almost_equal(*results)

        print("Improvements due to symmetry exploitation (in %):")
        percentages = (100 * (1 - timings[:, 1] / timings[:, 0]))
        print(percentages)


class ProjectionTest(unittest.TestCase):
    def setUp(self):
        interval = (0, 10)
        node_cnt = 11
        self.nodes = pi.Domain(interval, num=node_cnt)
        self.lag_base = pi.LagrangeFirstOrder.cure_interval(self.nodes)
        pi.register_base("lag_base", self.lag_base, overwrite=True)

        # "real" functions
        # because we are smarter
        self.z_values = np.linspace(interval[0], interval[1], 100 * node_cnt)
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
        self.nodes = pi.Domain((0, 1), num=2)
        self.lag_base = pi.LagrangeFirstOrder.cure_interval(self.nodes)
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
        def _no_roots(omega):
            return 2 + np.sin(np.abs(omega))

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

        self.no_roots = _no_roots
        self.frequent_eq = _frequent_equation
        self.char_eq = _char_equation
        self.univariate_eq = _univariate_equation
        self.complex_eq = _complex_equation

        self.n_roots = 10
        self.small_grid = np.arange(0, 1, 1)
        self.grid = np.arange(0, 50, 1)
        self.rtol = .1

    def test_no_roots(self):
        # function does not have any roots
        roots = pi.find_roots(function=self.no_roots,
                              grid=self.grid, cmplx=False)
        self.assertEqual(len(roots), 0)
        roots = pi.find_roots(function=self.no_roots,
                              grid=[self.grid, self.grid], cmplx=True)
        self.assertEqual(len(roots), 0)

    def test_all_roots(self):
        grid = np.linspace(np.pi/20, 3*np.pi/2, num=20)
        roots = pi.find_roots(function=self.frequent_eq, grid=grid,
                              n_roots=self.n_roots, rtol=self.rtol/100)

        real_roots = [(2*k - 1)*np.pi/2/10 for k in range(1, self.n_roots+1)]
        np.testing.assert_array_almost_equal(roots, real_roots)

    def test_in_fact_roots(self):
        roots = pi.find_roots(function=self.char_eq, grid=self.grid,
                              n_roots=self.n_roots, rtol=self.rtol)

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

        # we deliberately request to be given zero roots
        roots = pi.find_roots(self.char_eq, self.grid, 0, self.rtol)
        self.assertEqual(len(roots), 0)

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

    def test_n_dim_func(self):
        grid = [np.linspace(0, 10),
                np.linspace(0, 2)]
        roots = pi.find_roots(function=self.univariate_eq, grid=grid, n_roots=6,
                              rtol=self.rtol)
        # TODO check results!

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


class DomainTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_init(self):
        # if bounds is given, num or step are required
        with self.assertRaises(ValueError):
            pi.Domain(bounds=(0, 1))

        # when given a number of steps
        d = pi.Domain(bounds=(0, 10), num=11)
        np.testing.assert_array_equal(d.bounds, (0, 10))
        self.assertEqual(d.step, 1)  # should be added automatically
        np.testing.assert_array_equal(d.points, np.linspace(0, 10, 11))

        # test default order
        d = pi.Domain((0, 10), 11)
        np.testing.assert_array_equal(d.bounds, (0, 10))
        self.assertEqual(d.step, 1)  # should be added automatically
        np.testing.assert_array_equal(d.points, np.linspace(0, 10, 11))

        # when given a step size
        d = pi.Domain(bounds=(0, 10), step=1)
        np.testing.assert_array_equal(d.bounds, (0, 10))
        self.assertEqual(d.step, 1)
        np.testing.assert_array_equal(d.points, np.linspace(0, 10, 11))

        # test default order
        d = pi.Domain((0, 10), None, 1)
        np.testing.assert_array_equal(d.bounds, (0, 10))
        self.assertEqual(d.step, 1)
        np.testing.assert_array_equal(d.points, np.linspace(0, 10, 11))

        # although a warning is displayed if the given step cannot be reached
        # by using num steps
        with self.assertWarns(UserWarning) as cm:
            d = pi.Domain(bounds=(0, 1), step=.4)
        w = cm.warning
        self.assertTrue("changing to" in w.args[0])

        # if both are given, num takes precedence
        d = pi.Domain((0, 10), 11, 1)
        np.testing.assert_array_equal(d.bounds, (0, 10))
        self.assertEqual(d.step, 1)
        np.testing.assert_array_equal(d.points, np.linspace(0, 10, 11))

        # although a warning is displayed if the given step cannot be reached
        # by using num steps
        with self.assertRaises(ValueError):
            pi.Domain(bounds=(0, 1), step=2, num=10)

        # if points are given, it always takes precedence
        p = np.linspace(0, 100, num=101)
        d = pi.Domain(points=p)
        np.testing.assert_array_equal(p, d.points)
        np.testing.assert_array_equal(d.bounds, [0, 100])
        self.assertEqual(d.step, 1)

        # check default args
        d = pi.Domain(None, None, None, p)
        np.testing.assert_array_equal(p, d.points)
        np.testing.assert_array_equal(d.bounds, [0, 100])
        self.assertEqual(d.step, 1)

        # if bounds, num or step are given, they have to fit the provided data
        d = pi.Domain(bounds=(0, 100), points=p)
        np.testing.assert_array_equal(p, d.points)
        np.testing.assert_array_equal(d.bounds, [0, 100])
        self.assertEqual(d.step, 1)

        with self.assertRaises(ValueError):
            pi.Domain(bounds=(0, 125), points=p)

        # good
        d = pi.Domain(num=101, points=p)
        np.testing.assert_array_equal(p, d.points)
        np.testing.assert_array_equal(d.bounds, [0, 100])
        self.assertEqual(d.step, 1)

        # bad
        with self.assertRaises(ValueError):
            pi.Domain(num=111, points=p)

        # good
        d = pi.Domain(step=1, points=p)
        np.testing.assert_array_equal(p, d.points)
        np.testing.assert_array_equal(d.bounds, [0, 100])
        self.assertEqual(d.step, 1)

        # bad
        with self.assertRaises(ValueError):
            pi.Domain(step=5, points=p)

        # if steps in pints are not regular, step attribute should not be set
        pw = p[[10, 12, 44, 58, 79]]
        d = pi.Domain(points=pw)
        np.testing.assert_array_equal(pw, d.points)
        np.testing.assert_array_equal(d.bounds, [10, 79])
        self.assertEqual(d.step, None)

        # also giving a step for this should fail
        with self.assertRaises(ValueError):
            pi.Domain(step=5, points=pw)

    def test_degenerate(self):
        # degenerated domains should not raise errors
        d = pi.Domain(bounds=(0, 1), num=1)
        d = pi.Domain(points=np.array([7]))

    def test_handling(self):
        d = pi.Domain(bounds=(10, 50), num=5)

        # the object should behave like an array
        self.assertEqual(len(d), 5)
        self.assertEqual(d[2], 30)
        np.testing.assert_array_equal(d[1:4:2], [20, 40])
        self.assertEqual(d.shape, (5,))

    def test_repr(self):
        d = pi.Domain(bounds=(10, 50), num=5)
        # printing the object should provide meaningful information
        s = str(d)
        self.assertIn("(10, 50)", s)
        self.assertIn("5", s)
        self.assertIn("10", s)


class EvalDataTestCase(unittest.TestCase):
    def setUp(self):
        # test_data_time1 = pi.core.Domain((0, 10), 11)
        test_data_time1 = np.linspace(0, 10, 11)
        test_data_spatial1 = pi.core.Domain((0, 1), 5)
        xx, yy = np.meshgrid(test_data_time1, test_data_spatial1.points,
                             indexing="ij")
        test_output_data1 = 2 * xx + 4 * yy
        self.data1 = pi.core.EvalData(input_data=[test_data_time1,
                                                  test_data_spatial1],
                                      output_data=test_output_data1)

        test_data_time2 = pi.core.Domain((0, 10), 11)
        test_data_spatial2 = pi.core.Domain((0, 1), 5)
        test_output_data2 = np.random.rand(11, 5)
        self.data2 = pi.core.EvalData(input_data=[test_data_time2,
                                                  test_data_spatial2],
                                      output_data=test_output_data2)

        test_data_time3 = pi.core.Domain((0, 10), 101)
        test_output_data3 = 10 * test_data_time3.points + 5
        self.data3 = pi.core.EvalData(input_data=[test_data_time3],
                                      output_data=test_output_data3)
        self.data3_extra = pi.core.EvalData(input_data=[test_data_time3],
                                            output_data=test_output_data3,
                                            enable_extrapolation=True)

        test_data_time4 = pi.core.Domain((0, 10), 101)
        test_data_spatial4 = pi.core.Domain((0, 1), 11)
        test_output_data4 = np.random.rand(101, 11)
        self.data4 = pi.core.EvalData(input_data=[test_data_time4,
                                                  test_data_spatial4],
                                      output_data=test_output_data4)

        test_data_time5 = pi.core.Domain((0, 10), 101)
        test_data_spatial5 = pi.core.Domain((0, 10), 11)
        test_output_data5 = -np.random.rand(101, 11)
        self.data5 = pi.core.EvalData(input_data=[test_data_time5,
                                                  test_data_spatial5],
                                      output_data=test_output_data5)

        test_data_time6 = pi.core.Domain((0, 10), 101)
        test_data_spatial6 = pi.core.Domain((0, 1), 101)
        test_output_data6 = -np.random.rand(101, 101)
        self.data6 = pi.core.EvalData(input_data=[test_data_time6,
                                                  test_data_spatial6],
                                      output_data=test_output_data6)

        test_data_time7 = pi.core.Domain((0, 10), 101)
        test_data_spatial7 = pi.core.Domain((0, 1), 101)
        test_output_data7 = -np.random.rand(101, 101)
        self.data7 = pi.core.EvalData(input_data=[test_data_time7,
                                                  test_data_spatial7],
                                      output_data=test_output_data7)

        test_data_time8 = pi.core.Domain((0, 10), 11)
        test_data_spatial8_1 = pi.core.Domain((0, 20), 21)
        test_data_spatial8_2 = pi.core.Domain((0, 30), 31)
        xx, yy, zz = np.meshgrid(test_data_time8.points,
                                 test_data_spatial8_1.points,
                                 test_data_spatial8_2.points,
                                 indexing="ij")
        test_output_data8 = 2 * xx - 4 * yy + 6 * zz
        self.data8 = pi.core.EvalData(input_data=[test_data_time8,
                                                  test_data_spatial8_1,
                                                  test_data_spatial8_2],
                                      output_data=test_output_data8)
        self.data8_extra = pi.core.EvalData(input_data=[test_data_time8,
                                                        test_data_spatial8_1,
                                                        test_data_spatial8_2],
                                            output_data=test_output_data8,
                                            enable_extrapolation=True)

    def test_init(self):
        test_data_1 = pi.core.Domain((0, 10), 11)
        test_output_data_1 = np.random.rand(11,)

        # single domain can be given for input data
        ed = pi.EvalData(input_data=test_data_1, output_data=test_output_data_1)

        # but standard case is a list
        ed = pi.EvalData(input_data=[test_data_1],
                         output_data=test_output_data_1)

        # same goes for numpy arrays
        ed = pi.EvalData(input_data=test_data_1.points,
                         output_data=test_output_data_1)
        ed = pi.EvalData(input_data=[test_data_1.points],
                         output_data=test_output_data_1)

        test_data_2 = pi.core.Domain((0, 1), 5)
        test_output_data_2 = np.random.rand(11, 5)

        # if several input vectors are given, they must be provided as list
        pi.EvalData(input_data=[test_data_1, test_data_2],
                    output_data=test_output_data_2)

        # not as tuple
        with self.assertRaises(AssertionError):
            pi.EvalData(input_data=(test_data_1, test_data_2),
                        output_data=test_output_data_2)

        # and the output should fit the dimensions of input_data
        test_output_data_3 = np.random.rand(11, 7)  # should've been (11, 5)
        with self.assertRaises(AssertionError):
            pi.EvalData(input_data=(test_data_1, test_data_2),
                        output_data=test_output_data_3)

        # and have no ndim > len(input_data)
        test_output_data_4 = np.random.rand(11, 5, 3, 7)
        with self.assertRaises(AssertionError):
            pi.EvalData(input_data=[test_data_1, test_data_2],
                        output_data=test_output_data_4)

        # although, the output can have more dimensions (like a vector field)
        # if the flag `fill_axes` is given
        # (dummy axis will be appended automatically)
        pi.EvalData(input_data=[test_data_1, test_data_2],
                    output_data=test_output_data_4,
                    fill_axes=True)

        # labels can be given for the given input axes
        pi.EvalData(input_data=[test_data_1, test_data_2],
                    output_data=test_output_data_2,
                    input_labels=["x", "y"])

        # but they have to fit the dimensions of input_data
        with self.assertRaises(AssertionError):
            pi.EvalData(input_data=[test_data_1, test_data_2],
                        output_data=test_output_data_2,
                        input_labels=["x"])

        # empty entries will be appended if axis are filled
        e = pi.EvalData(input_data=[test_data_1, test_data_2],
                        output_data=test_output_data_4,
                        input_labels=["x", "y"],
                        fill_axes=True)
        self.assertEqual(e.input_labels[2], "")

        # yet again for scalar case the list can be omitted
        pi.EvalData(input_data=test_data_1,
                    output_data=test_output_data_1,
                    input_labels="x")

        # also units can be given for the given input axes
        pi.EvalData(input_data=[test_data_1, test_data_2],
                    output_data=test_output_data_2,
                    input_units=["metre", "seconds"])

        # but they have to fit the dimensions of input_data
        with self.assertRaises(AssertionError):
            pi.EvalData(input_data=[test_data_1, test_data_2],
                        output_data=test_output_data_2,
                        input_units=["foot"])

        # empty entries will be appended if axis are filled
        e = pi.EvalData(input_data=[test_data_1, test_data_2],
                        output_data=test_output_data_4,
                        input_units=["kelvin", "calvin"],
                        fill_axes=True)
        self.assertEqual(e.input_units[2], "")

        # yet again for scalar case the list can be omitted
        pi.EvalData(input_data=test_data_1,
                    output_data=test_output_data_1,
                    input_labels="mississippis")

        # extrapolation for 2d is disabled, due to bugs in scipy
        with self.assertRaises(ValueError):
            self.data1_extra = pi.core.EvalData(
                input_data=[test_data_1, test_data_1],
                output_data=np.random.rand(11, 11),
                enable_extrapolation=True)

    def test_interpolate1d(self):
        data = self.data3.interpolate([[2, 7]])
        self.assertEqual(data.output_data[0], self.data3.output_data[20])
        self.assertEqual(data.output_data[1], self.data3.output_data[70])

    def test_interpolate2dAxis1(self):
        data = self.data1.interpolate([[2, 5], [0.5]])
        np.testing.assert_array_almost_equal(data.output_data[0],
                                             self.data1.output_data[2, 2])
        np.testing.assert_array_almost_equal(data.output_data[1],
                                             self.data1.output_data[5, 2])

    def test_interpolate2dAxis2(self):
        data = self.data1.interpolate([[2], [0.25, 0.5]])
        np.testing.assert_array_almost_equal(data.output_data[0],
                                             self.data1.output_data[2, 1])
        np.testing.assert_array_almost_equal(data.output_data[1],
                                             self.data1.output_data[2, 2])

    def test_call1d(self):
        data = self.data3([[1]])
        np.testing.assert_array_almost_equal(
            data.output_data,
            self.data3.output_data[[10]])

        data = self.data3([[1, 3, 6]])
        np.testing.assert_array_almost_equal(
            data.output_data,
            self.data3.output_data[[10, 30, 60]])

        data = self.data3([slice(None)])
        np.testing.assert_array_almost_equal(data.output_data,
                                             self.data3.output_data[:])

    def test_call1d_extrapolation(self):
        values = [-10, 0, 10, 20]

        # if extrapolation is disabled, the boundary values are used
        data = self.data3([values])
        np.testing.assert_array_almost_equal(
            data.output_data,
            self.data3.output_data[[0, 0, 100, 100]])

        # if enabled, extrapolated output is generated
        data = self.data3_extra([values])
        extra_out = 10 * np.array(values) + 5
        np.testing.assert_array_almost_equal(
            data.output_data,
            extra_out)

    def test_call_1d_simple(self):
        # these calls may also omit the outermost list since the array is 1d
        data = self.data3([1])
        np.testing.assert_array_almost_equal(
            data.output_data,
            self.data3.output_data[[10]])

        data = self.data3([1, 3, 6])
        np.testing.assert_array_almost_equal(
            data.output_data,
            self.data3.output_data[[10, 30, 60]])

        data = self.data3(slice(None))
        np.testing.assert_array_almost_equal(data.output_data,
                                             self.data3.output_data[:])

    @unittest.skip
    def test_call1dWrong(self):
        data = self.data3([[1, 2, slice(None)]])
        np.testing.assert_array_almost_equal(data.output_data[:],
                                             self.data3.output_data[:])

    def test_call2d(self):
        data = self.data1([slice(None), [0.75]])
        np.testing.assert_array_almost_equal(data.output_data,
                                             self.data1.output_data[:, 3])

    def test_call2d_extrapolation(self):
        # if extrapolation is disabled, the boundary values are used
        values = [-.5, 0, 1, 1.5]
        data = self.data1([slice(None), values])
        np.testing.assert_array_almost_equal(
            data.output_data,
            self.data1.output_data[:, [0, 0, -1, -1]]
        )
        # since BiRectSpline does not support extra extrapolation, if
        # extrapolation is forced, interp2d is used which, sadly has a bug
        # see https://github.com/scipy/scipy/issues/8099 for details

        # data = self.data1_extra([[5], values])
        # extra_output = (2 * 5 - 4 * np.array(values))
        # np.testing.assert_array_almost_equal(data.output_data, extra_output)

    def test_call2d_multi(self):
        data = self.data1([slice(None), [0.25, 0.75]])
        np.testing.assert_array_almost_equal(data.output_data[:, 0],
                                             self.data1.output_data[:, 1])
        np.testing.assert_array_almost_equal(data.output_data[:, 1],
                                             self.data1.output_data[:, 3])
        data1 = data([slice(None), [0.25, 0.75]])
        np.testing.assert_array_almost_equal(data1.output_data[:, 0],
                                             self.data1.output_data[:, 1])

    def test_call2d_slice(self):
        data = self.data1([slice(1, 5), [0.75]])
        np.testing.assert_array_almost_equal(data.output_data,
                                             self.data1.output_data[1:5, 3])

    def test_call3d_one_axis(self):
        data = self.data8(
            [slice(None, None, 2), slice(None), slice(None)])
        np.testing.assert_array_almost_equal(data.output_data,
                                             self.data8.output_data[::2])

        data = self.data8(
            [slice(None), slice(None, None, 2), slice(None)])
        np.testing.assert_array_almost_equal(data.output_data,
                                             self.data8.output_data[:, ::2])

        data = self.data8(
            [slice(None), slice(None), slice(None, None, 2)])
        np.testing.assert_array_almost_equal(data.output_data,
                                             self.data8.output_data[..., ::2])

    def test_call3d_two_axis(self):
        data = self.data8(
            [slice(None, None, 2), slice(5, 14, 2), slice(None)])
        np.testing.assert_array_almost_equal(
            data.output_data,
            self.data8.output_data[::2, 5:14:2, :])

    def test_call3d_three_axis(self):
        data = self.data8(
            [slice(None, None, 2), slice(5, 14, 2), slice(2, 27, 5)])
        np.testing.assert_array_almost_equal(
            data.output_data,
            self.data8.output_data[::2, 5:14:2, 2:27:5])

    def test_call3d_extrapolation(self):
        # no neigherest neighbour available,
        # values outside are padded with zeros
        data = self.data8([[2], [3], [-10, 0, 30, 40]])
        np.testing.assert_array_almost_equal(
            data.output_data,
            np.hstack((0, self.data8.output_data[2, 3, [0, -1]] , 0))
        )

        x = [-10, 0, 10, 20]
        y = [-10, 0, 20, 30]
        z = [-10, 0, 30, 40]
        data = self.data8_extra([x, y, z])
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        test_output_data8 = 2 * xx - 4 * yy + 6 * zz
        np.testing.assert_array_almost_equal(data.output_data,
                                             test_output_data8)

    def test_interpolation_chain(self):
        # we start with a 2d data  object and interpolate it on a wider grid
        data = self.data6([slice(None, None, 2), slice(None, None, 2)])
        self.assertEqual(data.output_data.shape, (51, 51))
        np.testing.assert_almost_equal(data.output_data,
                                       self.data6.output_data[::2, ::2])

        # now we will degenerate this 2d set by selecting a particular axis
        data1 = data([[5], slice(None)])
        # normally the result would be of shape (1, 50) but this is essentially
        # a 2d date set, so it is converted to be of shape (50, ) with the first
        # axis now being the former 2nd one.
        self.assertEqual(data1.output_data.shape, (51, ))
        np.testing.assert_almost_equal(data1.output_data,
                                       self.data6.output_data[50, ::2])

        # this should work for both axes
        data2 = data([slice(None), [.5]])
        self.assertEqual(data2.output_data.shape, (51, ))
        np.testing.assert_almost_equal(data2.output_data,
                                       self.data6.output_data[::2, 50])

        # now we again pick an particular value and receive a float
        data3 = data2([[7]])
        self.assertEqual(data3.output_data.shape, ())

        np.testing.assert_almost_equal(data3.output_data,
                                       self.data6.output_data[70, 50])

    def test_add_const(self):
        data = self.data1 + 4
        np.testing.assert_array_almost_equal(data.output_data,
                                             self.data1.output_data + 4)
        # check whether the information have been copied or only referenced
        data.input_data[0].points[0] = 999
        np.testing.assert_array_equal(self.data1.input_data[0],
                                      self.data2.input_data[0])

    def test_radd_const(self):
        data = 4 + self.data1
        np.testing.assert_array_almost_equal(data.output_data,
                                             self.data1.output_data + 4)

        # check whether the information have been copied or only referenced
        data.input_data[0].points[0] = 999
        np.testing.assert_array_equal(self.data1.input_data[0],
                                      self.data2.input_data[0])

    def test_add_evaldata(self):
        data = self.data1 + self.data2
        np.testing.assert_array_almost_equal(
            data.output_data,
            self.data1.output_data + self.data2.output_data)

        # check whether the information have been copied or only referenced
        data.input_data[0].points[0] = 999
        np.testing.assert_array_equal(self.data1.input_data[0],
                                      self.data2.input_data[0])

    def test_add_evaldata_diff(self):
        data = self.data1 + self.data4
        data4red = self.data4(self.data1.input_data)
        np.testing.assert_array_almost_equal(data.output_data,
                                             self.data1.output_data
                                             + data4red.output_data)

        # check whether the information have been copied or only referenced
        data.input_data[0].points[0] = 999
        np.testing.assert_array_equal(self.data1.input_data[0],
                                      self.data2.input_data[0])

    def test_sub_const(self):
        data = self.data1 - 4
        np.testing.assert_array_almost_equal(data.output_data,
                                             self.data1.output_data - 4)

        # check whether the information have been copied or only referenced
        data.input_data[0].points[0] = 999
        np.testing.assert_array_equal(self.data1.input_data[0],
                                      self.data2.input_data[0])

    def test_rsub_const(self):
        data = 4 - self.data1
        np.testing.assert_array_almost_equal(data.output_data,
                                             4 - self.data1.output_data)

        # check whether the information have been copied or only referenced
        data.input_data[0].points[0] = 999
        np.testing.assert_array_equal(self.data1.input_data[0],
                                      self.data2.input_data[0])

    def test_sub_evaldata(self):
        data = self.data1 - self.data2
        np.testing.assert_array_almost_equal(data.output_data,
                                             self.data1.output_data
                                             - self.data2.output_data)

        # check whether the information have been copied or only referenced
        data.input_data[0].points[0] = 999
        np.testing.assert_array_equal(self.data1.input_data[0],
                                      self.data2.input_data[0])

    def test_mul_const(self):
        data = self.data1 * 4
        np.testing.assert_array_almost_equal(data.output_data,
                                             self.data1.output_data * 4)

        # check whether the information have been copied or only referenced
        data.input_data[0].points[0] = 999
        np.testing.assert_array_equal(self.data1.input_data[0],
                                      self.data2.input_data[0])

    def test_mul_evaldata(self):
        data = self.data1 * self.data2
        np.testing.assert_array_almost_equal(
            data.output_data,
            self.data1.output_data * self.data2.output_data)

        # check whether the information have been copied or only referenced
        data.input_data[0].points[0] = 999
        np.testing.assert_array_equal(self.data1.input_data[0],
                                      self.data2.input_data[0])

    def test_rmul_const(self):
        data = 4 * self.data1
        np.testing.assert_array_almost_equal(data.output_data,
                                             self.data1.output_data * 4)

        # check whether the information have been copied or only referenced
        data.input_data[0].points[0] = 999
        np.testing.assert_array_equal(self.data1.input_data[0],
                                      self.data2.input_data[0])

    def test_matmul_evaldata(self):
        data = self.data6 @ self.data7
        np.testing.assert_array_almost_equal(
            data.output_data,
            self.data6.output_data @ self.data7.output_data)

        # check whether the information have been copied or only referenced
        data.input_data[0].points[0] = 999
        np.testing.assert_array_equal(self.data6.input_data[0],
                                      self.data7.input_data[0])

    def test_rmatmul_evaldata(self):
        data = self.data7 @ self.data6
        np.testing.assert_array_almost_equal(
            data.output_data,
            self.data7.output_data @ self.data6.output_data)

        # check whether the information have been copied or only referenced
        data.input_data[0].points[0] = 999
        np.testing.assert_array_equal(self.data6.input_data[0],
                                      self.data7.input_data[0])

    def test_sqrt_evaldata(self):
        data = self.data1.sqrt()
        np.testing.assert_array_almost_equal(data.output_data,
                                             np.sqrt(self.data1.output_data))

        # check whether the information have been copied or only referenced
        data.input_data[0].points[0] = 999
        np.testing.assert_array_equal(self.data1.input_data[0],
                                      self.data2.input_data[0])

    def test_abs_evaldata(self):
        data = self.data5.abs()
        np.testing.assert_array_almost_equal(data.output_data,
                                             np.abs(self.data5.output_data))

        # check whether the information have been copied or only referenced
        data.input_data[0].points[0] = 999
        np.testing.assert_array_equal(self.data5.input_data[0],
                                      self.data5.input_data[0])
