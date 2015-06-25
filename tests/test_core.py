from __future__ import division
import unittest
import numpy as np
import pyqtgraph as pg
from pyinduct import core, utils

__author__ = 'stefan'

show_plots = False

class SanitizeInputTest(unittest.TestCase):

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

    def test_derivative(self):
        f = core.Function(np.sin, derivative_handles=[np.cos])
        self.assertRaises(ValueError, f.derivative, -1)  # stupid derivative
        self.assertRaises(ValueError, f.derivative, 2)  # unknown derivative
        d0 = f.derivative(0)
        self.assertEqual(f, d0)
        d1 = f.derivative(1)
        self.assertEqual(d1._function_handle, np.cos)
        self.assertRaises(ValueError, d1.derivative, 1)  # unknown derivative

class LagrangeFirstOrderTestCase(unittest.TestCase):
    def test_init(self):
        self.assertRaises(ValueError, core.LagrangeFirstOrder, 1, 0, 0)
        self.assertRaises(ValueError, core.LagrangeFirstOrder, 0, 1, 0)

    def test_edge_cases(self):
        p1 = core.LagrangeFirstOrder(0, 0, 1)
        self.assertEqual(p1.domain, [(-np.inf, np.inf)])
        self.assertEqual(p1.nonzero, [(0, 1)])
        self.assertEqual(p1(-.5), 0)
        self.assertEqual(p1(0), 1)
        self.assertEqual(p1(0.5), 0.5)
        self.assertEqual(p1(1), 0)
        self.assertEqual(p1(1.5), 0)

        p2 = core.LagrangeFirstOrder(0, 1, 1)
        self.assertEqual(p2.domain, [(-np.inf, np.inf)])
        self.assertEqual(p2.nonzero, [(0, 1)])
        self.assertEqual(p2(-.5), 0)
        self.assertEqual(p2(0), 0)
        self.assertEqual(p2(0.5), 0.5)
        self.assertEqual(p2(1), 1)
        self.assertEqual(p2(1.5), 0)

    def test_interior_case(self):
        p2 = core.LagrangeFirstOrder(0, 1, 2)
        self.assertEqual(p2.domain, [(-np.inf, np.inf)])
        self.assertEqual(p2.nonzero, [(0, 2)])
        self.assertEqual(p2(0), 0)
        self.assertEqual(p2(0.5), 0.5)
        self.assertEqual(p2(1), 1)
        self.assertEqual(p2(1.5), .5)
        self.assertEqual(p2(2), 0)

        # verify equality to zero anywhere outside of nonzero
        self.assertEqual(p2(-1e3), 0)
        self.assertEqual(p2(1e3), 0)

        # integral over whole nonzero area of self**2
        # self.assertEqual(p1.quad_int(), 2/3)

class MatrixFunctionTestCase(unittest.TestCase):

    def setUp(self):
        self.nodes, self.init_funcs = utils.cure_interval(core.LagrangeFirstOrder, (0, 1), node_count=2)

    def test_functional_call(self):
        res = core.calculate_function_matrix_differential(self.init_funcs, self.init_funcs, 0, 0)
        real_result = np.array([[1/3, 1/6], [1/6, 1/3]])
        self.assertTrue(np.allclose(res, real_result))

        res = core.calculate_function_matrix_differential(self.init_funcs, self.init_funcs, 1, 0)
        real_result = np.array([[-1/2, -1/2], [1/2, 1/2]])
        self.assertTrue(np.allclose(res, real_result))

        res = core.calculate_function_matrix_differential(self.init_funcs, self.init_funcs, 0, 1)
        real_result = np.array([[-1/2, 1/2], [-1/2, 1/2]])
        self.assertTrue(np.allclose(res, real_result))

        self.nodes, self.init_funcs = utils.cure_interval(core.LagrangeFirstOrder, (0, 1), node_count=9)
        res = core.calculate_function_matrix_differential(self.init_funcs, self.init_funcs, 1, 1)
        print(res)
        real_result = np.array([[1, -1], [-1, 1]])
        self.assertTrue(np.allclose(res, real_result))

    def test_scalar_call(self):
        res = core.calculate_function_matrix_differential(self.init_funcs, self.init_funcs, 0,  0, locations=(0.5, 0.5))
        real_result = np.array([[1/4, 1/4], [1/4, 1/4]])
        self.assertTrue(np.allclose(res, real_result))

        res = core.calculate_function_matrix_differential(self.init_funcs, self.init_funcs, 1,  0, locations=(0.5, 0.5))
        real_result = np.array([[-1/2, -1/2], [1/2, 1/2]])
        self.assertTrue(np.allclose(res, real_result))

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


class DotProductL2TestCase(unittest.TestCase):

    def setUp(self):
        self.f1 = core.Function(lambda x: 1, domain=(0, 10))
        self.f2 = core.Function(lambda x: 2, domain=(0, 5))
        self.f3 = core.Function(lambda x: 2, domain=(0, 5), nonzero=(2, 3))
        self.f4 = core.Function(lambda x: 2, domain=(0, 5), nonzero=(2, 2+1e-1))

        self.f5 = core.LagrangeFirstOrder(0, 1, 2)
        self.f6 = core.LagrangeFirstOrder(1, 2, 3)
        self.f7 = core.LagrangeFirstOrder(2, 3, 4)

    def test_domain(self):
        self.assertAlmostEqual(core.dot_product_l2(self.f1, self.f2), 10)
        self.assertAlmostEqual(core.dot_product_l2(self.f1, self.f3), 2)

    def test_nonzero(self):
        self.assertAlmostEqual(core.dot_product_l2(self.f1, self.f4), 2e-1)

    def test_lagrange(self):
        self.assertAlmostEqual(core.dot_product_l2(self.f5, self.f7), 0)
        self.assertAlmostEqual(core.dot_product_l2(self.f5, self.f6), 1/6)
        self.assertAlmostEqual(core.dot_product_l2(self.f7, self.f6), 1/6)
        self.assertAlmostEqual(core.dot_product_l2(self.f5, self.f5), 2/3)


class ProjectionTest(unittest.TestCase):

    def setUp(self):
        interval = (0, 10)
        node_cnt = 11
        self.nodes, self.initial_functions = utils.cure_interval(core.LagrangeFirstOrder, interval, node_count=node_cnt)

        # "real" functions
        self.z_values = np.linspace(interval[0], interval[1], 1e2*node_cnt)  # because we are smarter
        self.funcs = [core.Function(lambda x: 2*x),
                      core.Function(lambda x: x**2),
                      core.Function(lambda x: np.sin(x))
                      ]
        self.real_values = [[func(val) for val in self.z_values] for func in self.funcs]

    def test_types_projection(self):
        self.assertRaises(TypeError, core.project_on_initial_functions, 1, 2)
        self.assertRaises(TypeError, core.project_on_initial_functions, np.sin, np.sin)

    def test_projection_on_lag1st(self):
        weights = []

        # linear function -> should be fitted exactly
        weight = core.project_on_initial_functions(self.funcs[0], self.initial_functions[1])  # convenience wrapper
        weights.append(core.project_on_initial_functions(self.funcs[0], self.initial_functions))
        self.assertTrue(np.allclose(weights[-1], [self.funcs[0](z) for z in self.nodes]))

        # quadratic function -> should be fitted somehow close
        weights.append(core.project_on_initial_functions(self.funcs[1], self.initial_functions))
        self.assertTrue(np.allclose(weights[-1], [self.funcs[1](z) for z in self.nodes], atol=.5))

        # trig function -> will be crappy
        weights.append(core.project_on_initial_functions(self.funcs[2], self.initial_functions))

        if show_plots:
            # since test function are lagrange1st order, plotting the results is fairly easy
            self.app = pg.QtGui.QApplication([])
            for idx, w in enumerate(weights):
                pw = pg.plot(title="Weights {0}".format(idx))
                pw.plot(x=self.z_values, y=self.real_values[idx], pen="r")
                pw.plot(x=self.nodes, y=w, pen="b")

            self.app.exec_()

    def test_types_back_projection(self):
        self.assertRaises(TypeError, core.back_project_from_initial_functions, 1, 2)
        self.assertRaises(TypeError, core.back_project_from_initial_functions, 1.0, np.sin)

    def test_back_projection_from_lagrange_1st(self):
        vec_real_func = np.vectorize(self.funcs[0])
        real_weights = vec_real_func(self.nodes)
        func_handle = core.back_project_from_initial_functions(real_weights, self.initial_functions)
        vec_approx_func = np.vectorize(func_handle)
        self.assertTrue(np.allclose(vec_approx_func(self.z_values), vec_real_func(self.z_values)))

        if show_plots:
            # lines should match exactly
            self.app = pg.QtGui.QApplication([])
            pw = pg.plot(title="back projected linear function")
            pw.plot(x=self.z_values, y=vec_real_func(self.z_values), pen="r")
            pw.plot(x=self.z_values, y=vec_approx_func(self.z_values), pen="b")
            self.app.exec_()


class ChangeProjectionBaseTest(unittest.TestCase):

    def setUp(self):
        # real function
        self.z_values = np.linspace(0, 1, 1e3)
        self.real_func = core.Function(lambda x: x)
        self.real_func_handle = np.vectorize(self.real_func)

        # approximation by lag1st
        self.nodes, self.src_test_funcs = utils.cure_interval(core.LagrangeFirstOrder, (0, 1), node_count=2)
        self.src_weights = core.project_on_initial_functions(self.real_func, self.src_test_funcs)
        self.assertTrue(np.allclose(self.src_weights, [0, 1]))  # just to be sure
        self.src_approx_handle = np.vectorize(core.back_project_from_initial_functions(self.src_weights,
                                                                                    self.src_test_funcs))

        # approximation by sin(w*x)
        def trig_factory(freq):
            def func(x):
                return np.sin(freq*x)
            return func
        self.trig_test_funcs = np.array([core.Function(trig_factory(w), domain=(0, 1)) for w in range(1, 3)])

    def test_types_change_projection_base(self):
        self.assertRaises(TypeError, core.change_projection_base, 1, np.sin, np.cos)

    def test_lag1st_to_trig(self):
        # scalar case
        dest_weight = core.change_projection_base(self.src_weights, self.src_test_funcs, self.trig_test_funcs[0])
        dest_approx_handle_s = np.vectorize(core.back_project_from_initial_functions(dest_weight, self.trig_test_funcs[0]))

        # standard case
        dest_weights = core.change_projection_base(self.src_weights, self.src_test_funcs, self.trig_test_funcs)
        dest_approx_handle = np.vectorize(core.back_project_from_initial_functions(dest_weights, self.trig_test_funcs))
        error = np.sum(np.power(
            np.subtract(self.real_func_handle(self.z_values), dest_approx_handle(self.z_values)),
            2))
        # should fit pretty nice
        self.assertLess(error, 1e-2)

        if show_plots:
            self.app = pg.QtGui.QApplication([])
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
            self.app.exec_()
