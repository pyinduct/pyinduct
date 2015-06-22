from __future__ import division

__author__ = 'stefan'

import unittest
import numpy as np

from pyinduct import core, utils

import pyqtgraph as pg

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

    # def test_numeric(self):
    #     z_start = 0
    #     z_end = 1
    #     t_start = 0
    #     t_end = 5
    #     z_step = 0.01
    #     t_step = 0.01
    #     self.t_values = np.arange(t_start, t_end + t_step, t_step)
    #     self.z_values = np.arange(z_start, z_end + z_step, z_step)
    #
    #     # 1d tests
    #     self.test_data_1d = np.sin(self.z_values)
    #     self.func_data_1d = pyinduct.EvalData([self.z_values], self.test_data_1d)
    #
    #     # 2d tests
    #     # self.tt, self.zz = np.meshgrid(self.t_values, self.z_values, sparse=True)
    #     # self.test_data_2d = np.sin(self.zz) * self.tt
    #     # self.func_data_2d = pyinduct.EvalData([self.z_values, self.t_values], self.test_data_2d)

    # def test_eval(self):
    #     phi = pyinduct.Function(np.sin, domain=(0, np.pi))
    #     it = np.nditer(self.z_values, flags=['f_index'])
    #     while not it.finished:
    #         self.assertEqual(phi(it[0]), self.test_data_1d[it.index])
    #         it.iternext()
    #
    # def test_numeric_wo_data(self):
    #     self.assertRaises(TypeError, pyinduct.Function, self.test_data_1d)
    #
    # def test_numeric_1d(self):
    #     phi = pyinduct.Function(self.func_data_1d)
    #     it = np.nditer(self.z_values, flags=['f_index'])
    #     while not it.finished:
    #         self.assertEqual(phi(it[0]), self.test_data_1d[it.index])
    #         it.iternext()
    #
    #     # test one that is for sure not in
    #     self.assertRaises(ValueError, phi, 1e10)
    #
    # def test_numeric_2d(self):
    #     phi = pyinduct.Function(self.func_data_2d)
    #     itz = np.nditer(self.zz, flags=['f_index'])
    #     itt = np.nditer(self.tt, flags=['f_index'])
    #     while not itz.finished:
    #         while not itt.finished:
    #             self.assertEqual(phi(itz[0], itt[0]), self.test_data_2d[itz.index, itt.index])
    #             itt.iternext()
    #         itz.iternext()
    #
    #     # test one that is for sure not in
    #     self.assertRaises(ValueError, phi, 1e10, -1e10)


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

class InnerProductTestCase(unittest.TestCase):

    def setUp(self):
        self.f1 = core.Function(lambda x: 1, domain=(0, 10))
        self.f2 = core.Function(lambda x: 2, domain=(0, 5))
        self.f3 = core.Function(lambda x: 2, domain=(0, 5), nonzero=(2, 3))
        self.f4 = core.Function(lambda x: 2, domain=(0, 5), nonzero=(2, 2+1e-1))

        self.f5 = core.LagrangeFirstOrder(0, 1, 2)
        self.f6 = core.LagrangeFirstOrder(1, 2, 3)
        self.f7 = core.LagrangeFirstOrder(2, 3, 4)

    def test_domain(self):
        self.assertAlmostEqual(core.inner_product(self.f1, self.f2), 10)
        self.assertAlmostEqual(core.inner_product(self.f1, self.f3), 2)

    def test_nonzero(self):
        self.assertAlmostEqual(core.inner_product(self.f1, self.f4), 2e-1)

    def test_lagrange(self):
        self.assertAlmostEqual(core.inner_product(self.f5, self.f7), 0)
        self.assertAlmostEqual(core.inner_product(self.f5, self.f6), 1/6)
        self.assertAlmostEqual(core.inner_product(self.f7, self.f6), 1/6)
        self.assertAlmostEqual(core.inner_product(self.f5, self.f5), 2/3)


class ProjectionTest(unittest.TestCase):

    def setUp(self):
        interval = (0, 10)
        node_cnt = 11
        self.nodes, self.test_functions = utils.cure_interval(core.LagrangeFirstOrder, interval, node_count=node_cnt)

        # "real" functions
        self.z_values = np.linspace(interval[0], interval[1], 1e2*node_cnt)  # because we are smarter
        self.funcs = [core.Function(lambda x: 2*x),
                      core.Function(lambda x: x**2),
                      core.Function(lambda x: np.sin(x))
                      ]
        self.real_values = [[func(val) for val in self.z_values] for func in self.funcs]

    def test_types_projection(self):
        self.assertRaises(TypeError, core.project_on_test_functions, 1, 2)
        self.assertRaises(TypeError, core.project_on_test_functions, np.sin, np.sin)

    def test_projection_on_lag1st(self):
        weights = []

        # linear function -> should be fitted exactly
        weight = core.project_on_test_functions(self.funcs[0], self.test_functions[1])  # convenience wrapper
        weights.append(core.project_on_test_functions(self.funcs[0], self.test_functions))
        self.assertTrue(np.allclose(weights[-1], [self.funcs[0](z) for z in self.nodes]))

        # quadratic function -> should be fitted somehow close
        weights.append(core.project_on_test_functions(self.funcs[1], self.test_functions))
        self.assertTrue(np.allclose(weights[-1], [self.funcs[1](z) for z in self.nodes], atol=.5))

        # trig function -> will be crappy
        weights.append(core.project_on_test_functions(self.funcs[2], self.test_functions))

        if 0:
            # since test function are lagrange1st order, plotting the results is fairly easy
            self.app = pg.QtGui.QApplication([])
            for idx, w in enumerate(weights):
                pw = pg.plot(title="Weights {0}".format(idx))
                pw.plot(x=self.z_values, y=self.real_values[idx], pen="r")
                pw.plot(x=self.nodes, y=w, pen="b")

            self.app.exec_()

    def test_types_back_projection(self):
        self.assertRaises(TypeError, core.back_project_from_test_functions, 1, 2)
        self.assertRaises(TypeError, core.back_project_from_test_functions, 1.0, np.sin)

    def test_back_projection_from_lagrange_1st(self):
        vec_real_func = np.vectorize(self.funcs[0])
        real_weights = vec_real_func(self.nodes)
        func_handle = core.back_project_from_test_functions(real_weights, self.test_functions)
        vec_approx_func = np.vectorize(func_handle)
        self.assertTrue(np.allclose(vec_approx_func(self.z_values), vec_real_func(self.z_values)))

        if 0:
            # lines should exactly match
            self.app = pg.QtGui.QApplication([])
            pw = pg.plot(title="back projected linear function")
            pw.plot(x=self.z_values, y=vec_real_func(self.z_values), pen="r")
            pw.plot(x=self.z_values, y=vec_approx_func(self.z_values), pen="b")
            self.app.exec_()
