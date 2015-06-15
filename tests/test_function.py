from __future__ import division

__author__ = 'stefan'

import unittest
import numpy as np

from pyinduct import pyinduct

import pyqtgraph as pg

class FunctionTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_init(self):
        self.assertRaises(TypeError, pyinduct.Function, 42)
        pyinduct.Function(np.sin)

        for kwarg in ["domain", "nonzero"]:
            # some nice but wrong variants
            for val in ["4-2", dict(start=1, stop=2), [1, 2]]:
                self.assertRaises(TypeError, pyinduct.Function, np.sin, **{kwarg: val})

            # a correct one
            pyinduct.Function(np.sin, **{kwarg: (0, 10)})
            pyinduct.Function(np.sin, **{kwarg: [(0, 3), (5, 10)]})

            # check sorting
            p = pyinduct.Function(np.sin, **{kwarg: (0, -10)})
            self.assertEqual(getattr(p, kwarg), [(-10, 0)])
            p = pyinduct.Function(np.sin, **{kwarg: [(5, 0), (-10, -5)]})
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
        self.assertRaises(ValueError, pyinduct.LagrangeFirstOrder, 0, 5, 0)
        self.assertRaises(ValueError, pyinduct.LagrangeFirstOrder, 0, 5, 5)
        self.assertRaises(ValueError, pyinduct.LagrangeFirstOrder, 0, 0, 5)

    def test_rest(self):
        p1 = pyinduct.LagrangeFirstOrder(0, 1, 2)
        self.assertEqual(p1.domain, [(0, 2)])
        self.assertEqual(p1.nonzero, [(0, 2)])
        self.assertEqual(p1(0), 0)
        self.assertEqual(p1(0.5), 0.5)
        self.assertEqual(p1(1), 1)
        self.assertEqual(p1(1.5), .5)
        self.assertEqual(p1(2), 0)

        self.assertRaises(ValueError, p1, -1e3)
        self.assertRaises(ValueError, p1, 1e3)

        # self.assertEqual(p1.quad_int(), 2/3)


class IntersectionTestCase(unittest.TestCase):

    def test_wrong_arguments(self):
        # interval bounds not sorted
        self.assertRaises(ValueError, pyinduct.domain_intersection, (3, 2), (1, 3))
        # intervals not sorted
        self.assertRaises(ValueError, pyinduct.domain_intersection, [(4, 5), (1, 2)], (1, 3))
        # intervals useless
        self.assertRaises(ValueError, pyinduct.domain_intersection, [(4, 5), (5, 6)], (1, 3))

    def test_easy_intersections(self):
        self.assertEqual(pyinduct.domain_intersection((0, 2), (1, 3)), [(1, 2)])
        self.assertEqual(pyinduct.domain_intersection((0, 1), (1, 3)), [])
        self.assertEqual(pyinduct.domain_intersection((3, 5), (1, 3)), [])
        self.assertEqual(pyinduct.domain_intersection((3, 5), (1, 4)), [(3, 4)])
        self.assertEqual(pyinduct.domain_intersection((3, 5), (1, 6)), [(3, 5)])
        self.assertEqual(pyinduct.domain_intersection((3, 5), (6, 7)), [])

    def test_complex_intersections(self):
        self.assertEqual(pyinduct.domain_intersection([(0, 2), (3, 5)], (3, 4)), [(3, 4)])
        self.assertEqual(pyinduct.domain_intersection([(0, 2), (3, 5)], (1, 4)), [(1, 2), (3, 4)])
        self.assertEqual(pyinduct.domain_intersection((1, 4), [(0, 2), (3, 5)]), [(1, 2), (3, 4)])
        self.assertEqual(pyinduct.domain_intersection([(1, 3), (4, 6)], [(0, 2), (3, 5)]), [(1, 2), (4, 5)])
        self.assertEqual(pyinduct.domain_intersection([(-10, -4), (2, 5), (10, 17)], [(-20, -5), (3, 5), (7, 23)]),
                         [(-10, -5), (3, 5)], (10, 17))

class InnerProductTestCase(unittest.TestCase):

    def setUp(self):
        self.f1 = pyinduct.Function(lambda x: 1, domain=(0, 10))
        self.f2 = pyinduct.Function(lambda x: 2, domain=(0, 5))
        self.f3 = pyinduct.Function(lambda x: 2, domain=(0, 5), nonzero=(2, 3))
        self.f4 = pyinduct.Function(lambda x: 2, domain=(0, 5), nonzero=(2, 2+1e-1))

        self.f5 = pyinduct.LagrangeFirstOrder(0, 1, 2)
        self.f6 = pyinduct.LagrangeFirstOrder(1, 2, 3)
        self.f7 = pyinduct.LagrangeFirstOrder(2, 3, 4)

    def test_domain(self):
        self.assertAlmostEqual(pyinduct.inner_product(self.f1, self.f2), 10)
        self.assertAlmostEqual(pyinduct.inner_product(self.f1, self.f3), 2)

    def test_nonzero(self):
        self.assertAlmostEqual(pyinduct.inner_product(self.f1, self.f4), 2e-1)

    def test_lagrange(self):
        self.assertAlmostEqual(pyinduct.inner_product(self.f5, self.f7), 0)
        self.assertAlmostEqual(pyinduct.inner_product(self.f5, self.f6), 1/6)
        self.assertAlmostEqual(pyinduct.inner_product(self.f7, self.f6), 1/6)
        self.assertAlmostEqual(pyinduct.inner_product(self.f5, self.f5), 2/3)

class ProjectionTest(unittest.TestCase):

    def setUp(self):
        start = 0
        end = 10
        node_cnt = 100
        self.nodes = np.linspace(start, end, node_cnt)
        dz = (end - start) / (node_cnt-1)

        self.test_function = pyinduct.LagrangeFirstOrder(start, (end-start)/2, end)
        self.test_functions = [pyinduct.LagrangeFirstOrder(self.nodes[i]-dz, self.nodes[i], self.nodes[i]+dz)
                               for i in range(len(self.nodes))]

        # "real" functions
        self.z_values = np.linspace(start, end, 1e2*node_cnt)  # because we are smarter
        self.funcs = [pyinduct.Function(lambda x: x**2),
                      pyinduct.Function(lambda x: np.sin(x))
                      ]
        self.real_values = [[func(val) for val in self.z_values] for func in self.funcs]

    def test_types(self):
        self.assertRaises(TypeError, pyinduct.project_on_test_functions, 1, 2)
        self.assertRaises(TypeError, pyinduct.project_on_test_functions, np.sin, np.sin)

    def test_sin_on_lag1st(self):
        weights = []
        # quadratic function
        weight = pyinduct.project_on_test_functions(self.funcs[0], self.test_function)
        weights.append(pyinduct.project_on_test_functions(self.funcs[0], self.test_functions))
        # self.assertTrue(np.allclose(weights[-1], [self.funcs[0](z) for z in self.nodes], atol=0.5))

        # trig function
        weights.append(pyinduct.project_on_test_functions(self.funcs[1], self.test_functions))
        # self.assertTrue(np.allclose(weights[-1], [self.funcs[1](z) for z in self.nodes], atol=0.5))

        self.app = pg.QtGui.QApplication([])
        for idx, w in enumerate(weights):
            pw = pg.plot(title="Weights {0}".format(idx))
            pw.plot(x=self.z_values, y=self.real_values[idx], pen="r")
            pw.plot(x=self.nodes, y=w, pen="b")

        self.app.exec_()

