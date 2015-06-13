__author__ = 'stefan'

import unittest
import numpy as np

from pyinduct import pyinduct


class FunctionTestCase(unittest.TestCase):

    def setUp(self):
        z_start = 0
        z_end = 1
        t_start = 0
        t_end = 5
        z_step = 0.01
        t_step = 0.01
        self.t_values = np.arange(t_start, t_end+t_step, t_step)
        self.z_values = np.arange(z_start, z_end+z_step, z_step)

        # 1d tests
        self.test_data_1d = np.sin(self.z_values)
        self.func_data_1d = pyinduct.EvalData([self.z_values], self.test_data_1d)

        # 2d tests
        self.tt, self.zz = np.meshgrid(self.t_values, self.z_values, sparse=True)
        self.test_data_2d = np.sin(self.zz)*self.tt
        self.func_data_2d = pyinduct.EvalData([self.z_values, self.t_values], self.test_data_2d)

    def test_analytic_wo_domain(self):
        self.assertRaises(ValueError, pyinduct.Function, np.sin)

    def test_analytic(self):
        phi = pyinduct.Function(np.sin, domain=(0, 2*np.pi))
        it = np.nditer(self.z_values, flags=['f_index'])
        while not it.finished:
            self.assertEqual(phi(it[0]), self.test_data_1d[it.index])
            it.iternext()

    def test_numeric_wo_data(self):
        self.assertRaises(TypeError, pyinduct.Function, self.test_data_1d)

    def test_numeric_1d(self):
        phi = pyinduct.Function(self.func_data_1d)
        it = np.nditer(self.z_values, flags=['f_index'])
        while not it.finished:
            self.assertEqual(phi(it[0]), self.test_data_1d[it.index])
            it.iternext()

        # test one that is for sure not in
        self.assertRaises(ValueError, phi, 1e10)

    def test_numeric_2d(self):
        phi = pyinduct.Function(self.func_data_2d)
        itz = np.nditer(self.zz, flags=['f_index'])
        itt = np.nditer(self.tt, flags=['f_index'])
        while not itz.finished:
            while not itt.finished:
                self.assertEqual(phi(itz[0], itt[0]), self.test_data_2d[itz.index, itt.index])
                itt.iternext()
            itz.iternext()

        # test one that is for sure not in
        self.assertRaises(ValueError, phi, 1e10, -1e10)

    def tearDown(self):
        pass


class IntersectionTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_wrong_arguments(self):
        # interval bounds not sorted
        self.assertRaises(ValueError, pyinduct.domain_intersection, (3, 2), (1, 3))
        # intervals not sorted
        self.assertRaises(ValueError, pyinduct.domain_intersection, [(4, 5), (1, 2)], (1, 3))
        # intervals useless
        self.assertRaises(ValueError, pyinduct.domain_intersection, [(4, 5), (5, 6)], (1, 3))

    def test_easy_intersections(self):
        self.assertEqual(pyinduct.domain_intersection((0, 2), (1, 3)), [(1, 2)])
        self.assertEqual(pyinduct.domain_intersection((0, 1), (1, 3)), [(1, 1)])
        self.assertEqual(pyinduct.domain_intersection((3, 5), (1, 3)), [(3, 3)])
        self.assertEqual(pyinduct.domain_intersection((3, 5), (1, 4)), [(3, 4)])
        self.assertEqual(pyinduct.domain_intersection((3, 5), (1, 6)), [(3, 5)])
        self.assertEqual(pyinduct.domain_intersection((3, 5), (6, 7)), [])

    def test_complex_intersections(self):
        self.assertEqual(pyinduct.domain_intersection([(0, 2), (3, 5)], (3, 4)), [(3, 4)])
        self.assertEqual(pyinduct.domain_intersection([(0, 2), (3, 5)], (1, 4)), [(1, 2), (3, 4)])
        self.assertEqual(pyinduct.domain_intersection((1, 4), [(0, 2), (3, 5)]), [(1, 2), (3, 4)])
        self.assertEqual(pyinduct.domain_intersection([(1, 3), (4, 6)], [(0, 2), (3, 5)]), [(1, 2), (3, 3), (4, 5)])
        self.assertEqual(pyinduct.domain_intersection([(-10, -4), (2, 5), (10, 17)], [(-20, -5), (3, 5), (7, 23)]),
                         [(-10, -5), (3, 5)], (10, 17))
