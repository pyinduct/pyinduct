#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_fd
----------------------------------

Tests for `fd` module.
"""

from __future__ import division
import unittest
import numpy as np
from pyinduct import fd
import pyqtgraph as pg
import sys

if any([arg == 'discover' for arg in sys.argv]):
    show_plots = False
else:
    show_plots = True
    app = pg.QtGui.QApplication([])


class TestFD(unittest.TestCase):

    def setUp(self):
        # results for centered approximation
        self._M = 4
        # self._alpha = [0, 1, -1, 2, -2, 3, -3, 4, -4]  # -> N = 8
        self._alpha = [-4, -3, -2, -1, 0, 1, 2, 3, 4]  # -> N = 8
        self._x0 = 0

        self._results = np.zeros((self._M, len(self._alpha), len(self._alpha)))
        # M = 0
        self._results[0, 0, :] = [0, 0, 0, 0, 1, 0, 0, 0, 0]
        # M = 1
        self._results[1, 2, :] = [0, 0, 0, -1/2, 0, -1/2, 0, 0, 0]
        self._results[1, 4, :] = [0, 0, 1/12, -2/3, 0, 2/3, -1/12, 0, 0]
        self._results[1, 6, :] = [0, -1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60, 0]
        self._results[1, 8, :] = [1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280]
        # TODO order 2, 3 and 4

    def test_weights(self):
        weights = fd._calc_weights(self._M, self._alpha, self._x0)
        # a = weights[0, 0, :]
        # b = self._results[0, 0, :]
        a = weights[1][2]
        b = self._results[1, 2, :]
        print("output: {0}".format(a))
        print("target: {0}".format(b))
        self.assertTrue(np.array_equal(a, b))

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
