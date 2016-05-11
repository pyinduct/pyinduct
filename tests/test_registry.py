#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from pyinduct import register_base, deregister_base, get_base

import sys
print(sys.argv)
"""
test_registry
----------------------------------

Tests for `registry` module.
"""


# TODO add TestCases
class RegistryTests(unittest.TestCase):

    def setUp(self):

        pass

    def test_registration(self):
        pass

    def test_deregistration(self):
        pass

    def tearDown(self):
        pass

if __name__ == '__main__':
    pass
    # initial_function_suite = test_initial_function.suite
    # all_tests = unittest.TestSuite([initial_function_suite])
    # unittest.TextTestRunner().run(all_tests)
