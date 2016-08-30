#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import unittest
from pyinduct import register_base, is_registered, deregister_base, get_base, cure_interval, LagrangeFirstOrder

"""
test_registry
----------------------------------

Tests for `registry` module.
"""


# TODO add TestCases
class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.nodes, self.base = cure_interval(LagrangeFirstOrder, (0, 1), node_count=10)
        self.double_nodes, self.double_base = cure_interval(LagrangeFirstOrder, (0, 2), node_count=20)

    def test_registration(self):
        # base should not be registered
        self.assertFalse(is_registered("test_base"))

        # get_base should therefore return an error
        self.assertRaises(ValueError, get_base, "test_base")

        # register the base -------------------------------------------------
        # label is not a string
        self.assertRaises(TypeError, register_base, 1, self.base)

        # this should be fine
        register_base("test_base", self.base)

        # base should now be registered
        self.assertRaises(TypeError, is_registered, 1)  # checking status per index
        self.assertTrue(is_registered, "test_base")

        # double registration will raise an error
        self.assertRaises(ValueError, register_base, "test_base", self.double_base)

        # de-register the base -------------------------------------------------
        # de-register a base by index
        self.assertRaises(TypeError, deregister_base, 1)

        # de-register a base that is not present
        self.assertRaises(ValueError, deregister_base, "test_base_extra")

        # "test_base" was registered before, should work
        deregister_base("test_base")

        # base should  be deleted from registry
        self.assertFalse(is_registered("test_base"))

        # re-register the base -------------------------------------------------
        # re-registration should work now
        register_base("test_base", self.double_base)

        # base should now be registered
        self.assertTrue(is_registered, "test_base")

        # base should be identical with local copy
        self.assertTrue(np.array_equal(get_base("test_base"), self.double_base))  # order should default to 0
        self.assertTrue(np.array_equal(get_base("test_base", order=0), self.double_base))

        # getting the first derivative should work
        b = get_base("test_base", 1)

        # getting the second derivative should raise an error
        b = self.assertRaises(ValueError, get_base, "test_base", 2)

    def tearDown(self):
        if is_registered("test_base"):
            deregister_base("test_base")
