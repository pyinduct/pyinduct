#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
import pyinduct as pi
import pyinduct.registry as rg

"""
test_registry
----------------------------------

Tests for `registry` module.
"""


# TODO add TestCases
class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.nodes = pi.Domain((0, 1), num=10)
        self.base = pi.LagrangeFirstOrder.cure_interval(self.nodes)

        self.double_nodes = pi.Domain((0, 1), num=10)
        self.double_base = pi.LagrangeFirstOrder.cure_interval(
            self.double_nodes)

    def test_registration(self):
        # base should not be registered
        self.assertFalse(rg.is_registered("test_base"))

        # get_base should therefore return an error
        self.assertRaises(ValueError, rg.get_base, "test_base")

        # register the base -------------------------------------------------
        # label is not a string
        self.assertRaises(TypeError, rg.register_base, 1, self.base)

        # label must not be empty
        self.assertRaises(ValueError, rg.register_base, "", self.base)

        # this should be fine
        rg.register_base("test_base", self.base)

        # base should now be registered
        self.assertRaises(TypeError, rg.is_registered, 1)  # checking status per index
        self.assertTrue(rg.is_registered, "test_base")

        # double registration will raise an error
        self.assertRaises(ValueError, rg.register_base, "test_base", self.double_base)

        # de-register the base -------------------------------------------------
        # de-register a base by index
        self.assertRaises(TypeError, rg.deregister_base, 1)

        # de-register a base that is not present
        self.assertRaises(ValueError, rg.deregister_base, "test_base_extra")

        # "test_base" was registered before, should work
        rg.deregister_base("test_base")

        # base should  be deleted from registry
        self.assertFalse(rg.is_registered("test_base"))

        # re-register the base -------------------------------------------------
        # re-registration should work now
        rg.register_base("test_base", self.double_base)

        # base should now be registered
        self.assertTrue(rg.is_registered, "test_base")

        # base should be identical with local copy
        self.assertTrue(np.array_equal(rg.get_base("test_base"), self.double_base))  # order should default to 0
        self.assertTrue(np.array_equal(rg.get_base("test_base"), self.double_base))

    def tearDown(self):
        if rg.is_registered("test_base"):
            rg.deregister_base("test_base")
