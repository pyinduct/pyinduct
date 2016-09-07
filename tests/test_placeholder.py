import sys
import unittest

import numpy as np

from pyinduct import core as cr, simulation as sim, utils as ut, placeholder as ph
from pyinduct import register_base, LagrangeFirstOrder, cure_interval

# TODO Test for all Placeholders
if any([arg in {'discover', 'setup.py', 'test'} for arg in sys.argv]):
    show_plots = False
else:
    # show_plots = True
    show_plots = False

if show_plots:
    import pyqtgraph as pg

    app = pg.QtGui.QApplication([])


class TestPlaceHolder(unittest.TestCase):
    """
    Test cases for the Placeholder base class
    """

    def setUp(self):
        self.data = dict(a=10, b="hallo")

    def init_test(self):
        # wrong derivative orders
        self.assertRaises(ValueError, ph.Placeholder, self.data, [1, 2])  # wrong type
        self.assertRaises(ValueError, ph.Placeholder, self.data, (-1, 2))  # negative order
        self.assertRaises(ValueError, ph.Placeholder, self.data, (1.3, 2))  # non integer order

        # location
        self.assertRaises(TypeError, ph.Placeholder, self.data, (1, 2), location="here")  # wrong type

        # positive tests
        p = ph.Placeholder(self.data, (1, 2), location=-3.7)
        self.assertEquals(p.data, self.data)
        self.assertEquals(p.order, (1, 2))
        self.assertEquals(p.location, -3.7)

    def derive_test(self):
        p = ph.Placeholder(self.data, (1, 2), location=-3.7)
        p_dt = p.derive(temp_order=1)

        # derivative order of p_dt should be changed, rest should stay as is
        o1 = p.__dict__.pop("order")
        o2 = p_dt.__dict__.pop("order")
        self.assertEquals(o1, (1, 2))
        self.assertEquals(o2, (2, 2))
        self.assertEquals(p.__dict__, p_dt.__dict__)

    def locate_test(self):
        p = ph.Placeholder(self.data, (1, 2), location=None)
        p_at_10 = p(10)

        # location p_at_10 should be changed, rest should stay as is
        o1 = p.__dict__.pop("location")
        o2 = p_at_10.__dict__.pop("location")
        self.assertEquals(o1, None)
        self.assertEquals(o2, 10)
        self.assertEquals(p.__dict__, p_at_10.__dict__)


class TestCommonTarget(unittest.TestCase):
    def test_call(self):
        t1 = ph.Scalars(np.zeros(2), target_term=dict(name="E", order=1, exponent=1))
        t2 = ph.Scalars(np.zeros(2), target_term=dict(name="E", order=2, exponent=1))
        t3 = ph.Scalars(np.zeros(2), target_term=dict(name="f"))
        t4 = ph.Scalars(np.zeros(2), target_term=dict(name="f"))

        # simple case
        self.assertEqual(ph.get_common_target([t1]), t1.target_term)
        self.assertEqual(ph.get_common_target([t2]), t2.target_term)
        self.assertEqual(ph.get_common_target([t3]), t3.target_term)

        # E precedes f
        self.assertEqual(ph.get_common_target([t2, t3]), t2.target_term)
        self.assertEqual(ph.get_common_target([t4, t1, t3]), t1.target_term)

        # different derivatives produce problems
        self.assertRaises(ValueError, ph.get_common_target, [t1, t2])


class ScalarsTest(unittest.TestCase):
    def setUp(self):
        self.vector = np.array(range(10))
        self.matrix = np.array(range(100)).reshape((10, 10))

    def test_init(self):
        # iterable values have to be provided
        self.assertRaises(TypeError, ph.Scalars, None)

        # test defaults
        t = ph.Scalars(self.vector)
        t.data = np.atleast_2d(self.vector)
        t.target_term = dict(name="f")
        t.target_form = None


class FieldVariableTest(unittest.TestCase):
    def setUp(self):
        nodes, ini_funcs = cure_interval(LagrangeFirstOrder, (0, 1), node_count=2)
        register_base("test_funcs", ini_funcs, overwrite=True)

    def test_FieldVariable(self):
        self.assertRaises(TypeError, ph.FieldVariable, "test_funcs", [0, 0])  # list instead of tuple
        self.assertRaises(ValueError, ph.FieldVariable, "test_funcs", (3, 0))  # order too high
        self.assertRaises(ValueError, ph.FieldVariable, "test_funcs", (0, 3))  # order too high
        self.assertRaises(ValueError, ph.FieldVariable, "test_funcs", (2, 2))  # order too high
        self.assertRaises(ValueError, ph.FieldVariable, "test_funcs", (-1, 3))  # order negative

        # defaults
        a = ph.FieldVariable("test_funcs")
        self.assertEqual((0, 0), a.order)
        self.assertEqual("test_funcs", a.data["weight_lbl"])  # default weight label is function label
        self.assertEqual(None, a.location)
        self.assertEqual(1, a.data["exponent"])  # default exponent is 1

        b = ph.FieldVariable("test_funcs", order=(1, 1), location=7, weight_label="test_lbl", exponent=10)
        self.assertEqual((1, 1), b.order)
        self.assertEqual("test_lbl", b.data["weight_lbl"])  # default weight label is function label
        self.assertEqual(7, b.location)
        self.assertEqual(10, b.data["exponent"])

    def test_call_factory(self):
        a = ph.FieldVariable("test_funcs")
        b = a(1)
        self.assertEqual("test_funcs", b.data["weight_lbl"])  # default weight label is function label
        self.assertEqual(1, b.location)
        self.assertTrue(isinstance(b, ph.FieldVariable))
        self.assertTrue(a != b)

    def test_derive_factory(self):
        a = ph.FieldVariable("test_funcs")
        b = a(1).derive(spat_order=1)
        self.assertEqual("test_funcs", b.data["weight_lbl"])  # default weight label is function label
        self.assertEqual(1, b.location)
        self.assertEqual(1, b.order[1])
        c = b.derive(spat_order=1)
        self.assertEqual(2, c.order[1])
        self.assertTrue(isinstance(b, ph.FieldVariable))
        self.assertTrue(a != b)
        self.assertTrue(a.order[0] == b.order[0] == c.order[0])
        self.assertTrue(a.order[1] != b.order[1] != c.order[1])


class TestFunctionTest(unittest.TestCase):
    def setUp(self):
        nodes, ini_funcs = cure_interval(LagrangeFirstOrder, (0, 1), node_count=2)
        register_base("test_funcs", ini_funcs, overwrite=True)

    def test_call_factory(self):
        a = ph.TestFunction("test_funcs")
        b = a(1)
        self.assertEqual("test_funcs", b.data["func_lbl"])
        self.assertEqual(1, b.location)
        self.assertTrue(isinstance(b, ph.TestFunction))
        self.assertTrue(a != b)

    def test_derive_factory(self):
        a = ph.TestFunction("test_funcs")
        b = a(1).derive(1)
        self.assertEqual("test_funcs", b.data["func_lbl"])
        self.assertEqual(1, b.location)
        self.assertEqual(1, b.order[1])
        c = b.derive(1)
        self.assertEqual(2, c.order[1])
        self.assertTrue(isinstance(b, ph.TestFunction))
        self.assertTrue(a != b)
        self.assertTrue(a.order[1] != b.order[1] != c.order[1])


class ProductTest(unittest.TestCase):
    def scale(self, z):
        return 2

    def a2(self, z):
        return 5 * z

    def setUp(self):
        self.input = ph.Input(np.sin)

        phi = cr.Function(np.sin)
        psi = cr.Function(np.cos)
        self.t_funcs = np.array([phi, psi])
        register_base("funcs", self.t_funcs, overwrite=True)
        self.test_funcs = ph.TestFunction("funcs")

        self.s_funcs = np.array([cr.Function(self.scale)])[[0, 0]]
        register_base("scale_funcs", self.s_funcs, overwrite=True)
        self.scale_funcs = ph.ScalarFunction("scale_funcs")

        nodes, self.ini_funcs = cure_interval(LagrangeFirstOrder, (0, 1), node_count=2)
        register_base("prod_ini_funcs", self.ini_funcs, overwrite=True)
        self.field_var = ph.FieldVariable("prod_ini_funcs")
        self.field_var_dz = ph.SpatialDerivedFieldVariable("prod_ini_funcs", 1)

    def test_product(self):
        self.assertRaises(TypeError, ph.Product, cr.Function, cr.Function)  # only Placeholders allowed
        p1 = ph.Product(self.input, self.test_funcs)
        p2 = ph.Product(self.test_funcs, self.field_var)

        # test single argument call
        p3 = ph.Product(self.test_funcs)
        self.assertTrue(p3.b_empty)
        res = ut.evaluate_placeholder_function(p3.args[0], np.pi / 2)
        self.assertTrue(np.allclose(res, [1, 0]))

        # test automated evaluation of Product with Scaled function
        p4 = ph.Product(self.field_var, self.scale_funcs)
        self.assertTrue(isinstance(p4.args[0], ph.Placeholder))
        res = ut.evaluate_placeholder_function(p4.args[0], 0)
        self.assertTrue(np.allclose(res, self.scale(0) * np.array([self.ini_funcs[0](0), self.ini_funcs[1](0)])))
        self.assertEqual(p4.args[1], None)
        self.assertTrue(p4.b_empty)

        # test automated simplification of cascaded products
        p5 = ph.Product(ph.Product(self.field_var, self.scale_funcs),
                        ph.Product(self.test_funcs, self.scale_funcs))
        self.assertFalse(p5.b_empty)

        p6 = ph.Product(ph.Product(self.field_var_dz, self.scale_funcs),
                        ph.Product(self.test_funcs, self.scale_funcs))
        self.assertFalse(p6.b_empty)

        res = ut.evaluate_placeholder_function(p5.args[0], 0)
        self.assertTrue(np.allclose(res, self.scale(0) * np.array([self.ini_funcs[0](0), self.ini_funcs[1](0)])))
        res1 = ut.evaluate_placeholder_function(p5.args[0], 1)
        self.assertTrue(np.allclose(res1, self.scale(1) * np.array([self.ini_funcs[0](1), self.ini_funcs[1](1)])))

        res2 = ut.evaluate_placeholder_function(p5.args[1], 0)
        self.assertTrue(np.allclose(res2, self.scale(0) * np.array([self.t_funcs[0](0), self.t_funcs[1](0)])))
        res3 = ut.evaluate_placeholder_function(p5.args[1], 1)
        self.assertTrue(np.allclose(res3, self.scale(0) * np.array([self.t_funcs[0](1), self.t_funcs[1](1)])))

        # test methods
        self.assertEqual(p1.get_arg_by_class(ph.Input), [self.input])
        self.assertEqual(p1.get_arg_by_class(ph.TestFunction), [self.test_funcs])
        self.assertEqual(p2.get_arg_by_class(ph.TestFunction), [self.test_funcs])
        self.assertEqual(p2.get_arg_by_class(ph.FieldVariable), [self.field_var])


class EquationTermsTest(unittest.TestCase):
    def setUp(self):
        self.input = ph.Input(np.sin)
        self.phi = np.array([cr.Function(lambda x: 2 * x)])
        register_base("phi", self.phi, overwrite=True)
        self.test_func = ph.TestFunction("phi")

        nodes, self.ini_funcs = cure_interval(LagrangeFirstOrder, (0, 1), node_count=2)
        register_base("ini_funcs", self.ini_funcs, overwrite=True)
        self.xdt = ph.TemporalDerivedFieldVariable("ini_funcs", order=1)
        self.xdz_at1 = ph.SpatialDerivedFieldVariable("ini_funcs", order=1, location=1)

        self.prod = ph.Product(self.input, self.xdt)

    def test_EquationTerm(self):
        self.assertRaises(TypeError, ph.EquationTerm, "eleven", self.input)  # scale is not a number
        self.assertRaises(TypeError, ph.EquationTerm, 1, LagrangeFirstOrder(0, 1, 2))  # arg is invalid
        ph.EquationTerm(1, self.test_func)
        ph.EquationTerm(1, self.xdt)
        t1 = ph.EquationTerm(1, self.input)
        self.assertEqual(t1.scale, 1)
        self.assertEqual(t1.arg.args[0], self.input)  # automatically create Product object if only one arg is provided

    def test_ScalarTerm(self):
        self.assertRaises(TypeError, ph.ScalarTerm, 7)  # factor is number
        self.assertRaises(TypeError, ph.ScalarTerm, cr.Function(np.sin))  # factor is Function
        ph.ScalarTerm(self.input)
        self.assertRaises(ValueError, ph.ScalarTerm, self.test_func)  # integration has to be done
        t1 = ph.ScalarTerm(self.xdz_at1)
        self.assertEqual(t1.scale, 1.0)  # default scale
        # check if automated evaluation works
        self.assertTrue(np.allclose(t1.arg.args[0].data, np.array([-1, 1])))

    def test_IntegralTerm(self):
        self.assertRaises(TypeError, ph.IntegralTerm, 7, (0, 1))  # integrand is number
        self.assertRaises(TypeError, ph.IntegralTerm, cr.Function(np.sin), (0, 1))  # integrand is Function
        self.assertRaises(ValueError, ph.IntegralTerm, self.xdz_at1, (0, 1))  # nothing left after evaluation
        self.assertRaises(TypeError, ph.IntegralTerm, self.xdt, [0, 1])  # limits is list

        ph.IntegralTerm(self.test_func, (0, 1))  # integrand is Placeholder
        self.assertRaises(ValueError, ph.IntegralTerm, self.input, (0, 1))  # nothing to do
        ph.IntegralTerm(self.xdt, (0, 1))  # integrand is Placeholder
        ph.IntegralTerm(self.prod, (0, 1))  # integrand is Product

        t1 = ph.IntegralTerm(self.xdt, (0, 1))
        self.assertEqual(t1.scale, 1.0)  # default scale
        self.assertEqual(t1.arg.args[0], self.xdt)  # automated product creation
        self.assertEqual(t1.limits, (0, 1))


class WeakFormulationTest(unittest.TestCase):
    def setUp(self):
        self.u = np.sin
        self.input = ph.Input(self.u)  # control input
        nodes, self.ini_funcs = cure_interval(LagrangeFirstOrder, (0, 1), node_count=3)
        register_base("ini_funcs", self.ini_funcs, overwrite=True)

        self.phi = ph.TestFunction("ini_funcs")  # eigenfunction or something else
        self.dphi = ph.TestFunction("ini_funcs", order=1)  # eigenfunction or something else
        self.dphi_at1 = ph.TestFunction("ini_funcs", order=1, location=1)  # eigenfunction or something else
        self.field_var = ph.FieldVariable("ini_funcs")
        self.field_var_at1 = ph.FieldVariable("ini_funcs", location=1)

    def test_init(self):
        self.assertRaises(TypeError, sim.WeakFormulation, ["a", "b"])
        sim.WeakFormulation(ph.ScalarTerm(self.field_var_at1))  # scalar case
        sim.WeakFormulation([ph.ScalarTerm(self.field_var_at1), ph.IntegralTerm(self.field_var, (0, 1))])  # vector case
