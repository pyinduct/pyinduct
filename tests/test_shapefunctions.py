import sys
import unittest

import numpy as np
import sympy as sp
import time

import pyinduct as pi
import pyinduct.shapefunctions as sh
from pyinduct.visualization import create_colormap
import test_data.test_shapefunctions_data

if any([arg in {'discover', 'setup.py', 'test'} for arg in sys.argv]):
    show_plots = False
else:
    # show_plots = True
    show_plots = False

if show_plots:
    import pyqtgraph as pg

    app = pg.QtGui.QApplication([])


class CureTestCase(unittest.TestCase):
    def test_init(self):
        self.assertRaises(TypeError, sh.cure_interval, np.sin, [2, 3])
        self.assertRaises(TypeError, sh.cure_interval, np.sin, (2, 3))
        self.assertRaises(ValueError, sh.cure_interval, sh.LagrangeFirstOrder,
                          (0, 2))
        self.assertRaises(ValueError, sh.cure_interval, sh.LagrangeFirstOrder,
                          (0, 2), 2, 1)

    def test_smoothness(self):
        func_classes = [pi.LagrangeFirstOrder, pi.LagrangeSecondOrder]
        derivatives = {pi.LagrangeFirstOrder: range(0, 2),
                       pi.LagrangeSecondOrder: range(0, 3)}
        tolerances = {pi.LagrangeFirstOrder: [5e0, 1.5e2],
                      pi.LagrangeSecondOrder: [1e0, 1e2, 9e2]}

        for func_cls in func_classes:
            for order in derivatives[func_cls]:
                self.assertGreater(tolerances[func_cls][order], self.shape_generator(func_cls, order))

    def shape_generator(self, cls, der_order):
        """
        verify the correct connection with visual feedback
        """

        dz = pi.Domain((0, 1), step=.001)
        dt = pi.Domain((0, 0), num=1)

        nodes, funcs = pi.cure_interval(cls, dz.bounds, node_count=11)
        pi.register_base("test", funcs, overwrite=True)

        # approx_func = pi.Function(np.cos, domain=dz.bounds,
        #                           derivative_handles=[lambda z: -np.sin(z), lambda z: -np.cos(z)])
        approx_func = pi.Function(lambda z: np.sin(3 * z), domain=dz.bounds,
                                  derivative_handles=[lambda z: 3 * np.cos(3 * z), lambda z: -9 * np.sin(3 * z)])

        weights = approx_func(nodes)

        hull = pi.evaluate_approximation("test", np.atleast_2d(weights),
                                         temp_domain=dt, spat_domain=dz, spat_order=der_order)

        if show_plots:
            # plot shapefunctions
            c_map = create_colormap(len(funcs))
            pw = pg.plot(title="{}-Test".format(cls.__name__))
            pw.addLegend()
            pw.showGrid(x=True, y=True, alpha=0.5)

            [pw.addItem(pg.PlotDataItem(np.array(dz),
                                        weights[idx] * func.derive(der_order)(dz),
                                        pen=pg.mkPen(color=c_map[idx]),
                                        name="{}.{}".format(cls.__name__, idx)))
             for idx, func in enumerate(funcs)]

            # plot hull curve
            pw.addItem(pg.PlotDataItem(np.array(hull.input_data[1]), hull.output_data[0, :],
                                       pen=pg.mkPen(width=2), name="hull-curve"))
            # plot original function
            pw.addItem(pg.PlotDataItem(np.array(dz), approx_func.derive(der_order)(dz),
                                       pen=pg.mkPen(color="m", width=2, style=pg.QtCore.Qt.DashLine), name="original"))
            pg.QtCore.QCoreApplication.instance().exec_()

        return np.sum(np.abs(hull.output_data[0, :] - approx_func.derive(der_order)(dz)))


class NthOrderCureTestCase(unittest.TestCase):
    def test_element(self):
        nodes = np.array([1, 2])
        self.assertRaises(ValueError, sh.LagrangeNthOrder, 0, nodes)
        self.assertRaises(ValueError, sh.LagrangeNthOrder, 1, np.array([2, 1]))
        self.assertRaises(TypeError, sh.LagrangeNthOrder, 1, nodes, left=1)
        self.assertRaises(TypeError, sh.LagrangeNthOrder, 1, nodes, right=1)
        self.assertRaises(ValueError, sh.LagrangeNthOrder, 3, nodes, mid_num=3)
        self.assertRaises(ValueError, sh.LagrangeNthOrder, 3, nodes)

    def test_cure(self):
        self.assertRaises(TypeError, sh.cure_interval, np.sin, [2, 3])
        self.assertRaises(TypeError, sh.cure_interval, np.sin, (2, 3))
        self.assertRaises(ValueError, sh.cure_interval, sh.LagrangeNthOrder,
                          (0, 2))
        self.assertRaises(ValueError, sh.cure_interval, sh.LagrangeNthOrder,
                          (0, 2), 2, 1)

    def test_smoothness(self):
        self.tolerances = test_data.test_shapefunctions_data.tolerances
        for conf in range(5):
            orders = range(1, 5)
            self.t_smoothness(orders, conf)

    def t_smoothness(self, orders, conf):
        derivatives = dict([(order, range(0, order + 1)) for order in orders])

        # approximation function
        z = sp.symbols("z")
        sin_func = [sp.sin(3 * z)]
        [sin_func.append(sin_func[-1].diff()) for i in range(orders[-1])]
        lam_sin_func = [sp.lambdify(z, func) for func in sin_func]
        approx_func = pi.Function(lam_sin_func[0], domain=(0, 1), derivative_handles=lam_sin_func[1:])

        dz = pi.Domain((0, 1), step=.001)
        dt = pi.Domain((0, 0), num=1)

        for order in orders:
            num_nodes = 1 + (1 + conf) * order
            nodes, funcs = pi.cure_interval(sh.LagrangeNthOrder, (0, 1), node_count=num_nodes, order=order)
            pi.register_base("test", funcs, overwrite=True)
            weights = approx_func(nodes)

            for der_order in derivatives[order]:
                hull_test = pi.evaluate_approximation("test", np.atleast_2d(weights),
                                                      temp_domain=dt, spat_domain=nodes, spat_order=der_order)
                self.assertAlmostEqual(self.tolerances[(order, num_nodes, der_order)], np.sum(np.abs(
                    hull_test.output_data[0, :] - approx_func.derive(der_order)(nodes))) / len(nodes))

                if show_plots:
                    hull_show = pi.evaluate_approximation("test", np.atleast_2d(weights),
                                                          temp_domain=dt, spat_domain=dz, spat_order=der_order)
                    # plot shapefunctions
                    c_map = create_colormap(len(funcs))
                    win = pg.GraphicsWindow(title="Debug window")
                    win.resize(1500, 600)
                    pw1 = win.addPlot()
                    pw1.addLegend()
                    pw1.showGrid(x=True, y=True, alpha=0.5)
                    pw2 = win.addPlot(
                        title="{} lagrange shapefunctions of order {}, derivative {}".format(num_nodes, order,
                                                                                             der_order))
                    pw2.showGrid(x=True, y=True, alpha=0.5)

                    for idx, func in enumerate(funcs):
                        pw1.addItem(pg.PlotDataItem(np.array(dz),
                                                    weights[idx] * func.derive(der_order)(dz),
                                                    pen=pg.mkPen(color=c_map[idx]),
                                                    name="{}.{}".format(order, idx)))
                        pw2.addItem(pg.PlotDataItem(np.array(dz),
                                                    func.derive(der_order)(dz),
                                                    pen=pg.mkPen(color=c_map[idx])))

                    # plot hull curve
                    pw1.addItem(pg.PlotDataItem(np.array(hull_show.input_data[1]), hull_show.output_data[0, :],
                                                pen=pg.mkPen(width=2), name="hull-curve"))
                    # plot original function
                    pw1.addItem(pg.PlotDataItem(np.array(dz), approx_func.derive(der_order)(dz),
                                                pen=pg.mkPen(color="m", width=2, style=pg.QtCore.Qt.DashLine),
                                                name="original"))
                    pg.QtCore.QCoreApplication.instance().exec_()


class SpeedTest(unittest.TestCase):
    """
    Compare LagrangeNthOrder with LagrangeSecondOrder (have a look at terminal output).

    When it succeeds to get positive values (at least a few) under the "Difference" headline
    by the transport system example, too, you can delete:
    - this test case
    - LagrangeFirstOrder
    - LagrangeSecondOrder
    """

    def test_implementation(self):

        impl = dict()
        headlines = ("New implementation", "Old implementation")

        print(">>> PyInduct transport system example speed test: \n")
        for headline in headlines:
            impl[headline] = dict()
            impl[headline]["pi_time"], impl[headline]["funcs"] = self.t_implementation(headline)
            self.print_time(headline, impl[headline]["pi_time"])
        self.print_time("Difference", np.array(impl[headlines[1]]["pi_time"]) - np.array(impl[headlines[0]]["pi_time"]))

        print(">>> pyinduct.core.Function speed test:")
        z = np.linspace(0, 1, 1e4)
        for headline in headlines:
            impl[headline]["f_time"] = list()
            print("\n" + headline)
            for der_order in range(len(impl[headline]["funcs"][0]._derivative_handles) + 1):
                _t = time.time()
                for idx, func in enumerate(impl[headline]["funcs"]):
                    dummy = func.derive(der_order)(z)
                impl[headline]["f_time"].append(time.time() - _t)
                print("derivative {}: {} s".format(der_order, impl[headline]["f_time"][der_order]))
        print("\nDifference")
        for der_order, _ in enumerate(impl[headlines[0]]["f_time"]):
            print("derivative {}: {} s".format(der_order, impl[headlines[1]]["f_time"][der_order] -
                                                  impl[headlines[0]]["f_time"][der_order]))

    def t_implementation(self, headline):
        import pyinduct.core as cr
        import pyinduct.placeholder as ph
        import pyinduct.registry as reg
        import pyinduct.shapefunctions as sh
        import pyinduct.simulation as sim
        import pyinduct.trajectory as tr
        import pyinduct.visualization as vis
        import numpy as np
        import pyqtgraph as pg

        sys_name = 'transport system'
        v = 10
        l = 5
        T = 5
        spat_domain = sim.Domain(bounds=(0, l), num=51)
        temp_domain = sim.Domain(bounds=(0, T), num=1e2)

        init_x = cr.Function(lambda z: 0)

        _a = time.time()
        if headline.startswith("New"):
            nodes, init_funcs = sh.cure_interval(sh.LagrangeNthOrder, spat_domain.bounds, node_count=len(spat_domain),
                                                 order=2)
        else:
            nodes, init_funcs = sh.cure_interval(sh.LagrangeSecondOrder, spat_domain.bounds,
                                                 node_count=len(spat_domain))
        _b = time.time()

        func_label = 'init_funcs'
        reg.register_base(func_label, init_funcs, overwrite=True)

        u = sim.SimulationInputSum([
            tr.SignalGenerator('square', np.array(temp_domain), frequency=0.3, scale=2, offset=4, phase_shift=1),
            tr.SignalGenerator('gausspulse', np.array(temp_domain), phase_shift=temp_domain[15]),
            tr.SignalGenerator('gausspulse', np.array(temp_domain), phase_shift=temp_domain[25], scale=-4),
            tr.SignalGenerator('gausspulse', np.array(temp_domain), phase_shift=temp_domain[35]),
            tr.SignalGenerator('gausspulse', np.array(temp_domain), phase_shift=temp_domain[60], scale=-2),
        ])

        _c = time.time()
        weak_form = sim.WeakFormulation([
            ph.IntegralTerm(ph.Product(ph.TemporalDerivedFieldVariable(func_label, 1), ph.TestFunction(func_label)),
                            spat_domain.bounds),
            ph.IntegralTerm(ph.Product(ph.FieldVariable(func_label), ph.TestFunction(func_label, order=1)),
                            spat_domain.bounds,
                            scale=-v),
            ph.ScalarTerm(ph.Product(ph.FieldVariable(func_label, location=l), ph.TestFunction(func_label, location=l)),
                          scale=v),
            ph.ScalarTerm(ph.Product(ph.Input(u), ph.TestFunction(func_label, location=0)),
                          scale=-v),
        ], name=sys_name)
        _d = time.time()

        initial_states = np.atleast_1d(init_x)

        _e = time.time()
        canonical_form = sim.parse_weak_formulation(weak_form)
        _f = time.time()

        state_space_form = canonical_form.convert_to_state_space()

        _g = time.time()
        q0 = np.array([sim.project_on_base(initial_state, reg.get_base(
            canonical_form.weights, 0)) for initial_state in
                       initial_states]).flatten()
        _h = time.time()

        sim_domain, q = sim.simulate_state_space(state_space_form, q0, temp_domain)

        temporal_order = min(initial_states.size - 1, 0)
        _i = time.time()
        eval_data = sim.process_sim_data(canonical_form.weights, q, sim_domain, spat_domain, temporal_order, 0,
                                         name=canonical_form.name)
        _j = time.time()

        if show_plots:
            win0 = pg.plot(np.array(eval_data[0].input_data[0]), u.get_results(eval_data[0].input_data[0]),
                           labels=dict(left='u(t)', bottom='t'), pen='b')
            win0.showGrid(x=False, y=True, alpha=0.5)
            vis.save_2d_pg_plot(win0, 'transport_system')
            win1 = vis.PgAnimatedPlot(eval_data, title=eval_data[0].name,
                                      save_pics=False, labels=dict(left='x(z,t)', bottom='z'))
            pg.QtGui.QApplication.instance().exec_()

        return (_a, _b, _c, _d, _e, _f, _g, _h, _i, _j), init_funcs

    def print_time(self, headline, time):
        a, b, c, d, e, f, g, h, i, j = tuple(time)

        print(headline + "\n" +
              "\t cure interval: {} s \n"
              "\t create weak form: {} s \n"
              "\t parse weak form: {} s \n"
              "\t initial weights: {} s \n"
              "\t process data: {} s \n"
              "".format(-a + b, -c + d, -e + f, -g + h, -i + j))
