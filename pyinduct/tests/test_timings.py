"""
Test benches for the computational efficiency of toolbox routines.
"""
import time
import unittest

import numpy as np
import pyinduct as pi


def simulation_benchmark(spat_domain, settings):
    """
    This benchmark covers a typical simulation.

    Args:
        spat_domain (:py:class:`.Domain`): spatial domain for the simulation
        settings (dict): settings to use for simulation run

    Returns:
        timings
    """
    sys_name = 'transport system'
    v = 10

    temp_domain = pi.Domain(bounds=(0, 5), num=100)

    init_x = pi.Function(lambda z: 0)

    _a = time.clock()
    nodes = pi.Domain(settings["bounds"], settings["num"])
    cls = settings.get("shapefunction_class")
    base = cls.cure_interval(nodes, **settings)
    _b = time.clock()

    func_label = 'base'
    pi.register_base(func_label, base)

    u = pi.SimulationInputSum([
        pi.SignalGenerator('square', temp_domain.points, frequency=0.3, scale=2,
                           offset=4, phase_shift=1),
    ])

    _c = time.clock()
    weak_form = pi.WeakFormulation([
        pi.IntegralTerm(pi.Product(
            pi.FieldVariable(func_label, order=(1, 0)),
            pi.TestFunction(func_label)),
            spat_domain.bounds),
        pi.IntegralTerm(pi.Product(
            pi.FieldVariable(func_label),
            pi.TestFunction(func_label, order=1)),
            spat_domain.bounds, scale=-v),
        pi.ScalarTerm(pi.Product(
            pi.FieldVariable(func_label, location=spat_domain.bounds[-1]),
            pi.TestFunction(func_label, location=spat_domain.bounds[-1])),
            scale=v),
        pi.ScalarTerm(pi.Product(
            pi.Input(u),
            pi.TestFunction(func_label, location=0)), scale=-v),
    ],
        name=sys_name)
    _d = time.clock()

    initial_states = np.atleast_1d(init_x)

    _e = time.clock()
    can_eq = pi.parse_weak_formulation(weak_form)
    _f = time.clock()

    state_space_form = pi.create_state_space(can_eq)

    _g = time.clock()
    q0 = np.array([pi.project_on_base(initial_state, pi.get_base(can_eq.dominant_lbl))
                   for initial_state in initial_states]).flatten()
    _h = time.clock()

    sim_domain, q = pi.simulate_state_space(state_space_form, q0, temp_domain)

    temporal_order = min(initial_states.size - 1, 0)
    _i = time.clock()
    eval_data = pi.get_sim_result(can_eq.dominant_lbl, q, sim_domain, spat_domain, temporal_order, 0,
                                  name=can_eq.dominant_form.name)
    _j = time.clock()

    pi.deregister_base("base")

    return _b - _a, _d - _c, _f - _e, _h - _g, _j - _i


def product_benchmark(base):
    def projection_func(z):
        return np.sin(2 * z) + np.exp(z)

    _t = time.clock()
    res = pi.calculate_scalar_product_matrix(pi.dot_product_l2, base, base)
    _t_mat = time.clock() - _t

    _t = time.clock()
    res = pi.project_on_base(pi.Function(projection_func), base)
    _t_proj = time.clock() - _t

    return _t_mat, _t_proj


class ShapeFunctionTestBench(unittest.TestCase):
    """
    Compare LagrangeNthOrder with LagrangeSecondOrder (have a look at terminal
    output).

    When it succeeds to get positive values (at least a few) under the
    "Difference" headline by the transport system example, too, you can delete:
    - this test case
    - LagrangeFirstOrder
    - LagrangeSecondOrder
    """

    def setUp(self):
        self.node_cnt = 51
        self.domain = pi.Domain(bounds=(0, 1), num=1000)

        # first one is used as reference
        if True:
            self.candidates = [
                dict(shapefunction_class=pi.LagrangeFirstOrder,
                     bounds=self.domain.bounds,
                     num=self.node_cnt),
                dict(shapefunction_class=pi.LagrangeNthOrder,
                     bounds=self.domain.bounds,
                     num=self.node_cnt,
                     order=1), ]
        else:
            self.candidates = [
                dict(shapefunction_class=pi.LagrangeSecondOrder,
                     bounds=self.domain.bounds,
                     num=self.node_cnt),
                dict(shapefunction_class=pi.LagrangeNthOrder,
                     bounds=self.domain.bounds,
                     num=self.node_cnt,
                     order=2),
            ]
        print("comparing {} (1) against {} (2)".format(
            *[candidate["shapefunction_class"]
              for candidate in self.candidates]))

    def test_simulation(self):
        print(">>> transport system example speed test: \n")

        timings = []
        n_iteration = 3
        for i in range(n_iteration):
            print("running round {} of {}".format(i, n_iteration))
            results = []
            for candidate in self.candidates:
                results.append(simulation_benchmark(self.domain, candidate))

            timings.append(results)

        res = np.array(timings)
        mean = np.mean(res, axis=0)
        for idx in range(len(self.candidates)):
            self.print_time(
                "means of {} rounds for {} in [s]:".format(
                    n_iteration,
                    self.candidates[idx]["shapefunction_class"]),
                mean[idx])

        # process results
        diff = np.subtract(mean[1], mean[0])
        try:
            frac = 100 * diff / mean[0]
            self.print_time("relative difference  (2-1)/1 in [%]", frac)
        except FloatingPointError as e:
            print("relative difference not available")
        finally:
            self.print_time("absolute difference  (2-1) in [s]", diff)

    def test_evaluation(self):
        print(">>> evaluation speed test:")

        timings = []
        n_iteration = 3
        for i in range(n_iteration):
            print("running round {} of {}".format(i, n_iteration))
            results = []
            for candidate in self.candidates:
                nodes = pi.Domain(candidate["bounds"], candidate["num"])
                cls = candidate.get("shapefunction_class")
                base = cls.cure_interval(nodes, **candidate)
                results.append(product_benchmark(base))

            timings.append(results)

        # process results
        res = np.array(timings)
        mean = np.mean(res, axis=0)
        diff = np.subtract(mean[1], mean[0])
        try:
            frac = 100 * diff / mean[0]
            print("relative difference (2-1)/1 in [%]:\n\t {}".format(frac))
        except FloatingPointError as e:
            print("relative difference not available")
        finally:
            print("absolute difference (2-1) in [s]:\n\t {}".format(diff))

    def print_time(self, headline, times):
        print(headline + "\n" + "\t cure interval:    {}\n"
                                "\t create weak form: {}\n"
                                "\t parse weak form:  {}\n"
                                "\t initial weights:  {}\n"
                                "\t process data:     {}\n"
                                "".format(*times))
