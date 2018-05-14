from pyinduct.tests import test_examples

if __name__ == "__main__" or test_examples:
    import pyinduct as pi
    import numpy as np

    sys_name = 'euler bernoulli beam'

    # domains
    spat_domain = pi.Domain(bounds=(0, 1), num=9)
    temp_domain = pi.Domain(bounds=(0, 10), num=100)

    # physical properties
    height = .1  # [m]
    width = .1  # [m]
    e_module = 210e9  # [Pa]
    EI = 210e9 * (width * height**3)/12
    mu = 1  # [kg/m]

    nodes, init_funcs = pi.cure_interval(pi.LagrangeNthOrder,
                                         spat_domain.bounds,
                                         order=4,
                                         node_count=len(spat_domain))
    func_label = 'init_funcs'
    pi.register_base(func_label, init_funcs)

    class ImpulseExcitation(pi.SimulationInput):

        def _calc_output(self, **kwargs):
            t = kwargs["time"]
            if t < 1:
                value = 0
            elif t < 1.1:
                value = 1
            else:
                value = 0
            value = 1e-4*t**2
            return dict(output=value)

    u = ImpulseExcitation("Hammer")
    x = pi.FieldVariable(func_label)
    phi = pi.TestFunction(func_label)

    weak_form = pi.WeakFormulation([
        pi.ScalarTerm(pi.Product(x.derive(spat_order=3)(1), phi(1)), scale=EI),
        pi.ScalarTerm(pi.Product(pi.Input(u), phi(0)), scale=-EI),
        pi.ScalarTerm(pi.Product(x.derive(spat_order=2)(1), phi.derive(1)(1)), scale=-EI),
        pi.ScalarTerm(pi.Product(x.derive(spat_order=1)(1), phi.derive(2)(1)), scale=EI),
        pi.ScalarTerm(pi.Product(x(1), phi.derive(3)(1)), scale=-EI),
        pi.IntegralTerm(pi.Product(x, phi.derive(4)),
                        spat_domain.bounds,
                        scale=EI),
        pi.IntegralTerm(pi.Product(x.derive(temp_order=2), phi),
                        spat_domain.bounds,
                        scale=mu),
    ], name=sys_name)

    init_form = pi.Function.from_constant(0)
    init_form_dt = pi.Function.from_constant(0)
    initial_conditions = [init_form, init_form_dt]

    eval_data = pi.simulate_system(weak_form,
                                   initial_conditions,
                                   temp_domain,
                                   spat_domain,
                                   settings=dict(name="lsoda", nsteps=1e4))

    # pyqtgraph visualization
    # win0 = pg.plot(np.array(eval_data[0].input_data[0]).flatten(),
    #                u.get_results(eval_data[0].input_data[0]).flatten(),
    #                labels=dict(left='u(t)', bottom='t'), pen='b')
    # win0.showGrid(x=False, y=True, alpha=0.5)
    # vis.save_2d_pg_plot(win0, 'transport_system')

    win1 = pi.PgAnimatedPlot(eval_data, labels=dict(left='x(z,t)', bottom='z'))
    pi.show()

    pi.tear_down((func_label,),
                 (win1))

