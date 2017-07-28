"""
This script (*swm_observer.py*) shows how to simulate a distributed system with pyinduct, using the string
with mass example, which is illustrated in figure 5 and can described by the pde

.. math:: \\ddot{x}(z,t) = x''(z,t), \\qquad z\\in (0, 1)

and the boundary conditions

.. math:: \\ddot{x}(0,t) = m x'(0,t), \\qquad u(t) = x'(1,t)

where the deflection of the string is described by the field variable :math:`x(z,t)`.
The partial derivatives of :math:`x` w.r.t. time :math:`t` and space :math:`z` are denoted by means of dots
and primes, respectively. On the boundary by :math:`z=0` the mass :math:`m` is fixed at the string and on the
boundary by :math:`z=1` the deflection of the string can changed by use of the force (input variable) :math:`u(t)`.

Furthermore the flatness based controller and observer implementation is shown by this example. The design of the
controller and the observer is obtained from the paper

    * [Woi2012]: Frank Woittennek. „Beobachterbasierte Zustandsrückführungen für hyperbolische \
        verteiltparametrische Systeme“.In: Automatisierungstechnik 60.8 (2012).

The control law (equation 29) and the observer (equation 41) from [Woi2012] were simply tipped off. You can find
the implementation in the functions :py:func:`build_control_law` and :py:func:`build_observer_can`.
"""

# (sphinx directive) start import
import pyinduct as pi
import pyinduct.control as ct
from pyinduct.parabolic.control import scale_equation_term_list
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg


# (sphinx directive) end import

def build_system_equations(approx_label, input_vector, params):
    """
    The boundary conditions can considered through integration by parts of the weak formulation:

    .. math::
        :nowrap:

        \\begin{align*}
            \\langle \\ddot{x}(z,t), \\psi_j(z) \\rangle &= \\langle x''(z,t), \\psi_j(z) \\rangle \\\\
            &= x'(1,t)\\psi_j(1) - x'(0)\\psi_j(0) - \\langle x'(z,t), \\psi'_j(z) \\rangle \\\\
            &= u(t)\\psi_j(1) - m\\ddot{x}(0,t)\\psi_j(0) - \\langle x'(z,t), \\psi'_j(z) \\rangle ,
            \\qquad j = 1,...,N.
        \\end{align*}

    The field variable is approximated with the functions :math:`\\varphi_i, i=1,...,N` which are registered
    under the label *approx_label*

    .. math:: x(z,t) = \\sum_{i=1}^{N}c_i(t)\\varphi_i(z).

    In order to derive a numerical scheme the galerkin method is used, meaning :math:`\\psi_i = \\varphi_i, i=1,...,N`.

    Args:
        approx_label (string): Shapefunction label for approximation.
        input_vector (:py:class:`pyinduct.simulation.SimulationInputVector`): Holds the input variable.
        params: Python class with the member *m* (mass).

    Returns:
        :py:class:`pyinduct.simulation.StateSpace`: State space model

    """
    # (sphinx directive) start build_system_state_space
    limits = (0, 1)
    x = pi.FieldVariable(approx_label)
    psi = pi.TestFunction(approx_label)

    wf = pi.WeakFormulation(
        [
            pi.IntegralTerm(pi.Product(x.derive(temp_order=2), psi), limits=limits),
            pi.IntegralTerm(pi.Product(x.derive(spat_order=1), psi.derive(1)), limits=limits),
            pi.ScalarTerm(pi.Product(x(0).derive(temp_order=2), psi(0)), scale=params.m),
            pi.ScalarTerm(pi.Product(pi.Input(input_vector, index=0), psi(1)), scale=-1),
        ],
        name=approx_label
    )

    return [wf]
    # (sphinx directive) end build_system_state_space


def build_controller(approx_label, params):
    """
    The control law from [Woi2012] (equation 29)

    .. math::
        :nowrap:

        \\begin{align*}
            u(t) = &-\\frac{1-\\alpha}{1+\\alpha}x_2(1) +
            \\frac{(1-mk_1)\\bar{y}'(1) - \\alpha(1+mk_1)\\bar{y}'(-1)}{1+\\alpha} \\\\
            \\hphantom{=} &-\\frac{mk_0}{1+\\alpha}(\\bar{y}(1) + \\alpha\\bar{y}(-1))
        \\end{align*}

    is simply tipped off in this function, whereas

    .. math::
        :nowrap:

        \\begin{align*}
            \\bar{y}(\\theta) &=  \\left\\{\\begin{array}{lll}
                 \\xi_1 + m(1-e^{-\\theta/m})\\xi_2 +
                 \int_0^\\theta (1-e^{-(\\theta-\\tau)/m}) (x_1'(\\tau) + x_2(\\tau)) \, dz
                 & \\forall & \\theta\\in[-1, 0) \\\\
                 \\xi_1 + m(e^{\\theta/m}-1)\\xi_2 +
                 \int_0^\\theta (e^{(\\theta-\\tau)/m}-1) (x_1'(-\\tau) - x_2(-\\tau)) \, dz
                 & \\forall & \\theta\\in[0, 1]
            \\end{array}\\right. \\\\
            \\bar{y}'(\\theta) &=  \\left\\{\\begin{array}{lll}
                 e^{-\\theta/m}\\xi_2 + \\frac{1}{m}
                 \int_0^\\theta e^{-(\\theta-\\tau)/m} (x_1'(\\tau) + x_2(\\tau)) \, dz
                 & \\forall & \\theta\\in[-1, 0) \\\\
                 e^{\\theta/m}\\xi_2 + \\frac{1}{m}
                 \int_0^\\theta e^{(\\theta-\\tau)/m} (x_1'(-\\tau) - x_2(-\\tau)) \, dz
                 & \\forall & \\theta\\in[0, 1].
            \\end{array}\\right.
        \\end{align*}

    Args:
        approx_label (string): Shapefunction label for approximation.
        params: Python class with the members:

            - *m* (mass)
            - *k1_ct*, *k2_ct*, *alpha_ct* (controller parameters)

    Returns:
        :py:class:`pyinduct.simulation.FeedbackLaw`: Control law

    """
    # (sphinx directive) start build_control_law
    x = pi.FieldVariable(approx_label)
    dz_x1 = x.derive(spat_order=1)
    x2 = x.derive(temp_order=1)
    xi1 = x(0)
    xi2 = x(0).derive(temp_order=1)

    scalar_scale_funcs = [pi.Function(lambda theta: params.m * (1 - np.exp(-theta / params.m))),
                          pi.Function(lambda theta: params.m * (-1 + np.exp(theta / params.m))),
                          pi.Function(lambda theta: np.exp(-theta / params.m)),
                          pi.Function(lambda theta: np.exp(theta / params.m))]

    pi.register_base("int_scale1", pi.Base(pi.Function(lambda tau: 1 - np.exp(-(1 - tau) / params.m))))
    pi.register_base("int_scale2", pi.Base(pi.Function(lambda tau: -1 + np.exp((-1 + tau) / params.m))))
    pi.register_base("int_scale3", pi.Base(pi.Function(lambda tau: np.exp(-(1 - tau) / params.m) / params.m)))
    pi.register_base("int_scale4", pi.Base(pi.Function(lambda tau: np.exp((-1 + tau) / params.m) / params.m)))

    limits = (0, 1)
    y_bar_plus1 = [pi.ScalarTerm(xi1),
                   pi.ScalarTerm(xi2, scale=scalar_scale_funcs[0](1)),
                   pi.IntegralTerm(pi.Product(dz_x1, pi.ScalarFunction("int_scale1")), limits=limits),
                   pi.IntegralTerm(pi.Product(pi.ScalarFunction("int_scale1"), x2), limits=limits)]
    y_bar_minus1 = [pi.ScalarTerm(xi1),
                    pi.ScalarTerm(xi2, scale=scalar_scale_funcs[1](-1)),
                    pi.IntegralTerm(pi.Product(pi.ScalarFunction("int_scale2"), dz_x1), limits=limits, scale=-1),
                    pi.IntegralTerm(pi.Product(pi.ScalarFunction("int_scale2"), x2), limits=limits)]
    dz_y_bar_plus1 = [pi.ScalarTerm(xi2, scale=scalar_scale_funcs[2](1)),
                      pi.IntegralTerm(pi.Product(pi.ScalarFunction("int_scale3"), dz_x1), limits=limits),
                      pi.IntegralTerm(pi.Product(pi.ScalarFunction("int_scale3"), x2), limits=limits)]
    dz_y_bar_minus1 = [pi.ScalarTerm(xi2, scale=scalar_scale_funcs[3](-1)),
                       pi.IntegralTerm(pi.Product(pi.ScalarFunction("int_scale4"), dz_x1), limits=limits, scale=-1),
                       pi.IntegralTerm(pi.Product(pi.ScalarFunction("int_scale4"), x2), limits=limits)]

    return pi.Controller(pi.WeakFormulation(scale_equation_term_list(
        [pi.ScalarTerm(x2(1), scale=-(1 - params.alpha_ct))] +
        scale_equation_term_list(dz_y_bar_plus1, factor=(1 - params.m * params.k1_ct)) +
        scale_equation_term_list(dz_y_bar_minus1, factor=-params.alpha_ct * (1 + params.m * params.k1_ct)) +
        scale_equation_term_list(y_bar_plus1, factor=-params.m * params.k0_ct) +
        scale_equation_term_list(y_bar_minus1, factor=-params.alpha_ct * params.m * params.k0_ct),
        factor=(1 + params.alpha_ct) ** -1
    ), name=approx_label))
    # (sphinx directive) end build_control_law


def build_observer_org(sys_approx_label, obs_approx_label, input_vector, params):
    """
    """
    limits = (0, 1)
    psi = pi.TestFunction(obs_approx_label)

    # system variable
    x = pi.FieldVariable(sys_approx_label)
    # observer variables
    x1 = pi.FieldVariable(obs_approx_label)
    x2 = x1.derive(temp_order=1)
    xi1 = x1(0)
    xi2 = x2(0)
    # obserer gain
    L = np.array([-params.m / 2 * (params.alpha_ob - 1) * params.k0_ob,
                  0,
                  1 / 2 * (params.alpha_ob + 1) * params.k1_ob,
                  1 / 2 * (params.alpha_ob + 1) * params.k0_ob])

    obs_err = ct.Controller(pi.WeakFormulation(
        [
            pi.ScalarTerm(x1(0)),
            pi.ScalarTerm(x(0), scale=-1),
        ],
        name="observer error"))
    d_obs_err = ct.Controller(pi.WeakFormulation(
        [
            pi.ScalarTerm(x1.derive(temp_order=1)(0)),
            pi.ScalarTerm(x.derive(temp_order=1)(0), scale=-1),
        ],
        name="dt observer error"))
    input_vector.append([obs_err, d_obs_err])

    wf = pi.WeakFormulation(
        [
            pi.IntegralTerm(pi.Product(x1.derive(temp_order=1), psi), limits=limits, scale=-1),
            pi.IntegralTerm(pi.Product(x2.derive(temp_order=1), psi), limits=limits, scale=-1),
            pi.ScalarTerm(pi.Product(xi1.derive(temp_order=1), psi(0)), scale=-1),
            pi.ScalarTerm(pi.Product(xi2.derive(temp_order=1), psi(0)), scale=-1),
            pi.IntegralTerm(pi.Product(x2, psi), limits=limits),
            # shift of <x1'', psi>
            pi.ScalarTerm(pi.Product(pi.Input(input_vector, index=0), psi(1))),
            pi.ScalarTerm(pi.Product(x1.derive(spat_order=1)(0), psi(0)), scale=-1),
            pi.IntegralTerm(pi.Product(x1.derive(spat_order=1), psi.derive(order=1)), limits=limits, scale=-1),
            # shift end
            pi.ScalarTerm(pi.Product(xi2, psi(0))),
            pi.ScalarTerm(pi.Product(x1.derive(spat_order=1)(0), psi(0)), scale=1 / params.m),
            # observer gain
            # from <\dot xi1, psi(0)> and the bc \xi1 = y + (alpha-1) \tilde y
            # pi.ScalarTerm(pi.Product(pi.Input(input_vector, index=2), psi(0)), scale=-(params.alpha_ob-1)),
            pi.ScalarTerm(pi.Product(pi.Input(input_vector, index=2), psi(0)), scale=0),
            # from <\dot x1, psi> = <x2, psi> - <L[0] \tilde y, psi>
            pi.IntegralTerm(pi.Product(pi.Input(input_vector, index=1), psi), limits=limits, scale=-L[0]),
            # from <\dot xi1, psi(0)> = <xi2, psi(0)> - <L[2] \tilde y, psi(0)>
            pi.ScalarTerm(pi.Product(pi.Input(input_vector, index=1), psi(0)), scale=-L[2]),
            # from <\dot xi2, psi(0)> = m <x1'(0), psi(0)> - <L[3] \tilde y, psi(0)>
            pi.ScalarTerm(pi.Product(pi.Input(input_vector, index=1), psi(0)), scale=-L[3]),
        ],
        name=obs_approx_label
    )

    return [wf]


def build_observer_can(sys_approx_label, obs_approx_label, input_vector, params):
    """
    The observer from [Woi2012] (equation 41)

    .. math::
        :nowrap:

        \\begin{align*}
            \\dot{\\hat{\\eta}}_3(\\theta,t) &= -\\hat{\\eta}_3'(\\theta,t)-\\frac{2}{m}(h(\\theta)-1)\\theta u(t)
                - m^{-1} \\hat{y}(t) \\\\
            &\\hphantom{=}-(k_0(1-\\theta)+k_1-m^{-1})\\tilde{y}(t) \\\\
            \\dot{\\hat{\\eta}}_2(t) &= \\hat{\\eta}_2(t) + \\frac{2}{m}u(t)-((1+\\alpha)k_1+2k_0)\\tilde{y}(t) \\\\
            \\dot{\\hat{\\eta}}_1(t) &= \\frac{2}{m}u(t) - (1+\\alpha)k_0\\tilde{y}(t)
        \\end{align*}

    is simply tipped off in this function. The boundary condition (equation 41d)

    .. math:: \\hat{\\eta}_3(-1,t) = \\hat{\\eta}_2(t) -\hat{y}(t)-(\\alpha-1)\\tilde{y}(t)

    is considered through integration by parts of the term
    :math:`-\\langle\\hat{\\eta}_3'(\\theta),\\psi_j(\\theta)\\rangle` from the weak formulation of equation 41a:

    .. math::
        :nowrap:

        \\begin{align*}
            -\\langle\\hat{\\eta}_3'(\\theta),\\psi_j(\\theta)\\rangle &=
            -\\hat{\\eta}_3(1)\\psi_j'(1) + \\hat{\\eta}_3(-1)\\psi_j'(-1)
            \\langle\\hat{\\eta}_3(\\theta),\\psi_j'(\\theta)\\rangle.
        \\end{align*}

    Args:
        sys_approx_label (string): Shapefunction label for system approximation.
        obs_approx_label (string): Shapefunction label for observer approximation.
        input_vector (:py:class:`pyinduct.simulation.SimulationInputVector`): Holds the input variable.
        params: Python class with the members:

            - *m* (mass)
            - *k1_ob*, *k2_ob*, *alpha_ob* (observer parameters)

    Returns:
        :py:class:`pyinduct.simulation.Observer`: Observer
    """
    # (sphinx directive) start build_observer_can
    limits = (-1, 1)

    def heavi(z):
        return 0 if z < 0 else (0.5 if z == 0 else 1)

    pi.register_base("obs_scale1",
                     pi.Base(pi.Function(lambda z: -2 / params.m * (heavi(z) - 1) * z, domain=limits)))
    pi.register_base("obs_scale2",
                     pi.Base(
                         pi.Function(lambda z: -(params.k0_ob * (1 - z) + params.k1_ob - 1 / params.m), domain=limits)))
    obs_scale1 = pi.ScalarFunction("obs_scale1")
    obs_scale2 = pi.ScalarFunction("obs_scale2")

    eta1 = pi.FieldVariable(obs_approx_label[0])
    eta2 = pi.FieldVariable(obs_approx_label[1])
    eta3 = pi.FieldVariable(obs_approx_label[2])
    psi = pi.TestFunction(obs_approx_label[2])

    obs_err = ct.Controller(pi.WeakFormulation(
        [
            pi.ScalarTerm(pi.FieldVariable(sys_approx_label, location=0), scale=-1),
            pi.ScalarTerm(eta3(-1).derive(spat_order=1), scale=-params.m / 2),
            pi.ScalarTerm(eta3(1).derive(spat_order=1), scale=-params.m / 2),
            pi.ScalarTerm(eta1(0), scale=-params.m / 2),
        ],
        name="nf observer error"))
    input_vector.append([obs_err])

    d_eta1 = pi.WeakFormulation(
        [
            pi.ScalarTerm(eta1(0).derive(temp_order=1), scale=-1),
            pi.ScalarTerm(pi.Input(input_vector, index=0), scale=2 / params.m),
            pi.ScalarTerm(pi.Input(input_vector, index=1), scale=-(1 + params.alpha_ob) * params.k0_ob)
        ],
        name=obs_approx_label[0]
    )
    d_eta2 = pi.WeakFormulation(
        [
            pi.ScalarTerm(eta2(0).derive(temp_order=1), scale=-1),
            # index error in paper
            pi.ScalarTerm(eta1(0)),
            pi.ScalarTerm(pi.Input(input_vector, index=0), scale=2 / params.m),
            pi.ScalarTerm(pi.Input(input_vector, index=1),
                          scale=-(1 + params.alpha_ob) * params.k1_ob - 2 * params.k0_ob)
        ],
        name=obs_approx_label[1]
    )
    d_eta3 = pi.WeakFormulation(
        [
            pi.IntegralTerm(pi.Product(eta3.derive(temp_order=1), psi), limits=limits, scale=-1),
            # sign error in paper
            pi.IntegralTerm(pi.Product(pi.Product(obs_scale1, psi), pi.Input(input_vector, index=0)), limits=limits,
                            scale=-1),
            pi.IntegralTerm(pi.Product(pi.Product(obs_scale2, psi), pi.Input(input_vector, index=1)), limits=limits),
            # \hat y
            pi.IntegralTerm(pi.Product(eta3(-1).derive(spat_order=1), psi), limits=limits, scale=1 / 2),
            pi.IntegralTerm(pi.Product(eta3(1).derive(spat_order=1), psi), limits=limits, scale=1 / 2),
            pi.IntegralTerm(pi.Product(eta1, psi), limits=limits, scale=1 / 2),
            # shift
            pi.IntegralTerm(pi.Product(eta3, psi.derive(1)), limits=limits),
            pi.ScalarTerm(pi.Product(eta3(1), psi(1)), scale=-1),
            # bc
            pi.ScalarTerm(pi.Product(psi(-1), eta2(0))),
            pi.ScalarTerm(pi.Product(pi.Input(input_vector, index=1), psi(-1)), scale=1 - params.alpha_ob),
            # bc \hat y
            pi.ScalarTerm(pi.Product(eta3(-1).derive(spat_order=1), psi(-1)), params.m / 2),
            pi.ScalarTerm(pi.Product(eta3(1).derive(spat_order=1), psi(-1)), params.m / 2),
            pi.ScalarTerm(pi.Product(psi(-1), eta1(1)), params.m / 2),
        ],
        name=obs_approx_label[2]
    )

    return [d_eta1, d_eta2, d_eta3]
    # (sphinx directive) end build_observer_can


class SecondOrderFeedForward(pi.SimulationInput):
    def __init__(self, desired_handle, params):
        pi.SimulationInput.__init__(self)
        self._y = desired_handle
        self._params = params

    def _calc_output(self, **kwargs):
        y_p = self._y(kwargs["time"] + 1)
        y_m = self._y(kwargs["time"] - 1)
        f = + self._params.k0_ct * (y_p[0] + self._params.alpha_ct * y_m[0]) \
            + self._params.k1_ct * (y_p[1] + self._params.alpha_ct * y_m[1]) \
            + y_p[2] + self._params.alpha_ct * y_m[2]

        return dict(output=self._params.m / (1 + self._params.alpha_ct) * f)


class Parameters:
    def __init__(self):
        # system parameters
        self.m = 1.0
        self._tau = 1.0  # hard written to 1 in this example script
        self._sigma = 1.0  # hard written to 1 in this example script

        # controller parameters
        self.k0_ct = 10
        self.k1_ct = 10
        self.alpha_ct = 0

        # observer parameters
        self.k0_ob = 10
        self.k1_ob = 10
        self.alpha_ob = 0

    def get_tau(self):
        return self._tau

    def get_sigma(self):
        return self._sigma

    def set_tau_or_sigma(self, tau_or_sigma):
        if tau_or_sigma != 1:
            raise ValueError("Parameters tau and sigma are hard written to 1 in this example script!")

    tau = property(get_tau, set_tau_or_sigma)
    sigma = property(get_sigma, set_tau_or_sigma)


# (sphinx directive) start actual script
if __name__ == "__main__":

    # which observer
    nf_observer = True

    # temporal and spatial domain specification
    t_end = 8
    temp_domain = pi.Domain(bounds=(0, t_end), step=.01)
    spat_domain = pi.Domain(bounds=(0, 1), step=.01)

    # system/observer parameters
    params = Parameters()

    # labels
    sys_lbl = "sys"
    sys_ctrl_lbl = "ctrl"
    obs_can_lbl = ["eta1", "eta2", "eta3"]
    obs_org_lbl = "obs_org"

    # initial function
    sys_nodes, sys_funcs = pi.cure_interval(pi.LagrangeNthOrder, spat_domain.bounds, node_count=11, order=1)
    ctrl_nodes, ctrl_funcs = pi.cure_interval(pi.LagrangeNthOrder, spat_domain.bounds, node_count=13, order=1)
    pi.register_base(sys_lbl, sys_funcs)
    pi.register_base(sys_ctrl_lbl, ctrl_funcs)
    spat_domains = {sys_lbl: sys_nodes}
    if nf_observer:
        obs_lbl = obs_can_lbl
        pi.register_base(obs_lbl[0], pi.Base(pi.Function.from_constant(1)))
        pi.register_base(obs_lbl[1], pi.Base(pi.Function.from_constant(1)))
        obs_nodes, obs_funcs = pi.cure_interval(pi.LagrangeNthOrder, (-1, 1), node_count=25, order=4)
        pi.register_base(obs_lbl[2], obs_funcs)
        spat_domains.update({obs_lbl[0]: pi.Domain((0, 0), num=1)})
        spat_domains.update({obs_lbl[1]: pi.Domain((0, 0), num=1)})
        spat_domains.update({obs_lbl[2]: obs_nodes})
    else:
        obs_lbl = obs_org_lbl
        obs_nodes, obs_funcs = pi.cure_interval(pi.LagrangeNthOrder, spat_domain.bounds, node_count=9, order=2)
        pi.register_base(obs_lbl, obs_funcs)
        spat_domains.update({obs_lbl: obs_nodes})

    # system input
    if 1:
        # trajectory for the new input (closed_loop_traj)
        smooth_transition = pi.SmoothTransition((0, 1), (2, 4), method="poly", differential_order=2)
        closed_loop_traj = SecondOrderFeedForward(smooth_transition, params)
        # controller
        ctrl = build_controller(sys_lbl, params)
        u = pi.SimulationInputSum([closed_loop_traj, ctrl])
    else:
        # trajectory for the original input (open_loop_traj)
        open_loop_traj = pi.FlatString(y0=0, y1=1, z0=spat_domain.bounds[0], z1=spat_domain.bounds[1],
                                       t0=1, dt=3, params=params)
        # u = pi.SimulationInputSum([open_loop_traj])
        u = pi.SimulationInputSum([pi.ConstantTrajectory(0)])
    input_vector = pi.SimulationInputVector([u])

    # system equations
    sys_wf = build_system_equations(sys_lbl, input_vector, params)
    sys_ic = {sys_lbl: np.array([pi.Function.from_constant(0, domain=(0, 1)),
                                 pi.Function.from_constant(0, domain=(0, 1))])}

    # observer equations
    if nf_observer:
        obs_wf = build_observer_can(sys_lbl, obs_lbl, input_vector, params)
        obs_ics = {obs_lbl[0]: np.array([pi.Function.from_constant(0)]),
                   obs_lbl[1]: np.array([pi.Function.from_constant(0)]),
                   obs_lbl[2]: np.array([pi.Function.from_constant(0, domain=(-1, 1))])}
    else:
        obs_wf = build_observer_org(sys_lbl, obs_lbl, input_vector, params)
        obs_ics = {obs_lbl: np.array([pi.Function.from_constant(0, domain=(0, 1)),
                                      pi.Function.from_constant(0, domain=(0, 1))])}

    # simulation
    canonical_equations = pi.parse_weak_formulations(sys_wf + obs_wf)
    state_space = pi.create_state_space(canonical_equations)
    ics = dict(list(sys_ic.items()) + list(obs_ics.items()))
    initial_states = pi.project_on_bases(canonical_equations, ics)
    sim_domain, weights = pi.simulate_state_space(state_space, initial_states, temp_domain)

    # evaluate data
    x_data = pi.get_sim_results(temp_domain, spat_domains, weights, state_space, labels=[sys_lbl])[0]
    if nf_observer:
        eta1_data = pi.get_sim_results(temp_domain, {obs_lbl[0]: pi.Domain((0, 1), num=1e1)},
                                       weights, state_space, labels=[obs_lbl[0]],
                                       derivative_orders={obs_lbl[0]: (0, 0)})[0]
        dz_et3_m1_0 = pi.get_sim_results(temp_domain, {obs_lbl[2]: pi.Domain((-1, 0), num=1e1)},
                                         weights, state_space, labels=[obs_lbl[2]],
                                         derivative_orders={obs_lbl[2]: (0, 1)})[1]
        dz_et3_0_p1 = pi.get_sim_results(temp_domain, {obs_lbl[2]: pi.Domain((0, 1), num=1e1)},
                                         weights, state_space, labels=[obs_lbl[2]],
                                         derivative_orders={obs_lbl[2]: (0, 1)})[1]
        x_obs_data = pi.EvalData(eta1_data.input_data, -params.m / 2 * (
            dz_et3_m1_0.output_data + np.fliplr(dz_et3_0_p1.output_data) + eta1_data.output_data
        ))
    else:
        x_obs_data = pi.get_sim_results(sim_domain, spat_domains, weights, state_space, labels=[obs_lbl],
                                        derivative_orders={obs_lbl: (0, 0)})[0]

    # animation
    plot1 = pi.PgAnimatedPlot([x_data, x_obs_data])
    plot2 = pi.PgSurfacePlot(x_data)
    plot3 = pi.PgSurfacePlot(x_obs_data)
    pg.QtGui.QApplication.instance().exec_()
    pi.MplSlicePlot([x_data, x_obs_data], spatial_point=0)
    plt.show()
