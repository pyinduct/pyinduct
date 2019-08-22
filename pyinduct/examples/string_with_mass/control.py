from copy import deepcopy
from pyinduct.examples.string_with_mass.system import *
from pyinduct.parabolic.control import scale_equation_term_list



class SwmObserverError(pi.StateFeedback):
    """
    For a smooth fade-in of the observer error.

    Args:
        control_law (:py:class:`.WeakFormulation`): Function handle that
            calculates the control output if provided with correct weights.
        smooth (array-like): Arguments for :py:class:`.SmoothTransition`
    """

    def __init__(self, control_law, smooth=None):
        super().__init__(control_law)

        self.smooth = smooth
        if smooth is not None:
            self.smooth_transition = pi.SmoothTransition(
                smooth[0], smooth[1], smooth[2], smooth[3])
        else:
            self.smooth_transition = lambda t: [1]

    def _calc_output(self, **kwargs):
        res = super(SwmObserverError, self)._calc_output(**kwargs)

        return dict(output=res["output"] * self.smooth_transition(kwargs["time"])[0])


class SecondOrderFeedForward(pi.SimulationInput):
    def __init__(self, desired_handle):
        pi.SimulationInput.__init__(self)
        self._y = desired_handle

    def _calc_output(self, **kwargs):
        y_p = self._y(kwargs["time"] + 1)
        y_m = self._y(kwargs["time"] - 1)
        f = (+ ctrl_gain.k0 * (y_p[0] + ctrl_gain.alpha * y_m[0])
             + ctrl_gain.k1 * (y_p[1] + ctrl_gain.alpha * y_m[1])
             + y_p[2] + ctrl_gain.alpha * y_m[2])

        return dict(output=param.m / (1 + ctrl_gain.alpha) * f)


def build_controller(sys_lbl, ctrl_lbl):
    r"""
    The control law from [Woi2012] (equation 29)

    .. math::
        :nowrap:

        \begin{align*}
            u(t) = &-\frac{1-\alpha}{1+\alpha}x_2(1) +
            \frac{(1-mk_1)\bar{y}'(1) - \alpha(1+mk_1)\bar{y}'(-1)}{1+\alpha} \\
            \hphantom{=} &-\frac{mk_0}{1+\alpha}(\bar{y}(1) + \alpha\bar{y}(-1))
        \end{align*}

    is simply tipped off in this function, whereas

    .. math::
        :nowrap:

        \begin{align*}
            \bar{y}(\theta) &=  \left\{\begin{array}{lll}
                 \xi_1 + m(1-e^{-\theta/m})\xi_2 +
                 \int_0^\theta (1-e^{-(\theta-\tau)/m}) (x_1'(\tau) + x_2(\tau)) \, dz
                 & \forall & \theta\in[-1, 0) \\
                 \xi_1 + m(e^{\theta/m}-1)\xi_2 +
                 \int_0^\theta (e^{(\theta-\tau)/m}-1) (x_1'(-\tau) - x_2(-\tau)) \, dz
                 & \forall & \theta\in[0, 1]
            \end{array}\right. \\
            \bar{y}'(\theta) &=  \left\{\begin{array}{lll}
                 e^{-\theta/m}\xi_2 + \frac{1}{m}
                 \int_0^\theta e^{-(\theta-\tau)/m} (x_1'(\tau) + x_2(\tau)) \, dz
                 & \forall & \theta\in[-1, 0) \\
                 e^{\theta/m}\xi_2 + \frac{1}{m}
                 \int_0^\theta e^{(\theta-\tau)/m} (x_1'(-\tau) - x_2(-\tau)) \, dz
                 & \forall & \theta\in[0, 1].
            \end{array}\right.
        \end{align*}

    Args:
        approx_label (string): Shapefunction label for approximation.

    Returns:
        :py:class:`.StateFeedback`: Control law
    """
    x1 = pi.FieldVariable(ctrl_lbl + "_1_ctrl")
    x2 = pi.FieldVariable(ctrl_lbl + "_2_ctrl")
    x2_sys = pi.FieldVariable(sys_lbl + "_2_visu")
    xi1 = pi.FieldVariable(ctrl_lbl + "_3_ctrl")(0)
    xi2 = pi.FieldVariable(ctrl_lbl + "_4_ctrl")(0)
    dz_x1 = x1.derive(spat_order=1)

    limits_cf = (-1, 1)
    scalar_scale_funcs = [
        pi.Function(lambda theta: param.m * (1 - np.exp(-theta / param.m)),
                    domain=limits_cf),
        pi.Function(lambda theta: param.m * (-1 + np.exp(theta / param.m)),
                    domain=limits_cf),
        pi.Function(lambda theta: np.exp(-theta / param.m), domain=limits_cf),
        pi.Function(lambda theta: np.exp(theta / param.m), domain=limits_cf)]

    limits = (0, 1)
    pi.register_base("int_scale1", pi.Base(pi.Function(
        lambda tau: 1 - np.exp(-(1 - tau) / param.m), domain=limits)))
    pi.register_base("int_scale2", pi.Base(pi.Function(
        lambda tau: -1 + np.exp((-1 + tau) / param.m), domain=limits)))
    pi.register_base("int_scale3", pi.Base(pi.Function(
        lambda tau: np.exp(-(1 - tau) / param.m) / param.m, domain=limits)))
    pi.register_base("int_scale4", pi.Base(pi.Function(
        lambda tau: np.exp((-1 + tau) / param.m) / param.m, domain=limits)))

    y_bar_plus1 = [
        pi.ScalarTerm(xi1),
        pi.ScalarTerm(xi2, scale=scalar_scale_funcs[0](1)),
        pi.IntegralTerm(pi.Product(pi.ScalarFunction("int_scale1"), dz_x1),
                        limits=limits),
        pi.IntegralTerm(pi.Product(pi.ScalarFunction("int_scale1"), x2),
                        limits=limits)
    ]
    y_bar_minus1 = [
        pi.ScalarTerm(xi1),
        pi.ScalarTerm(xi2, scale=scalar_scale_funcs[1](-1)),
        pi.IntegralTerm(pi.Product(pi.ScalarFunction("int_scale2"), dz_x1),
                        limits=limits, scale=-1),
        pi.IntegralTerm(pi.Product(pi.ScalarFunction("int_scale2"), x2),
                        limits=limits)
    ]
    dz_y_bar_plus1 = [
        pi.ScalarTerm(xi2, scale=scalar_scale_funcs[2](1)),
        pi.IntegralTerm(pi.Product(pi.ScalarFunction("int_scale3"), dz_x1),
                        limits=limits),
        pi.IntegralTerm(pi.Product(pi.ScalarFunction("int_scale3"), x2),
                        limits=limits)
    ]
    dz_y_bar_minus1 = [
        pi.ScalarTerm(xi2, scale=scalar_scale_funcs[3](-1)),
        pi.IntegralTerm(pi.Product(pi.ScalarFunction("int_scale4"), dz_x1),
                        limits=limits, scale=-1),
        pi.IntegralTerm(pi.Product(pi.ScalarFunction("int_scale4"), x2),
                        limits=limits)
    ]

    k = flatness_based_controller(x2_sys(1), y_bar_plus1, y_bar_minus1,
                                  dz_y_bar_plus1, dz_y_bar_minus1,
                                  "explicit_controller")
    return k

def approximate_controller(sys_lbl, modal_lbl):
    # TODO
    pass


def flatness_based_controller(x2_plus1, y_bar_plus1, y_bar_minus1,
                              dz_y_bar_plus1, dz_y_bar_minus1, name):
    k = pi.StateFeedback(pi.WeakFormulation(scale_equation_term_list(
        [pi.ScalarTerm(x2_plus1, scale=-(1 - ctrl_gain.alpha))] +
        scale_equation_term_list(dz_y_bar_plus1, factor=(1 - param.m * ctrl_gain.k1)) +
        scale_equation_term_list(dz_y_bar_minus1, factor=-ctrl_gain.alpha * (1 + param.m * ctrl_gain.k1)) +
        scale_equation_term_list(y_bar_plus1, factor=-param.m * ctrl_gain.k0) +
        scale_equation_term_list(y_bar_minus1, factor=-ctrl_gain.alpha * param.m * ctrl_gain.k0)
        ,factor=(1 + ctrl_gain.alpha) ** -1
    ), name=name))
    return k


def init_observer_gain(sys_fem_lbl, sys_modal_lbl, obs_fem_lbl, obs_modal_lbl):
    from pyinduct.examples.string_with_mass.observer_evp_scripts.modal_approximation import get_observer_gain
    x1_sys_fem = pi.FieldVariable(sys_fem_lbl + "_1_visu")
    x3_obs_fem = pi.FieldVariable(obs_fem_lbl + "_3_visu")
    x3_obs_modal = pi.FieldVariable(obs_modal_lbl + "_3_visu")

    # define observer error
    smooth = [(0, 1), (0, 1), "poly", 2]
    fem_observer_error = SwmObserverError(pi.WeakFormulation([
        pi.ScalarTerm(x3_obs_fem(1)),
        pi.ScalarTerm(x1_sys_fem(0), scale=-1),
    ], "fem_observer_error"), smooth=smooth)
    modal_observer_error = SwmObserverError(pi.WeakFormulation([
        pi.ScalarTerm(x3_obs_modal(1)),
        pi.ScalarTerm(x1_sys_fem(0), scale=-1),
    ], "modal_observer_error"), smooth=smooth)

    # symbolic observer gain
    l, l_bc = get_observer_gain()
    L = SwmBaseCanonicalFraction(
        [pi.LambdifiedSympyExpression([l[2]], sym.theta, (-1, 1))],
        [l[0], l[1]])

    # register bases with precomputed observer gain approximations
    pi.register_base(obs_fem_lbl + "observer_gain", pi.Base([SwmBaseCanonicalFraction(
        [pi.ConstantFunction(float(integrate_function(
            lambda th: L.members["funcs"][0](th) * f.members["funcs"][0](th),
            [(-1, 1)])[0]), domain=(-1, 1))],
        [L.members["scalars"][0] * f.members["scalars"][0],
         L.members["scalars"][1] * f.members["scalars"][1]])
         for f in pi.get_base(obs_fem_lbl + "_test")]))
    pi.register_base(obs_modal_lbl + "observer_gain", pi.Base([SwmBaseCanonicalFraction(
        [pi.ConstantFunction(float(integrate_function(
            lambda th: L.members["funcs"][0](th) * f.members["funcs"][0](th),
            [(-1, 1)])[0]), domain=(-1, 1))],
        [L.members["scalars"][0] * f.members["scalars"][0],
         L.members["scalars"][1] * f.members["scalars"][1]])
        for f in pi.get_base(obs_modal_lbl + "_test")]))

    return fem_observer_error, modal_observer_error


def ocf_inverse_state_transform(org_state):
    r"""
    Transformation of the the state
    :math:`x(z,t) = (x(z,t), \dot x(z,t), x(0,t), \dot x(0,t))^T
    = (x_1(z,t), x_2(z,t), \xi_1(t), \xi_2(t))^T`
    into the coordinates of the observer canonical form

    .. math::
        :nowrap:

        \begin{align*}
            \bar x_1(t) &= w_2'(1) \\
            \bar x_2(t) &= w_1'(1) + w_2'(1) \\
            \bar x_3(\theta, t) &= \frac{1}{2}(w_2(1-\theta) + w_1'(1-\theta)),
            \quad \forall \theta > 0 \\
            \bar x_3(\theta, t) &= \frac{1}{2}(w_2(1+\theta) - w_1'(1+\theta)) +
            w_1'(1) - \theta w_2'(1),
            \quad \forall \theta \le 0 \\
            w_i(z) &= 2\int_0^z \left( \xi_i + \frac{1}{m}\int_0^\zeta
            x_i(\bar\zeta) \,d\bar\zeta \right) \,d\zeta,\quad i=1,2.
        \end{align*}

    Args:
        org_state (:py:class:`.SwmBaseFraction`): State

    Returns:
        :py:class:`.SwmBaseCanonicalFraction`: Transformation
    """
    w = list()
    dz_w = list()
    for x, xi in zip(org_state.members["funcs"], org_state.members["scalars"]):
        def int_handle1(zeta, xi=xi, x=x):
            return 2 * (xi +
                        param.m ** -1 * integrate_function(x, [(0, zeta)])[0])
        def int_handle2(z):
            return integrate_function(int_handle1, [(0, z)])[0]
        w.append(deepcopy(int_handle2))
        dz_w.append(int_handle1)

    y1 = dz_w[1](1)
    y2 = dz_w[0](1) + dz_w[1](1)

    def y3(theta):
        if theta > 0:
            return .5 * (w[1](1 - theta) + dz_w[0](1 - theta))
        else:
            return .5 * (w[1](1 + theta) - dz_w[0](1 + theta)) + dz_w[0](1) - theta * dz_w[1](1)

    return SwmBaseCanonicalFraction([pi.Function(y3, (-1, 1))], [y1, y2])


def apply_control_mode(sys_fem_lbl, sys_modal_lbl,
                       obs_fem_lbl, obs_modal_lbl, mode):

    # obligatory
    for postfix in ("_1_visu", "_2_visu", "_3_visu"):
        pi.get_base(obs_modal_lbl + postfix).matching_base_lbls = [obs_modal_lbl]
        pi.get_base(obs_modal_lbl + postfix).intermediate_base_lbls = [obs_modal_lbl]
        pi.get_base(obs_fem_lbl + postfix).matching_base_lbls = [obs_fem_lbl]
        pi.get_base(obs_fem_lbl + postfix).intermediate_base_lbls = [obs_fem_lbl]
    for postfix in ("_1_visu", "_2_visu", "_3_visu", "_4_visu"):
        pi.get_base(sys_fem_lbl + postfix).intermediate_base_lbls = [sys_fem_lbl]
        pi.get_base(sys_fem_lbl + postfix).matching_base_lbls = [sys_fem_lbl]
    for postfix in ("_1_ctrl", "_2_ctrl", "_3_ctrl", "_4_ctrl"):
            pi.get_base(sys_fem_lbl + postfix).intermediate_base_lbls = [sys_fem_lbl + "_ctrl"]
            pi.get_base(sys_fem_lbl + postfix).matching_base_lbls = [sys_fem_lbl + "_ctrl"]
            pi.get_base(sys_modal_lbl + postfix).intermediate_base_lbls = [sys_modal_lbl + "_ctrl"]
            pi.get_base(sys_modal_lbl + postfix).matching_base_lbls = [sys_modal_lbl + "_ctrl"]

    # controller gets the weights from the system
    if mode == "closed_loop":
        # for the fem control case
        pi.get_base(sys_fem_lbl + "_ctrl").intermediate_base_lbls = [sys_fem_lbl]
        pi.get_base(sys_fem_lbl + "_ctrl").matching_base_lbls = [sys_fem_lbl]
        # for the modal control case
        pi.get_base(sys_modal_lbl + "_ctrl").intermediate_base_lbls = [sys_fem_lbl]

    # controller gets the weights from the modal observer
    elif mode == "modal_observer":
        # for the modal control case
        pi.get_base(sys_modal_lbl + "_ctrl").intermediate_base_lbls = [sys_modal_lbl]
        pi.get_base(sys_modal_lbl + "_ctrl").matching_base_lbls = [sys_modal_lbl]
        # for the fem control case
        pi.get_base(sys_fem_lbl + "_ctrl").intermediate_base_lbls = [sys_modal_lbl]
        # route to the modal observer
        pi.get_base(sys_modal_lbl).intermediate_base_lbls = [obs_modal_lbl]
        # hint that the modal observer evolve with the same weights
        pi.get_base(sys_modal_lbl).matching_base_lbls = [obs_modal_lbl]

    # controller gets the weights from the fem observer
    elif mode == "fem_observer":
            # compute transformation
            pi.get_base(sys_fem_lbl + "_ctrl").intermediate_base_lbls = [sys_fem_lbl + "_trafo"]
            # route to the fem observer
            pi.get_base(sys_fem_lbl + "_trafo").intermediate_base_lbls = [obs_fem_lbl]
            # hint that the fem observer base evolve with the same weights
            pi.get_base(sys_fem_lbl + "_trafo").matching_base_lbls = [obs_fem_lbl]

    elif mode == "open_loop":
        pass

    else:
        raise NotImplementedError

