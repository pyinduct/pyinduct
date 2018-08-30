from pyinduct.examples.string_with_mass.system import *
from pyinduct.parabolic.control import scale_equation_term_list

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

def build_controller(base_lbl):
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
        :py:class:`pyinduct.simulation.Controller`: Control law
    """
    x1 = pi.FieldVariable(base_lbl + "_1_visu")
    x2 = pi.FieldVariable(base_lbl + "_2_visu")
    xi1 = pi.FieldVariable(base_lbl + "_3_visu")(0)
    xi2 = pi.FieldVariable(base_lbl + "_4_visu")(0)
    dz_x1 = x1.derive(spat_order=1)

    scalar_scale_funcs = [pi.Function(lambda theta: param.m * (1 - np.exp(-theta / param.m))),
                          pi.Function(lambda theta: param.m * (-1 + np.exp(theta / param.m))),
                          pi.Function(lambda theta: np.exp(-theta / param.m)),
                          pi.Function(lambda theta: np.exp(theta / param.m))]

    pi.register_base("int_scale1", pi.Base(pi.Function(lambda tau: 1 - np.exp(-(1 - tau) / param.m))))
    pi.register_base("int_scale2", pi.Base(pi.Function(lambda tau: -1 + np.exp((-1 + tau) / param.m))))
    pi.register_base("int_scale3", pi.Base(pi.Function(lambda tau: np.exp(-(1 - tau) / param.m) / param.m)))
    pi.register_base("int_scale4", pi.Base(pi.Function(lambda tau: np.exp((-1 + tau) / param.m) / param.m)))

    limits = (0, 1)
    y_bar_plus1 = [pi.ScalarTerm(xi1),
                   pi.ScalarTerm(xi2, scale=scalar_scale_funcs[0](1)),
                   pi.IntegralTerm(pi.Product(pi.ScalarFunction("int_scale1"), dz_x1), limits=limits),
                   pi.IntegralTerm(pi.Product(pi.ScalarFunction("int_scale1"), x2), limits=limits)
                   ]
    y_bar_minus1 = [pi.ScalarTerm(xi1),
                    pi.ScalarTerm(xi2, scale=scalar_scale_funcs[1](-1)),
                    pi.IntegralTerm(pi.Product(pi.ScalarFunction("int_scale2"), dz_x1), limits=limits, scale=-1),
                    pi.IntegralTerm(pi.Product(pi.ScalarFunction("int_scale2"), x2), limits=limits)
                    ]
    dz_y_bar_plus1 = [pi.ScalarTerm(xi2, scale=scalar_scale_funcs[2](1)),
                      pi.IntegralTerm(pi.Product(pi.ScalarFunction("int_scale3"), dz_x1), limits=limits),
                      pi.IntegralTerm(pi.Product(pi.ScalarFunction("int_scale3"), x2), limits=limits)
                      ]
    dz_y_bar_minus1 = [pi.ScalarTerm(xi2, scale=scalar_scale_funcs[3](-1)),
                       pi.IntegralTerm(pi.Product(pi.ScalarFunction("int_scale4"), dz_x1), limits=limits, scale=-1),
                       pi.IntegralTerm(pi.Product(pi.ScalarFunction("int_scale4"), x2), limits=limits)
                       ]

    k = flatness_based_controller(x2(1), y_bar_plus1, y_bar_minus1,
                                  dz_y_bar_plus1, dz_y_bar_minus1,
                                  "explicit_controller")
    return k


def flatness_based_controller(x2_plus1, y_bar_plus1, y_bar_minus1,
                              dz_y_bar_plus1, dz_y_bar_minus1, name):
    k = pi.Controller(pi.WeakFormulation(scale_equation_term_list(
        [pi.ScalarTerm(x2_plus1, scale=-(1 - ctrl_gain.alpha))] +
        scale_equation_term_list(dz_y_bar_plus1, factor=(1 - param.m * ctrl_gain.k1)) +
        scale_equation_term_list(dz_y_bar_minus1, factor=-ctrl_gain.alpha * (1 + param.m * ctrl_gain.k1)) +
        scale_equation_term_list(y_bar_plus1, factor=-param.m * ctrl_gain.k0) +
        scale_equation_term_list(y_bar_minus1, factor=-ctrl_gain.alpha * param.m * ctrl_gain.k0)
        ,factor=(1 + ctrl_gain.alpha) ** -1
    ), name=name))
    return k
