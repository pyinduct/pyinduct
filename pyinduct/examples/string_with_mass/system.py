from pyinduct.examples.string_with_mass.utils import *
import pyinduct as pi
from tqdm import tqdm


def build_canonical_weak_formulation(sys_lbl, spatial_domain, input_, name="system"):
    r"""
    The observer from [Woi2012] (equation 41)

    .. math::
        :nowrap:

        \begin{align*}
            \dot{\hat{\eta}}_1(t) &= \frac{2}{m}u(t) - (1+\alpha)k_0\tilde{y}(t) \\
            \dot{\hat{\eta}}_2(t) &= \hat{\eta}_1(t) + \frac{2}{m}u(t)-((1+\alpha)k_1+2k_0)\tilde{y}(t) \\
            \dot{\hat{\eta}}_3(\theta,t) &= -\hat{\eta}_3'(\theta,t)-\frac{2}{m}(1-h(\theta))\theta u(t)
                - m^{-1} \hat{y}(t) \\
            &\qquad -(k_0(1-\theta)+k_1-m^{-1})\tilde{y}(t)
        \end{align*}

    .. math:: \hat{\eta}_3(-1,t) = \hat{\eta}_2(t) -\hat{y}(t)-(\alpha-1)\tilde{y}(t)

    is considered through integration by parts of the term
    :math:`-\langle\hat{\eta}_3'(\theta),\psi_j(\theta)\rangle` from the weak formulation of equation 41a:

    .. math::
        :nowrap:

        \begin{align*}
            -\langle\hat{\eta}_3'(\theta),\psi_j(\theta)\rangle &=
            -\hat{\eta}_3(1)\psi_j'(1) + \hat{\eta}_3(-1)\psi_j'(-1)
            \langle\hat{\eta}_3(\theta),\psi_j'(\theta)\rangle.
        \end{align*}

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
    pass
    # obs_err = pi.Controller(pi.WeakFormulation(
    #     [
    #         pi.ScalarTerm(pi.FieldVariable(sys_approx_label, location=0), scale=-1),
    #         pi.ScalarTerm(eta3(-1).derive(spat_order=1), scale=-params.m / 2),
    #         pi.ScalarTerm(eta3(1).derive(spat_order=1), scale=-params.m / 2),
    #         pi.ScalarTerm(eta1(0), scale=-params.m / 2),
    #     ],
    #     name="nf observer error"))
    # input_vector.append([obs_err])
    #
    # d_eta1 = pi.WeakFormulation(
    #     [
    #         pi.ScalarTerm(eta1(0).derive(temp_order=1), scale=-1),
    #         pi.ScalarTerm(pi.Input(input_vector, index=0), scale=2 / params.m),
    #         pi.ScalarTerm(pi.Input(input_vector, index=1),
    #                       scale=0 if approx_bounded_part else -(1 + params.alpha_ob) * params.k0_ob)
    #     ],
    #     name=obs_approx_label[0]
    # )
    # d_eta2 = pi.WeakFormulation(
    #     [
    #         pi.ScalarTerm(eta2(0).derive(temp_order=1), scale=-1),
    #         # index error in paper
    #         pi.ScalarTerm(eta1(0)),
    #         pi.ScalarTerm(pi.Input(input_vector, index=0), scale=2 / params.m),
    #         pi.ScalarTerm(pi.Input(input_vector, index=1),
    #                       scale=0 if approx_bounded_part else -(1 + params.alpha_ob) * params.k1_ob - 2 * params.k0_ob)
    #     ],
    #     name=obs_approx_label[1]
    # )
    # d_eta3 = pi.WeakFormulation(
    #     [
    #         pi.IntegralTerm(pi.Product(eta3.derive(temp_order=1), psi), limits=limits, scale=-1),
    #         # sign error in paper
    #         pi.IntegralTerm(pi.Product(pi.Product(obs_scale1, psi), pi.Input(input_vector, index=0)), limits=limits,
    #                         scale=-1),
    #         pi.IntegralTerm(pi.Product(pi.Product(obs_scale2, psi), pi.Input(input_vector, index=1)), limits=limits,
    #                         scale=0 if approx_bounded_part else 1),
    #         # \hat y
    #         pi.IntegralTerm(pi.Product(eta3(-1).derive(spat_order=1), psi), limits=limits, scale=1 / 2),
    #         pi.IntegralTerm(pi.Product(eta3(1).derive(spat_order=1), psi), limits=limits, scale=1 / 2),
    #         pi.IntegralTerm(pi.Product(eta1, psi), limits=limits, scale=1 / 2),
    #         # shift
    #         pi.IntegralTerm(pi.Product(eta3, psi.derive(1)), limits=limits),
    #         pi.ScalarTerm(pi.Product(eta3(1), psi(1)), scale=-1),
    #         # bc
    #         pi.ScalarTerm(pi.Product(psi(-1), eta2(0))),
    #         pi.ScalarTerm(pi.Product(pi.Input(input_vector, index=1), psi(-1)), scale=0 if approx_unbounded_part else 1 - params.alpha_ob),
    #         # bc \hat y
    #         pi.ScalarTerm(pi.Product(eta3(-1).derive(spat_order=1), psi(-1)), params.m / 2),
    #         pi.ScalarTerm(pi.Product(eta3(1).derive(spat_order=1), psi(-1)), params.m / 2),
    #         pi.ScalarTerm(pi.Product(psi(-1), eta1(1)), params.m / 2),
    #         # approximate unbouned part
    #         # pi.ScalarTerm(pi.Product(pi.Input(input_vector, index=1), psi(1)), scale=1-params.alpha_ob if approx_unbounded_part else 0),
    #     ],
    #     name=obs_approx_label[2]
    # )
    #
    # return [d_eta1, d_eta2, d_eta3]


def build_original_weak_formulation(sys_lbl, spatial_domain, input_, name="system"):
    r"""
    Projection (see :py:meth:`.SwmBaseFraction.scalar_product_hint`

    .. math::
        :nowrap:

        \begin{align*}
            \langle\dot x(z,t), \psi(z)\rangle &=
            \langle x_2(z,t),\psi_1(z)\rangle + \langle x_1''(z,t), \psi_2(z)\rangle +
            \xi_2(t)\psi_3 + x_1'(0)\psi_4
        \end{align*}

    Boundary conditions

    .. math::
        :nowrap:

        \begin{align*}
            x_1(0,t) = \xi_1(t), \qquad u(t) = x_1'(1,t)
        \end{align*}

    Implemented

    .. math::
        :nowrap:

        \begin{align*}
            \langle\dot x(z,t), \psi(z)\rangle =
            &\langle x_2(z,t),\psi_1(z)\rangle + \langle x_1'(z,t), \psi_2'(z)\rangle \\
            &+ u(t)\psi_2(1) - x_1'(0,t)\psi_2(0)
            +\xi_2(t)\psi_3 + x_1'(0)\psi_4
        \end{align*}

    Args:
        sys_lbl (str): Base label
        spatial_domain (:py:class:`.Domain`): Spatial domain of the system.
        name (str): Name of the system.

    Returns:
        :py:class:`.WeakFormulation`

    """
    x = pi.FieldVariable(sys_lbl)
    psi = pi.TestFunction(sys_lbl)
    psi1_xi2_at_0 = pi.TestFunction(sys_lbl + "_1_xi2_at_0")
    psi1_x2 = pi.TestFunction(sys_lbl + "_12")
    psi2_x1 = pi.TestFunction(sys_lbl + "_21")
    psi4_x1 = pi.TestFunction(sys_lbl + "_4_x1")
    u = pi.Input(input_)

    bounds = spatial_domain.bounds
    dummy_location = 0
    wf = pi.WeakFormulation([
        # dot
        pi.IntegralTerm(pi.Product(x.derive(temp_order=1), psi), limits=bounds, scale=-1),
        # integrals
        pi.IntegralTerm(pi.Product(x, psi1_x2), limits=bounds),
        pi.IntegralTerm(pi.Product(x.derive(spat_order=1), psi2_x1.derive(1)), limits=bounds, scale=-1),
        # scalars
        pi.ScalarTerm(pi.Product(u, psi2_x1(1))),
        pi.ScalarTerm(pi.Product(x.derive(spat_order=1)(0), psi2_x1(0)), scale=-1),
        # dot
        pi.ScalarTerm(pi.Product(x(dummy_location), psi1_xi2_at_0(dummy_location))),
        pi.ScalarTerm(pi.Product(x.derive(spat_order=1)(0), psi4_x1(dummy_location)), scale=param.m),
    ], name=name)

    return wf


def build_fem_bases(base_lbl, nodes1, nodes2):
    assert nodes1.bounds == nodes2.bounds
    fem_funcs1 = pi.LagrangeNthOrder.cure_interval(nodes1, order=4)
    fem_funcs2 = pi.LagrangeNthOrder.cure_interval(nodes2, order=4)
    zero_function = pi.Function.from_constant(0, domain=nodes1.bounds)
    one_function = pi.Function.from_constant(1, domain=nodes1.bounds)

    base1, base10, base12, base14_at_0 = [list() for _ in range(4)]
    for i, f in enumerate(fem_funcs1):
        if i == 0:
            base1.append(SwmBaseFraction([f, zero_function], [1, 0]))
            base14_at_0.append(SwmBaseFraction([zero_function, zero_function], [0, f(0)]))
        else:
            base1.append(SwmBaseFraction([f, zero_function], [0, 0]))
            base14_at_0.append(SwmBaseFraction([zero_function, zero_function], [0, 0]))
        base10.append(SwmBaseFraction([zero_function, zero_function], [0, 0]))
        base12.append(SwmBaseFraction([zero_function, f], [0, 0]))

    base2, base20, base21 = [list() for _ in range(3)]
    for f in fem_funcs2:
        base2.append(SwmBaseFraction([zero_function, f], [0, 0]))
        base20.append(SwmBaseFraction([zero_function, zero_function], [0, 0]))
        base21.append(SwmBaseFraction([f, zero_function], [0, 0]))

    base4 = [SwmBaseFraction([zero_function, zero_function], [0, 1])]
    base40 = [SwmBaseFraction([zero_function, zero_function], [0, 0])]
    base4_x1 = [SwmBaseFraction([one_function, zero_function], [0, 0])]

    # bases for the system / weak formulation
    pi.register_base(base_lbl, pi.Base(base1 + base2 + base4))
    pi.register_base(base_lbl + "_12", pi.Base(base12 + base20 + base40))
    pi.register_base(base_lbl + "_21", pi.Base(base10 + base21 + base40))
    pi.register_base(base_lbl + "_1_xi2_at_0", pi.Base(base14_at_0 + base20 + base40))
    pi.register_base(base_lbl + "_4_x1", pi.Base(base10 + base20 + base4_x1))

    # bases for visualization
    fb1 = list(fem_funcs1.fractions)
    fb2 = list(fem_funcs2.fractions)
    ob1 = [one_function] + [zero_function for _ in range(len(nodes1) - 1)]
    ob4 = [one_function]
    zb1 = [zero_function for _ in range(len(nodes1))]
    zb2 = [zero_function for _ in range(len(nodes2))]
    zb4 = [zero_function]
    pi.register_base(base_lbl + "_1_visu", pi.Base(fb1 + zb2 + zb4))
    pi.register_base(base_lbl + "_2_visu", pi.Base(zb1 + fb2 + zb4))
    pi.register_base(base_lbl + "_3_visu", pi.Base(ob1 + zb2 + zb4))
    pi.register_base(base_lbl + "_4_visu", pi.Base(zb1 + zb2 + ob4))


def register_evp_base(base_lbl, eigenvectors, sp_var, domain):
    if len(eigenvectors) % 2 == 1:
        raise ValueError("Only even number of eigenvalues supported.")

    base = list()
    for i, ev in enumerate(eigenvectors):

        # append eigenvector as SwmBaseFraction
        if domain == (0, 1) and sp_var == sym.z:
            base.append(SwmBaseFraction([
                pi.LambdifiedSympyExpression([ev[0], sp.diff(ev[0], sp_var)],
                                             sp_var, domain),
                pi.LambdifiedSympyExpression([ev[1], sp.diff(ev[1], sp_var)],
                                             sp_var, domain)],
                [float(ev[2]), float(ev[3])]))

        elif domain == (-1, 1) and sp_var == sym.theta:
            base.append(SwmBaseCanonicalFraction([
                pi.LambdifiedSympyExpression([ev[2], sp.diff(ev[2], sp_var)],
                                             sp_var, domain)],
                [float(ev[0]), float(ev[1])]))

        else:
            raise NotImplementedError

    pi.register_base(base_lbl, pi.Base(base))


class SwmBaseFraction(pi.ComposedFunctionVector):
    l2_scalar_product = True

    def __init__(self, functions, scalars=None):
        if scalars is None:
            functions, scalars = functions
        pi.ComposedFunctionVector.__init__(self, functions, scalars)

    @staticmethod
    def scalar_product(left, right):
        if SwmBaseFraction.l2_scalar_product:
            def _scalar_product(left, right):
                return (
                    pi.dot_product_l2(left.members["funcs"][0], right.members["funcs"][0]) +
                    pi.dot_product_l2(left.members["funcs"][1], right.members["funcs"][1]) +
                    left.members["scalars"][0] * right.members["scalars"][0] +
                    left.members["scalars"][1] * right.members["scalars"][1]
                )

        else:
            def _scalar_product(left, right):
                return (
                    pi.dot_product_l2(left.members["funcs"][0].derive(1), right.members["funcs"][0].derive(1)) +
                    pi.dot_product_l2(left.members["funcs"][1], right.members["funcs"][1]) +
                    left.members["scalars"][0] * right.members["scalars"][0] +
                    left.members["scalars"][1] * right.members["scalars"][1] * param.m
                )

        if isinstance(left, np.ndarray):
            res = list()
            for l, r in zip(left, right):
                res.append(_scalar_product(l, r))

            return np.array(res)

        else:
            return _scalar_product(left, right)

    def scalar_product_hint(self):
        r"""
        Scalar product for the string with mass system:

        .. math::
            :nowrap:

            \begin{align*}
              \langle x, y\rangle = \int_0^1 (x_1'(z)y_1'(z) + x_2(z)y_2(z) \,dz
              + x_3 y_3 + m x_4 y_4
            \end{align*}

        Returns:
            list(callable): Scalar product function handle wrapped inside a list.
        """
        return [self.scalar_product]

    def __call__(self, z):
        return np.array([f(z) for f in self.members["funcs"]] +
                        [self.members["scalars"][0]] +
                        [self.members["scalars"][1]])

    def evaluation_hint(self, values):
        return self(values)[0]

    def derive(self, order):
        if order == 0:
            return self
        else:
            return SwmBaseFraction(
                [f.derive(order) for f in self.members["funcs"]], [0, 0])


class SwmBaseCanonicalFraction(pi.ComposedFunctionVector):
    def __init__(self, functions, scalars=None):
        if scalars is None:
            functions, scalars = functions
        pi.ComposedFunctionVector.__init__(self, functions, scalars)

    @staticmethod
    def scalar_product(left, right):
        def _scalar_product(left, right):
            return (
                pi.dot_product_l2(left.members["funcs"][0], right.members["funcs"][0]) +
                left.members["scalars"][0] * right.members["scalars"][0] +
                left.members["scalars"][1] * right.members["scalars"][1]
            )

        if isinstance(left, np.ndarray):
            res = list()
            for l, r in zip(left, right):
                res.append(_scalar_product(l, r))

            return np.array(res)

        else:
            return _scalar_product(left, right)

    def scalar_product_hint(self):
        r"""
        Scalar product for the canonical form of the string with mass system:

        Returns:
            list(callable): Scalar product function handle wrapped inside a list.
        """
        return [self.scalar_product]

    def __call__(self, z):
        return np.array([self.members["scalars"][0]] +
                        [self.members["scalars"][1]] +
                        [f(z) for f in self.members["funcs"]])

    def evaluation_hint(self, values):
        return self(values)[0]

    def derive(self, order):
        if order == 0:
            return self
        else:
            return SwmBaseCanonicalFraction(
                [f.derive(order) for f in self.members["funcs"]], [0, 0])
