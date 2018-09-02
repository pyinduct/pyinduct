from pyinduct.examples.string_with_mass.utils import *
from pyinduct.core import integrate_function
import collections
import pyinduct as pi
from tqdm import tqdm


def build_canonical_weak_formulation(obs_lbl, spatial_domain, u,
                                     obs_err, name="system"):
    r"""
    Observer canonical form of the string with mass example

    .. math::
        :nowrap:

        \begin{align*}
            \dot{x}_1(t) &= \frac{2}{m}u(t) \\
            \dot{x}_2(t) &= x_1(t) + \frac{2}{m}u(t) \\
            \dot{x}_3(z,t) &= -x_3'(z,t)-\frac{2}{m}(1-h(z))z u(t) - m^{-1} y(t)
        \end{align*}

    Boundary condition

    .. math:: x_3(-1,t) = x_2(t) - y(t)

    Weak formulation

    .. math::
        :nowrap:

        \begin{align*}
            -\langle \dot x(z, t), \psi(z)\rangle &=
            \frac{2}{m}u(t)\psi_1 + \frac{2}{m}u(t)\psi_2 + x_1\psi_2
            -x_3(1,t)\psi_3(1) - m^{-1}\langle y(t), \psi_3(z) \rangle \\
            &+ \underbrace{x_3(-1,t)\psi_3(-1)}_{x_2(t)\psi_3(-1) - y(t)\psi_3(-1)}
            + \langle x_3(z,t) , \psi_3'(z)\rangle
            + \frac{2}{m}\langle (1 - h(z))z , \psi_3(z)\rangle u(t)
        \end{align*}

    Output equation

    .. math:: x_3(1,t) =  y(t)

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
    x = pi.FieldVariable(obs_lbl)
    psi = pi.TestFunction(obs_lbl + "_test")
    psi1 = pi.TestFunction(obs_lbl + "_10")
    psi2 = pi.TestFunction(obs_lbl + "_20")
    psi3 = pi.TestFunction(obs_lbl + "_30")
    psi2_x1 = pi.TestFunction(obs_lbl + "_21")
    psi3_x2_at_m1 = pi.TestFunction(obs_lbl + "_32_at_m1")
    psi3_integrated = pi.TestFunction(obs_lbl + "_3_integrated")
    psi_3_int_intup_scale_f = pi.TestFunction(obs_lbl + "_3_integred_with_input_scale")
    psi_3_obs_gain = pi.TestFunction(obs_lbl + "observer_gain")

    dummy = 0
    bounds = spatial_domain.bounds
    wf = pi.WeakFormulation(
        [
            pi.IntegralTerm(pi.Product(x.derive(temp_order=1), psi), limits=bounds, scale=-1),
            pi.ScalarTerm(pi.Product(u, psi1(dummy)), scale=2 / param.m),
            pi.ScalarTerm(pi.Product(u, psi2(dummy)), scale=2 / param.m),
            pi.ScalarTerm(pi.Product(u, psi_3_int_intup_scale_f(dummy)), scale=-1),
            pi.ScalarTerm(pi.Product(x(dummy), psi2_x1(dummy))),
            pi.ScalarTerm(pi.Product(x(1), psi3(1)), scale=-1),
            pi.ScalarTerm(pi.Product(x(dummy), psi3_x2_at_m1(dummy))),
            pi.ScalarTerm(pi.Product(x(1), psi3(-1)), scale=-1),
            pi.IntegralTerm(pi.Product(x, psi3.derive(1)), limits=bounds),
            pi.ScalarTerm(pi.Product(x(1), psi3_integrated(dummy)), scale=-param.m ** -1),
            pi.ScalarTerm(pi.Product(obs_err, psi_3_obs_gain(dummy))),
            pi.ScalarTerm(pi.Product(obs_err, psi3(-1)), scale=-obs_gain.alpha + 1),
        ],
        name=name
    )

    return wf


def build_original_weak_formulation(sys_lbl, spatial_domain, u, name="system"):
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


def build_fem_bases(base_lbl, n1, n2, cf_base_lbl, ncf, modal_base_lbl):
    nodes1 = pi.Domain((0, 1), n1)
    nodes2 = pi.Domain((0, 1), n2)
    cf_nodes = pi.Domain((-1, 1), ncf)
    assert nodes1.bounds == nodes2.bounds

    fem_funcs1 = pi.LagrangeNthOrder.cure_interval(nodes1, order=1)
    fem_funcs2 = pi.LagrangeNthOrder.cure_interval(nodes2, order=1)
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
    pi.register_base(base_lbl, pi.Base(
        # to control the modal system
        # base1 + base2 + base4, intermediate_base=modal_base_lbl))
        base1 + base2 + base4))
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
    pi.register_base(base_lbl + "_1_visu", pi.Base(fb1 + zb2 + zb4, matching_bases=[base_lbl], intermediate_base=base_lbl))
    pi.register_base(base_lbl + "_2_visu", pi.Base(zb1 + fb2 + zb4, matching_bases=[base_lbl], intermediate_base=base_lbl))
    pi.register_base(base_lbl + "_3_visu", pi.Base(ob1 + zb2 + zb4, matching_bases=[base_lbl], intermediate_base=base_lbl))
    pi.register_base(base_lbl + "_4_visu", pi.Base(zb1 + zb2 + ob4, matching_bases=[base_lbl], intermediate_base=base_lbl))

    def heavi(z):
        return 0 if z < 0 else (0.5 if z == 0 else 1)

    # bases for the canonical form
    cf_fem_funcs = pi.LagrangeNthOrder.cure_interval(cf_nodes, order=3)
    cf_zero_func = pi.Function.from_constant(0, domain=cf_nodes.bounds)
    cf_one_func = pi.Function.from_constant(1, domain=cf_nodes.bounds)

    cf_base1 = [SwmBaseCanonicalFraction([cf_zero_func], [1, 0])]
    cf_base10 = [SwmBaseCanonicalFraction([cf_zero_func], [0, 0])]

    cf_base2 = [SwmBaseCanonicalFraction([cf_zero_func], [0, 1])]
    cf_base20 = [SwmBaseCanonicalFraction([cf_zero_func], [0, 0])]
    cf_base21 = [SwmBaseCanonicalFraction([cf_zero_func], [1, 0])]

    cf_base3, cf_base30, cf_base32_at_m1, cf_base3_integrated, cf_base3_int_ip_scale = [
        list() for _ in range(5)]
    for f in cf_fem_funcs:
        cf_base3.append(SwmBaseCanonicalFraction([f], [0, 0]))
        cf_base30.append(SwmBaseCanonicalFraction([cf_zero_func], [0, 0]))
        cf_base32_at_m1.append(SwmBaseCanonicalFraction([cf_zero_func], [0, f(-1)]))
        cf_base3_integrated.append(SwmBaseCanonicalFraction([pi.Function.from_constant(
            float(integrate_function(f, [(-1, 1)])[0]), domain=cf_nodes.bounds)], [0, 0]))
        cf_base3_int_ip_scale.append(SwmBaseCanonicalFraction([pi.Function.from_constant(
            float(integrate_function(lambda z: 2 / param.m * z * f(z), [(-1, 0)])[0]),
            domain=cf_nodes.bounds)], [0, 0]))

    # bases for the system / weak formulation
    pi.register_base(cf_base_lbl, pi.Base(cf_base1 + cf_base2 + cf_base3,
                                          # for explicit fem observer
                                          matching_bases=[cf_base_lbl + "_gain",
                                                          cf_base_lbl + "30"]))
    pi.register_base(cf_base_lbl + "_test", pi.Base(cf_base1 + cf_base2 + cf_base3))
    pi.register_base(cf_base_lbl + "_10", pi.Base(cf_base1 + cf_base20 + cf_base30))
    pi.register_base(cf_base_lbl + "_20", pi.Base(cf_base10 + cf_base2 + cf_base30))
    pi.register_base(cf_base_lbl + "_30", pi.Base(cf_base10 + cf_base20 + cf_base3))
    pi.register_base(cf_base_lbl + "_21", pi.Base(cf_base10 + cf_base21 + cf_base30))
    pi.register_base(cf_base_lbl + "_32_at_m1", pi.Base(cf_base10 + cf_base20 + cf_base32_at_m1))
    pi.register_base(cf_base_lbl + "_3_integrated",
                     pi.Base(cf_base10 + cf_base20 + cf_base3_integrated))
    pi.register_base(cf_base_lbl + "_3_integred_with_input_scale",
                     pi.Base(cf_base10 + cf_base20 + cf_base3_int_ip_scale))

    # bases for visualization
    ob1 = [cf_one_func]
    ob2 = [cf_one_func]
    fb3 = list(cf_fem_funcs.fractions)
    zb1 = [cf_zero_func]
    zb2 = [cf_zero_func]
    zb3 = [cf_zero_func for _ in range(len(cf_nodes))]
    pi.register_base(cf_base_lbl + "_1_visu", pi.Base(ob1 + zb2 + zb3, matching_bases=[cf_base_lbl]))
    pi.register_base(cf_base_lbl + "_2_visu", pi.Base(zb1 + ob2 + zb3, matching_bases=[cf_base_lbl]))
    pi.register_base(cf_base_lbl + "_3_visu", pi.Base(zb1 + zb2 + fb3, matching_bases=[cf_base_lbl],
                                                      intermediate_base=cf_base_lbl))

def build_modal_bases(base_lbl, n, cf_base_lbl, ncf):
    # get eigenvectors
    m = max((n, ncf))
    from pyinduct.examples.string_with_mass.observer_evp_scripts.modal_approximation \
        import build_bases_for_modal_observer_approximation
    primal_base, primal_base_nf, dual_base, dual_base_nf, eig_vals = (
        build_bases_for_modal_observer_approximation(m))
    primal_base, dual_base = primal_base[:n], dual_base[:n]
    primal_base_nf, dual_base_nf = primal_base_nf[:ncf], dual_base_nf[:ncf]

    # bases for the original form
    register_evp_base(base_lbl, primal_base, sym.z, (0, 1),
                      matching_bases=[cf_base_lbl],
                      intermediate_base=cf_base_lbl)
    register_evp_base(base_lbl + "test", dual_base, sym.z, (0, 1))

    # bases for the canonical form
    input_scale = 2 / param.m * sym.theta

    register_evp_base(cf_base_lbl, primal_base_nf, sym.theta, (-1, 1),
                      matching_bases=base_lbl)
    register_evp_base(cf_base_lbl + "_test", dual_base_nf, sym.theta, (-1, 1))
    register_evp_base(cf_base_lbl + "_10",
                      [[ev[0] * 1, ev[1] * 0, ev[2] * 0]
                       for ev in dual_base_nf], sym.theta, (-1, 1))
    register_evp_base(cf_base_lbl + "_20",
                      [[ev[0] * 0, ev[1] * 1, ev[2] * 0]
                       for ev in dual_base_nf], sym.theta, (-1, 1))
    register_evp_base(cf_base_lbl + "_30",
                      [[ev[0] * 0, ev[1] * 0, ev[2] * 1]
                       for ev in dual_base_nf], sym.theta, (-1, 1))
    register_evp_base(cf_base_lbl + "_21",
                      [[ev[1] * 1, ev[0] * 0, ev[2] * 0]
                       for ev in dual_base_nf], sym.theta, (-1, 1))
    register_evp_base(cf_base_lbl + "_32_at_m1",
                      [[ev[0] * 0, ev[2].subs(sym.theta, -1), ev[1] * 0]
                       for ev in dual_base_nf], sym.theta, (-1, 1))
    register_evp_base(cf_base_lbl + "_3_integrated",
                      [[ev[0] * 0, ev[1] * 0, sp.integrate(ev[2], (sym.theta, -1, 1))]
                       for ev in dual_base_nf], sym.theta, (-1, 1))
    register_evp_base(cf_base_lbl + "_3_integred_with_input_scale",
                      [[ev[0] * 0, ev[1] * 0, sp.integrate(ev[2] * input_scale, (sym.theta, -1, 0))]
                       for ev in dual_base_nf], sym.theta, (-1, 1))

    pi.register_base(
        cf_base_lbl + "_1_visu", pi.Base(
            [pi.Function.from_constant(ev.members["scalars"][0], domain=(-1, 1))
             for ev in pi.get_base(cf_base_lbl)]))
    pi.register_base(
        cf_base_lbl + "_2_visu", pi.Base(
            [pi.Function.from_constant(ev.members["scalars"][1], domain=(-1, 1))
             for ev in pi.get_base(cf_base_lbl)]))
    pi.register_base(
        cf_base_lbl + "_3_visu", pi.Base(
            [ev.members["funcs"][0] for ev in pi.get_base(cf_base_lbl)],
        matching_bases=[cf_base_lbl], intermediate_base=cf_base_lbl))


def register_evp_base(base_lbl, eigenvectors, sp_var, domain,
                      intermediate_base=None, matching_bases=list()):
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

    pi.register_base(base_lbl, pi.Base(base,
                                       intermediate_base=intermediate_base,
                                       matching_bases=matching_bases))


class SwmBaseFraction(pi.ComposedFunctionVector):
    l2_scalar_product = True

    def __init__(self, functions, scalars):
        assert len(functions) == 2
        assert len(scalars) == 2
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

    def get_member(self, idx):
        return self


class SwmBaseCanonicalFraction(pi.ComposedFunctionVector):
    def __init__(self, functions, scalars):
        assert len(functions) == 1
        assert len(scalars) == 2
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
        if isinstance(values, (collections.Iterable, pi.Domain)):
            res = list()
            for val in values:
                res.append(self(val))
            return np.array(res)[:, 2]

        else:
            return self(values)[2]

    def derive(self, order):
        if order == 0:
            return self
        else:
            return SwmBaseCanonicalFraction(
                [f.derive(order) for f in self.members["funcs"]], [0, 0])

    def get_member(self, idx):
        return self

