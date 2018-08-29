from pyinduct.examples.string_with_mass.utils import *
import pyinduct as pi
from tqdm import tqdm


def build_weak_formulation(sys_lbl, spatial_domain, input_, name="system"):
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
    fem_funcs1 = pi.LagrangeNthOrder.cure_interval(nodes1, order=3)
    fem_funcs2 = pi.LagrangeNthOrder.cure_interval(nodes2, order=2)
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


def build_primal_modal_bases(base_lbl, eigenvalues, domain=(0, 1), complex_=False):
    if len(eigenvalues) % 2 == 0:
        raise ValueError("Only odd number of eigenvalues supported.")

    phi_store_dict = dict()
    base = list()
    eigenvector = get_primal_eigenvector(True).subs(subs_list)
    for i, ev in tqdm(enumerate(eigenvalues),
                      desc="build modal base",
                      total=len(eigenvalues)):
        phi = eigenvector

        # if we have a second 0 eigenvalue: stop and think
        if np.isclose(ev, 0) and sum([np.isclose(ev_, 0) for ev_ in eigenvalues[: i]]) == 1:
            raise ValueError("Only one eigenvalue by 0 can be considered!")

        # calculate the limit (instead substitution) if eigenvalue = 0
        if np.isclose(ev, 0):
            phi = sp.Matrix([[sp.limit(phi[0], sym.lam, ev)],
                             [sp.limit(phi[1], sym.lam, ev)]])
        else:
            phi = phi.subs(sym.lam, ev)

        # take the already calculated imaginary part of
        # the eigenvector (see next comment)
        if i in phi_store_dict:
            phi = phi_store_dict[i]

        # if complex functions are not desired and a eigenvalue
        # with non vanishing imaginary part is considered then divide
        # the corresponding complex function in real and imaginary
        # part and store one part for the next conjugate
        # complex eigenvalue / loop
        elif not complex_:
            phi_real, phi_imag = phi.expand(complex=True).as_real_imag()
            phi = phi_real
            if np.isclose(np.imag(ev), 0):
                np.testing.assert_array_almost_equal(phi_imag, [[0], [0]])
            else:
                phi_store_dict.update({i + 1: phi_imag})

        # append eigenvector as SwmBaseFraction
        funcs1 = [phi[0], sp.diff(phi[0], sym.z)]
        funcs2 = [phi[1], sp.diff(phi[1], sym.z)]
        if complex_:
            scalar = [complex(sp.limit(phi[0], sym.z, 0))]
        else:
            scalar = [float(sp.limit(phi[0], sym.z, 0))]
        base.append(SwmBaseFraction([
            pi.LambdifiedSympyExpression(funcs1, sym.z, domain, complex_),
            pi.LambdifiedSympyExpression(funcs2, sym.z, domain, complex_)],
            scalar))

    pi.register_base(base_lbl, pi.Base(base))


class SwmBaseFraction(pi.ComposedFunctionVector):
    def __init__(self, functions, scalars=None):
        if scalars is None:
            functions, scalars = functions
        pi.ComposedFunctionVector.__init__(self, functions, scalars)

    @staticmethod
    def scalar_product(left, right):
        def _scalar_product(left, right):
            return (
                pi.dot_product_l2(left.members["funcs"][0], right.members["funcs"][0]) +
                pi.dot_product_l2(left.members["funcs"][1], right.members["funcs"][1]) +
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
