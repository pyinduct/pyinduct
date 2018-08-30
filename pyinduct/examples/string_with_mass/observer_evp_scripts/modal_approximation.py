from sympy.utilities import lambdify
from scipy.integrate import quad
from tqdm import tqdm
import numpy as np
import sympy as sp


def build_bases_for_modal_observer_approximation(m):
    if m % 2 == 1:
        raise ValueError("Only even number of eigenvalues supported.")

    n = int(m / 2)

    # solve eigenvalue problems in  normal form coordinates by hand: manual = 1
    # or derive from the solutions in original coordinates: manual = 0
    manual = 1


    # parameters
    mass = 1
    spring_damper = 0
    if spring_damper:
        kd0, ks0, kd1, ks1 = [1] * 4
    else:
        k0, k1, alpha = 9, 10, 0
    from pyinduct.examples.string_with_mass.utils import sym
    om, theta, z, t, u, yt, m = sym.om, sym.theta, sym.z, sym.t, sym.u, sym.yt, sym.m
    coefficients = [c(t) for c in sp.symbols("c_:{}".format(n*2))]


    # some helper
    def sum_(iterable):
        sum = iterable[0] * 0
        for v in iterable:
            sum += v
        return sum
    def discard_small_values(mat, eps=10 ** -6):
        mat = np.array(mat).astype(complex)
        rmat = np.real(mat)
        rmat[np.abs(rmat) < eps] = 0
        imat = np.imag(mat)
        imat[np.abs(imat) < eps] = 0
        return np.real_if_close(mat)
    def pprint(item, discard_sv=True):
        if discard_sv and isinstance(item, (np.ndarray, sp.Matrix)):
            item = sp.Matrix(discard_small_values(item))
        elif discard_sv and isinstance(item, (list, tuple)):
            item = [sp.Matrix(discard_small_values(it)) for it in item]
        else:
            item = sp.Matrix(item)
        sp.pprint(item, num_columns=200)
    def numeric_integration(f1, f2, bounds):
        iv, lb, ub = bounds
        f1 = sp.sympify(f1)
        if any([c in f1.atoms(sp.Function) or sp.diff(c, t) in f1.atoms(sp.Function) for c in coefficients]):
            vec, _ = linear_eq_to_matrix([f1], coefficients)
            if_ = [lambdify(iv, el * f2, modules="numpy") for el in vec]
            res = np.sum([quad(f, lb, ub)[0] * c for c, f in zip(coefficients, if_)])
            vec, _ = linear_eq_to_matrix([f1], [sp.diff(c, t) for c in coefficients])
            if_ = [lambdify(iv, el * f2, modules="numpy") for el in vec]
            res += np.sum([quad(f, lb, ub)[0] * sp.diff(c, t) for c, f in zip(coefficients, if_)])
        else:
            if_ = lambdify(iv, (f1 * f2).doit(), modules="numpy")
            res = quad(if_, lb, ub)[0]
        return res
    def l2_ip(f1, f2):
        return numeric_integration(f1, f2, (z, 0, 1))
    def l2_ip_nf(f1, f2, lb=-1, ub=1):
        return numeric_integration(f1, f2, (theta, lb, ub))
    def inner_product(primal, dual):
        return (l2_ip(sp.diff(primal[0], z), sp.diff(dual[0], z))
                + l2_ip(primal[1], dual[1])
                + primal[2] * dual[2] +
                + mass * primal[3] * dual[3])
    def inner_product_nf(primal, dual):
        return (primal[0] * dual[0] + primal[1] * dual[1]
                + l2_ip_nf(primal[2], dual[2]))
    def linear_eq_to_matrix(leq, coef):
        mat, _leq = sp.linear_eq_to_matrix(leq, coef)
        return mat, -_leq


    # compute eigenvalues
    from pyinduct.examples.string_with_mass.utils import find_eigenvalues
    eig_om, eig_vals = find_eigenvalues(n)


    # fill primal base list
    from pyinduct.examples.string_with_mass.observer_evp_scripts.evp_primal import phi0, phi00, real_phi, imag_phi
    real_phi, imag_phi, phi0, phi00 = [
        it.subs(m, mass)
        for it in (real_phi, imag_phi, phi0, phi00)]
    primal_base = [list(phi0), list(phi00)]
    for _om in eig_om[1:]:
        primal_base.append(list(real_phi.subs(om, _om)))
        primal_base.append(list(imag_phi.subs(om, _om)))
    if 1: # manual:
        from pyinduct.examples.string_with_mass.observer_evp_scripts.evp_primal_nf import (
            real_eta as __real_eta, imag_eta as __imag_eta, eta0 as __eta0,
            eta00 as __eta00)
        __real_eta, __imag_eta, __eta0, __eta00 = [
            it.subs(m, mass)
            for it in (__real_eta, __imag_eta, __eta0, __eta00)]
        primal_base_nf = [list(__eta0), list(__eta00)]
        for _om in eig_om[1:]:
            primal_base_nf.append(list(__real_eta.subs(om, _om)))
            primal_base_nf.append(list(__imag_eta.subs(om, _om)))
    else:
        _theta = theta - 1
        raise NotImplementedError(
            "Leads to a differential equation, which coincides with the"
            "eigenvalue problem. Hence, there is no alternetive 'easy way'.")


    # fill dual base list
    from pyinduct.examples.string_with_mass.observer_evp_scripts.evp_dual import psi0, psi00, real_psi, imag_psi
    real_psi, imag_psi, psi0, psi00 = [
        it.subs(m, mass)
        for it in (real_psi, imag_psi, psi0, psi00)]
    dual_base = [list(psi00), list(psi0)]
    for _om in eig_om[1:]:
        dual_base.append(list(real_psi.subs(om, _om)))
        dual_base.append(list(imag_psi.subs(om, _om)))
    if manual:
        from pyinduct.examples.string_with_mass.observer_evp_scripts.evp_dual_nf import (
            real_eta as _real_eta, imag_eta as _imag_eta, eta0 as _eta0,
            eta00 as _eta00)
        _real_eta, _imag_eta, _eta0, _eta00 = [
            it.subs(m, mass)
            for it in (_real_eta, _imag_eta, _eta0, _eta00)]
        dual_base_nf = [list(_eta00), list(_eta0)]
        for _om in eig_om[1:]:
            dual_base_nf.append(list(_real_eta.subs(om, _om)))
            dual_base_nf.append(list(_imag_eta.subs(om, _om)))
    else:
        _theta = theta + 1
        dual_base_flat = [_theta - 1, sp.Integer(1)]
        for _om in eig_om[1:]:
            dual_base_flat.append(
                (sp.cos(om * _theta)).subs(om, _om))
            dual_base_flat.append(
                (sp.sin(om * _theta)).subs(om, _om))
        dual_base_nf = [[f.subs(theta, -1),
                         sp.diff(f, theta).subs(theta, -1),
                         sp.diff(f, theta, theta)]
                        for f in dual_base_flat]


    # build bi-orthonormal base
    for i, (ef, d_ef, d_ef_nf) in enumerate(zip(primal_base, dual_base, dual_base_nf)):
        c = inner_product(ef, d_ef)
        dual_base[i] = [it / c for it in d_ef]
    b = np.array([1, 1, (sp.sign(theta) - 1) * .5 * theta]) * 2 / mass
    for i, _ in enumerate(dual_base_nf):
        if i % 2 == 0:
            scale = dual_base[i + 1][1].subs(z, 1) / inner_product_nf(
                b, dual_base_nf[i + 1])
            dual_base_nf[i] = list(np.array(dual_base_nf[i]) * -scale * (-1 if i == 0 else 1))
            dual_base_nf[i + 1] = list(np.array(dual_base_nf[i + 1]) * scale)


    # print bases
    print("\n primal base")
    pprint(primal_base, discard_sv=False)
    print("\n dual base")
    pprint(dual_base, discard_sv=False)
    print("\n primal normal form base")
    pprint(primal_base_nf, discard_sv=False)
    print("\n dual normal form base")
    pprint(dual_base_nf, discard_sv=False)


    # approximate state
    x = sum_([c * sp.Matrix(f) for c, f in zip(coefficients, primal_base)])
    x1 = np.sum([c * f[0] for c, f in zip(coefficients, primal_base)])
    x2 = np.sum([c * f[1] for c, f in zip(coefficients, primal_base)])
    xi1 = np.sum([c * f[2] for c, f in zip(coefficients, primal_base)])
    xi2 = np.sum([c * f[3] for c, f in zip(coefficients, primal_base)])


    # approximate normal form state
    eta = sum_([c * sp.Matrix(f) for c, f in zip(coefficients, primal_base_nf)])
    eta1 = np.sum([c * f[0] for c, f in zip(coefficients, primal_base_nf)])
    eta2 = np.sum([c * f[1] for c, f in zip(coefficients, primal_base_nf)])
    eta3 = np.sum([c * f[2] for c, f in zip(coefficients, primal_base_nf)])


    # test functions
    psi = [sp.Matrix(tf) for tf in dual_base]
    psi1 = [tf[0] for tf in dual_base]
    psi2 = [tf[1] for tf in dual_base]
    tau1 = [tf[2] for tf in dual_base]
    tau2 = [tf[3] for tf in dual_base]


    # normal form test functions
    nu = [sp.Matrix(tf) for tf in dual_base_nf]
    nu1 = [tf[0] for tf in dual_base_nf]
    nu2 = [tf[1] for tf in dual_base_nf]
    nu3 = [tf[2] for tf in dual_base_nf]


    # observer gain
    a = np.array([0,
                  0,
                  mass ** -1])
    a_bc = np.array([1])
    if spring_damper:
        a0 = mass * (kd1 + 1) / 2
        a1 = a0 ** -1 * (ks0)
        a2 = a0 ** -1 * (ks1 * (2 * kd0 + ks0 + 2) / 2 + kd1 * (kd0 + 1) + ks0 * (1 + kd1))
        a3 = (a0 ** -1 *
              (ks1 * (mass / 2 * sp.sign(theta) + (kd0 + 1) / 2 *(1 - theta) + ks0 / 2 * (sp.sign(theta) * theta ** 2 / 2 - theta + .5)) +
               (kd1 + 1) * (kd0 + 1) / 2 + ks0 * (1 + kd1) * (1 - theta) / 2)
        )
        a_desired = np.array([a1, a2, a3])
        a_bc_desired = np.array([a0 ** -1 * (mass * (1 - kd1) / 2)])
    else:
        a_desired = np.array([(1+alpha) * k0,
                              (1+alpha) * k1 + 2 * k0,
                              k0 * (1 - theta) + k1])
        a_bc_desired = np.array([alpha])
    l = a - a_desired
    l_bc = a_bc - a_bc_desired


    # observer projections
    if 1:
        observer_projections = [
            (inner_product_nf(l, ftf) + l_bc * ftf3.subs(theta, -1)) * yt(t)
            for ftf, ftf3 in tqdm(zip(nu, nu3), total=n * 2)]
    else:
        observer_projections = [
            (-inner_product_nf(a_desired, ftf) - a_bc_desired * ftf3.subs(theta, -1) - ftf3.subs(theta, 1)) * yt(t)
            for ftf, ftf3 in tqdm(zip(nu, nu3), total=n * 2)]


    # just the unbounded part
    L_unbounded = sp.Matrix([[l_bc[0] * ftf3.subs(theta, -1)] for ftf3 in nu3])


    # project test functions on state space
    C = list()
    system_projections = [
        -inner_product(sp.diff(x, t), tf)
        + l2_ip(sp.diff(x2, z), sp.diff(tf1, z))
        + (u(t) * tf2).subs(z, 1)
        - (sp.diff(x1, z) * tf2).subs(z, 0)
        - l2_ip(sp.diff(x1, z), sp.diff(tf2, z))
        + xi2 * t1 + sp.diff(x1, z).subs(z, 0) * t2
        for tf, tf1, tf2, t1, t2
        in tqdm(zip(psi, psi1, psi2, tau1, tau2), total=n * 2)
    ]
    C.append(sp.linear_eq_to_matrix([xi1], coefficients)[0])
    system_projections_nf = [
        -inner_product_nf(sp.diff(eta, t), ftf)
        + eta1 * ftf2
        + l2_ip_nf(eta3, sp.diff(ftf3, theta))
        - eta3.subs(theta, 1) * ftf3.subs(theta, 1)
        + (eta2 - eta3.subs(theta, 1)) * ftf3.subs(theta, -1)
        - eta3.subs(theta, 1) * l2_ip_nf(mass ** -1, ftf3)
        + (2 / mass * ftf1
        + 2 / mass * ftf2
        + l2_ip_nf(-2 / mass * theta, ftf3, lb=-1, ub=0)) * u(t)
        for ftf, ftf1, ftf2, ftf3
        in tqdm(zip(nu, nu1, nu2, nu3), total=n * 2)
    ]
    C.append(sp.linear_eq_to_matrix([eta3.subs(theta, 1)], coefficients)[0])


    # add observer projections to the system projections
    projections = [sp + op for sp, op in zip(system_projections,
                                             observer_projections)]
    projections_nf = [sp + op for sp, op in zip(system_projections_nf,
                                                observer_projections)]


    # parse matrices
    E1, E0, G, J, A, B, L = [[None, None] for _ in range(7)]
    for i, proj in enumerate([projections, projections_nf]):
        E1[i], proj = linear_eq_to_matrix(proj, [sp.diff(c, t) for c in coefficients])
        E0[i], proj = linear_eq_to_matrix(proj, coefficients)
        G[i], proj = linear_eq_to_matrix(proj, [u(t)])
        J[i], proj = linear_eq_to_matrix(proj, [yt(t)])
        if proj != proj * 0:
            print("\n Something went wrong! This is left:")
            pprint(proj, discard_sv=False)
            raise ValueError
        print("\n E1")
        pprint(E1[i])
        print("\n E0")
        pprint(E0[i])

        # compute state space
        np.testing.assert_array_almost_equal(np.eye(n * 2), -E1[i])
        A[i] = E0[i]
        B[i] = G[i]
        L[i] = J[i]


    # display matrices
    print("\n A")
    np.testing.assert_array_almost_equal(A[0], A[1])
    pprint(A[0])
    pprint(A[1])
    print("\n B")
    np.testing.assert_array_almost_equal(B[0], B[1])
    pprint((B[0], B[1]))
    print("\n C")
    np.testing.assert_array_almost_equal(C[0], C[1])
    pprint(C[0])
    pprint(C[1])
    print("\n L")
    np.testing.assert_array_almost_equal(L[0], L[1])
    pprint((L[0], L[1]))


    # compare eigenvalue
    print("\n open loop eigenvalues")
    pprint((np.linalg.eigvals(np.array(A[0]).astype(complex)),
            np.linalg.eigvals(np.array(A[1]).astype(complex)),
            eig_vals))
    if spring_damper:
        if __name__ == "__main__" and n <= 5:
            import pyinduct as pi
            def char_eq(lam):
                return complex(
                    np.exp(lam * 2) * lam ** 2 + a_bc_desired[0] * lam ** 2 +
                    inner_product_nf(sp.Matrix(a_desired), sp.Matrix([1, lam, sp.exp(lam * (theta + 1)) * lam ** 2]))
                )
            char_eq_vec = np.vectorize(char_eq)
            eig_vals_d = sp.Matrix(pi.find_roots(
                char_eq_vec, [np.linspace(-2, 0, n), np.linspace(-10, 10, n)],
                cmplx=True))
        else:
            eig_vals_d = sp.Matrix([0])
    else:
        eig_vals_d = list(np.roots([1, k1, k0]))
        for i in range(1, n):
            if alpha == 0:
                eig_vals_d.append(-sp.oo)
                eig_vals_d.append(-sp.oo)
            elif alpha < 0:
                eig_vals_d.append(np.log(np.abs(alpha)) + 1j * 2 * i * np.pi)
                eig_vals_d.append(np.log(np.abs(alpha)) - 1j * 2 * i * np.pi)
            else:
                eig_vals_d.append(np.log(np.abs(alpha)) + 1j * (2*i - 1) * np.pi)
                eig_vals_d.append(np.log(np.abs(alpha)) + 1j * (-2*i - 1) * np.pi)
    print("\n closed loop eigenvalues")
    eig_vals_cl = [np.linalg.eigvals(np.array(A[0] + L[0] * C[0]).astype(complex)),
                   np.linalg.eigvals(np.array(A[1] + L[1] * C[1]).astype(complex))]
    pprint((eig_vals_cl[0], eig_vals_cl[1], eig_vals_d))

    return primal_base, primal_base_nf, dual_base, dual_base_nf, eig_vals
