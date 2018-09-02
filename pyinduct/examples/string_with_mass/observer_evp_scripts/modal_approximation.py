from pyinduct.examples.string_with_mass.utils import sym, param, obs_gain
from sympy.utilities import lambdify
from scipy.integrate import quad
from tqdm import tqdm
import numpy as np
import sympy as sp


def _sum(iterable):
    sum = iterable[0] * 0
    for v in iterable:
        sum += v
    return sum


def _discard_small_values(mat, eps=10 ** -6):
    mat = np.array(mat).astype(complex)
    rmat = np.real(mat)
    rmat[np.abs(rmat) < eps] = 0
    imat = np.imag(mat)
    imat[np.abs(imat) < eps] = 0
    return np.real_if_close(mat)


def _pprint(item, discard_sv=True):
    if discard_sv and isinstance(item, (np.ndarray, sp.Matrix)):
        item = sp.Matrix(_discard_small_values(item))
    elif discard_sv and isinstance(item, (list, tuple)):
        item = [sp.Matrix(_discard_small_values(it)) for it in item]
    else:
        item = sp.Matrix(item)
    sp.pprint(item, num_columns=200)


def _numeric_integration(f1, f2, bounds, coef):
    iv, lb, ub = bounds
    f1 = sp.sympify(f1)
    if any([c in f1.atoms(sp.Function) or sp.diff(c, sym.t) in f1.atoms(sp.Function)
            for c in coef]):
        vec, _ = _linear_eq_to_matrix([f1], coef)
        if_ = [lambdify(iv, el * f2, modules="numpy") for el in vec]
        res = np.sum(
            [quad(f, lb, ub)[0] * c for c, f in zip(coef, if_)])
        vec, _ = _linear_eq_to_matrix([f1],
                                      [sp.diff(c, sym.t) for c in coef])
        if_ = [lambdify(iv, el * f2, modules="numpy") for el in vec]
        res += np.sum([quad(f, lb, ub)[0] * sp.diff(c, sym.t) for c, f in
                       zip(coef, if_)])
    else:
        if_ = lambdify(iv, (f1 * f2).doit(), modules="numpy")
        res = quad(if_, lb, ub)[0]
    return res


def _l2_ip(f1, f2, coef):
    return _numeric_integration(f1, f2, (sym.z, 0, 1), coef)


def _l2_ip_nf(f1, f2, coef, lb=-1, ub=1):
    return _numeric_integration(f1, f2, (sym.theta, lb, ub), coef)


def _inner_product(primal, dual, coef):
    return (_l2_ip(sp.diff(primal[0], sym.z), sp.diff(dual[0], sym.z), coef)
            + _l2_ip(primal[1], dual[1], coef)
            + primal[2] * dual[2] +
            + param.m * primal[3] * dual[3])


def _inner_product_nf(primal, dual, coef):
    return (primal[0] * dual[0] + primal[1] * dual[1]
            + _l2_ip_nf(primal[2], dual[2], coef))


def _linear_eq_to_matrix(leq, coef):
    mat, _leq = sp.linear_eq_to_matrix(leq, coef)
    return mat, -_leq


def build_bases_for_modal_observer_approximation(m):
    if m % 2 == 1:
        raise ValueError("Only even number of eigenvalues supported.")

    n = int(m / 2)
    coef = [c(sym.t) for c in sp.symbols("c_:{}".format(n*2))]

    # solve eigenvalue problems in  normal form coordinates by hand: manual = 1
    # or derive from the solutions in original coordinates: manual = 0
    manual = 1


    # compute eigenvalues
    from pyinduct.examples.string_with_mass.utils import find_eigenvalues
    eig_om, eig_vals = find_eigenvalues(n)


    # fill primal base list
    from pyinduct.examples.string_with_mass.observer_evp_scripts.evp_primal import phi0, phi00, real_phi, imag_phi
    real_phi, imag_phi, phi0, phi00 = [
        it.subs(sym.m, param.m)
        for it in (real_phi, imag_phi, phi0, phi00)]
    primal_base = [list(phi0), list(phi00)]
    for _om in eig_om[1:]:
        primal_base.append(list(real_phi.subs(sym.om, _om)))
        primal_base.append(list(imag_phi.subs(sym.om, _om)))
    if 1: # manual:
        from pyinduct.examples.string_with_mass.observer_evp_scripts.evp_primal_nf import (
            real_eta as __real_eta, imag_eta as __imag_eta, eta0 as __eta0,
            eta00 as __eta00)
        __real_eta, __imag_eta, __eta0, __eta00 = [
            it.subs(sym.m, param.m)
            for it in (__real_eta, __imag_eta, __eta0, __eta00)]
        primal_base_nf = [list(__eta0), list(__eta00)]
        for _om in eig_om[1:]:
            primal_base_nf.append(list(__real_eta.subs(sym.om, _om)))
            primal_base_nf.append(list(__imag_eta.subs(sym.om, _om)))
    else:
        _theta = sym.theta - 1
        raise NotImplementedError(
            "Leads to a differential equation, which coincides with the"
            "eigenvalue problem. Hence, there is no alternetive 'easy way'.")


    # fill dual base list
    from pyinduct.examples.string_with_mass.observer_evp_scripts.evp_dual import psi0, psi00, real_psi, imag_psi
    real_psi, imag_psi, psi0, psi00 = [
        it.subs(sym.m, param.m)
        for it in (real_psi, imag_psi, psi0, psi00)]
    dual_base = [list(psi00), list(psi0)]
    for _om in eig_om[1:]:
        dual_base.append(list(real_psi.subs(sym.om, _om)))
        dual_base.append(list(imag_psi.subs(sym.om, _om)))
    if manual:
        from pyinduct.examples.string_with_mass.observer_evp_scripts.evp_dual_nf import (
            real_eta as _real_eta, imag_eta as _imag_eta, eta0 as _eta0,
            eta00 as _eta00)
        _real_eta, _imag_eta, _eta0, _eta00 = [
            it.subs(sym.m, param.m)
            for it in (_real_eta, _imag_eta, _eta0, _eta00)]
        dual_base_nf = [list(_eta00), list(_eta0)]
        for _om in eig_om[1:]:
            dual_base_nf.append(list(_real_eta.subs(sym.om, _om)))
            dual_base_nf.append(list(_imag_eta.subs(sym.om, _om)))
    else:
        _theta = sym.theta + 1
        dual_base_flat = [_theta - 1, sp.Integer(1)]
        for _om in eig_om[1:]:
            dual_base_flat.append(
                (sp.cos(sym.om * _theta)).subs(sym.om, _om))
            dual_base_flat.append(
                (sp.sin(sym.om * _theta)).subs(sym.om, _om))
        dual_base_nf = [[f.subs(sym.theta, -1),
                         sp.diff(f, sym.theta).subs(sym.theta, -1),
                         sp.diff(f, sym.theta, sym.theta)]
                        for f in dual_base_flat]


    # build bi-orthonormal base
    for i, (ef, d_ef, d_ef_nf) in enumerate(zip(primal_base, dual_base, dual_base_nf)):
        c = _inner_product(ef, d_ef, coef)
        dual_base[i] = [it / c for it in d_ef]
    b = np.array([1, 1, (sp.sign(sym.theta) - 1) * .5 * sym.theta]) * 2 / param.m
    for i, _ in enumerate(dual_base_nf):
        if i % 2 == 0:
            scale = dual_base[i + 1][1].subs(sym.z, 1) / _inner_product_nf(
                b, dual_base_nf[i + 1], coef)
            dual_base_nf[i] = list(np.array(dual_base_nf[i]) * -scale * (-1 if i == 0 else 1))
            dual_base_nf[i + 1] = list(np.array(dual_base_nf[i + 1]) * scale)


    # print bases
    print("\n primal base")
    _pprint(primal_base, discard_sv=False)
    print("\n dual base")
    _pprint(dual_base, discard_sv=False)
    print("\n primal normal form base")
    _pprint(primal_base_nf, discard_sv=False)
    print("\n dual normal form base")
    _pprint(dual_base_nf, discard_sv=False)


    return primal_base, primal_base_nf, dual_base, dual_base_nf, eig_vals


def get_observer_gain(spring_damper_params=list()):

    # observer gain
    a = np.array([0,
                  0,
                  param.m ** -1])
    a_bc = np.array([1])
    if len(spring_damper_params) > 0:
        kd0, ks0, kd1, ks1 = spring_damper_params

        a0 = param.m * (kd1 + 1) / 2
        a1 = a0 ** -1 * (ks0)
        a2 = a0 ** -1 * (ks1 * (2 * kd0 + ks0 + 2) / 2 + kd1 * (kd0 + 1) + ks0 * (1 + kd1))
        a3 = (a0 ** -1 *
              (ks1 * (param.m / 2 * sp.sign(sym.theta) + (kd0 + 1) / 2 *(1 - sym.theta) + ks0 / 2 * (sp.sign(sym.theta) * sym.theta ** 2 / 2 - sym.theta + .5)) +
               (kd1 + 1) * (kd0 + 1) / 2 + ks0 * (1 + kd1) * (1 - sym.theta) / 2)
              )
        a_desired = np.array([a1, a2, a3])
        a_bc_desired = np.array([a0 ** -1 * (param.m * (1 - kd1) / 2)])
    else:
        a_desired = np.array([(1+obs_gain.alpha) * obs_gain.k0,
                              (1+obs_gain.alpha) * obs_gain.k1 + 2 * obs_gain.k0,
                              obs_gain.k0 * (1 - sym.theta) + obs_gain.k1])
        a_bc_desired = np.array([obs_gain.alpha])
    l = a - a_desired
    l_bc = a_bc - a_bc_desired

    return l, l_bc


def validate_modal_bases(primal_base, primal_base_nf, dual_base, dual_base_nf,
                         eig_vals):
    m = len(primal_base)
    n = int(m / 2)
    assert all([len(it_) == m for it_ in (primal_base_nf, dual_base, dual_base_nf, eig_vals)])

    coef = [c(sym.t) for c in sp.symbols("c_:{}".format(n*2))]

    # approximate state
    x = _sum([c * sp.Matrix(f) for c, f in zip(coef, primal_base)])
    x1 = np.sum([c * f[0] for c, f in zip(coef, primal_base)])
    x2 = np.sum([c * f[1] for c, f in zip(coef, primal_base)])
    xi1 = np.sum([c * f[2] for c, f in zip(coef, primal_base)])
    xi2 = np.sum([c * f[3] for c, f in zip(coef, primal_base)])


    # approximate normal form state
    eta = _sum([c * sp.Matrix(f) for c, f in zip(coef, primal_base_nf)])
    eta1 = np.sum([c * f[0] for c, f in zip(coef, primal_base_nf)])
    eta2 = np.sum([c * f[1] for c, f in zip(coef, primal_base_nf)])
    eta3 = np.sum([c * f[2] for c, f in zip(coef, primal_base_nf)])


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


    # choose kind of desired dynamic
    spring_damper = 0
    if spring_damper:
        kd0, ks0, kd1, ks1 = [1] * 4
        spring_damper_params = kd0, ks0, kd1, ks1
    else:
        k0, k1, alpha = 9, 10, 0
        spring_damper_params = list()
    l, l_bc = get_observer_gain(spring_damper_params)

    # observer projections
    if 1:
        observer_projections = [
            (_inner_product_nf(l, ftf, coef) + l_bc * ftf3.subs(sym.theta, -1)) * sym.yt(sym.t)
            for ftf, ftf3 in tqdm(zip(nu, nu3), total=n * 2)]
    else:
        observer_projections = [
            (-_inner_product_nf(a_desired, ftf, coef) - a_bc_desired * ftf3.subs(sym.theta, -1) - ftf3.subs(sym.theta, 1)) * sym.yt(sym.t)
            for ftf, ftf3 in tqdm(zip(nu, nu3), total=n * 2)]


    # just the unbounded part
    L_unbounded = sp.Matrix([[l_bc[0] * ftf3.subs(sym.theta, -1)] for ftf3 in nu3])


    # project test functions on state space
    C = list()
    system_projections = [
        -_inner_product(sp.diff(x, sym.t), tf, coef)
        + _l2_ip(sp.diff(x2, sym.z), sp.diff(tf1, sym.z), coef)
        + (sym.u(sym.t) * tf2).subs(sym.z, 1)
        - (sp.diff(x1, sym.z) * tf2).subs(sym.z, 0)
        - _l2_ip(sp.diff(x1, sym.z), sp.diff(tf2, sym.z), coef)
        + xi2 * t1 + sp.diff(x1, sym.z).subs(sym.z, 0) * t2
        for tf, tf1, tf2, t1, t2
        in tqdm(zip(psi, psi1, psi2, tau1, tau2), total=n * 2)
    ]
    C.append(sp.linear_eq_to_matrix([xi1], coef)[0])
    system_projections_nf = [
        -_inner_product_nf(sp.diff(eta, sym.t), ftf, coef)
        + eta1 * ftf2
        + _l2_ip_nf(eta3, sp.diff(ftf3, sym.theta), coef)
        - eta3.subs(sym.theta, 1) * ftf3.subs(sym.theta, 1)
        + (eta2 - eta3.subs(sym.theta, 1)) * ftf3.subs(sym.theta, -1)
        - eta3.subs(sym.theta, 1) * _l2_ip_nf(param.m ** -1, ftf3, coef)
        + (2 / param.m * ftf1
           + 2 / param.m * ftf2
           + _l2_ip_nf(-2 / param.m * sym.theta, ftf3, coef, lb=-1, ub=0)) * sym.u(sym.t)
        for ftf, ftf1, ftf2, ftf3
        in tqdm(zip(nu, nu1, nu2, nu3), total=n * 2)
    ]
    C.append(sp.linear_eq_to_matrix([eta3.subs(sym.theta, 1)], coef)[0])


    # add observer projections to the system projections
    projections = [sp + op for sp, op in zip(system_projections,
                                             observer_projections)]
    projections_nf = [sp + op for sp, op in zip(system_projections_nf,
                                                observer_projections)]


    # parse matrices
    E1, E0, G, J, A, B, L = [[None, None] for _ in range(7)]
    for i, proj in enumerate([projections, projections_nf]):
        E1[i], proj = _linear_eq_to_matrix(proj, [sp.diff(c, sym.t) for c in coef])
        E0[i], proj = _linear_eq_to_matrix(proj, coef)
        G[i], proj = _linear_eq_to_matrix(proj, [sym.u(sym.t)])
        J[i], proj = _linear_eq_to_matrix(proj, [sym.yt(sym.t)])
        if proj != proj * 0:
            print("\n Something went wrong! This is left:")
            _pprint(proj, discard_sv=False)
            raise ValueError
        print("\n E1")
        _pprint(E1[i])
        print("\n E0")
        _pprint(E0[i])

        # compute state space
        np.testing.assert_array_almost_equal(np.eye(n * 2), -E1[i])
        A[i] = E0[i]
        B[i] = G[i]
        L[i] = J[i]


    # display matrices
    print("\n A")
    np.testing.assert_array_almost_equal(A[0], A[1])
    _pprint(A[0])
    _pprint(A[1])
    print("\n B")
    np.testing.assert_array_almost_equal(B[0], B[1])
    _pprint((B[0], B[1]))
    print("\n C")
    np.testing.assert_array_almost_equal(C[0], C[1])
    _pprint(C[0])
    _pprint(C[1])
    print("\n L")
    np.testing.assert_array_almost_equal(L[0], L[1])
    _pprint((L[0], L[1]))


    # compare eigenvalue
    print("\n open loop eigenvalues")
    _pprint((np.linalg.eigvals(np.array(A[0]).astype(complex)),
             np.linalg.eigvals(np.array(A[1]).astype(complex)),
             eig_vals))
    if spring_damper:
        if __name__ == "__main__" and n <= 5:
            import pyinduct as pi
            def char_eq(lam):
                return complex(
                    np.exp(lam * 2) * lam ** 2 + a_bc_desired[0] * lam ** 2 +
                    _inner_product_nf(sp.Matrix(a_desired), sp.Matrix([1, lam, sp.exp(lam * (sym.theta + 1)) * lam ** 2]), coef)
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
    _pprint((eig_vals_cl[0], eig_vals_cl[1], eig_vals_d))

    return A[0], B[0], C[0], L[0], L_unbounded
