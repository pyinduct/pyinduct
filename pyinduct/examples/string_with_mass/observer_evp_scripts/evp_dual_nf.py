import sympy as sp


# symbols
from pyinduct.examples.string_with_mass.utils import sym
lam, om, theta, m = sym.lam, sym.om, sym.theta, sym.m

# this base is scaled with x_1(0) = y = eta_1 = 1
eta1 = 1
eta2 = lam * eta1
eta3_tm1 = lam * eta2
eta3 = sp.exp(lam * (theta + 1)) * eta3_tm1
eta = sp.Matrix([eta1, eta2, eta3])

# output u*(t) have to coincide with the flat output x2(1, t) of the dual original system
u = (2 / m * (eta[0] + eta[1] - sp.integrate(theta * eta[2], (theta, -1, 0)))).subs(lam, 1j * om).expand(complex=True)
u_real, u_imag = u.as_real_imag()
scale = 1j * (u_real - 1j * u_imag)
eta = eta *scale

# for lambda = 0
eta0 = eta.subs(lam, 0).subs(om, 0) * 1j
eta00 = sp.diff(eta, lam).subs(lam, 0).subs(om, 0) * 1j
# to meet the modes from the original primal coordinates
eta00 -= eta0

# decomposition
real_eta, imag_eta = eta.subs(lam, 1j * om).expand(complex=True).as_real_imag()



if __name__ == "__main__":
    print("\n eigenvector for lambda = 0")
    sp.pprint(eta0)
    print("\n generalized eigenvector for lambda = 0")
    sp.pprint(eta00)
    print("\n eigenvector for lambda \\neq 0")
    sp.pprint(eta)
    sp.pprint((real_eta, imag_eta))

    from pyinduct.examples.swm_utils import get_eigenvalues
    mass = 1
    _, eig_vals = get_eigenvalues(3, mass)

    ef = eta
    evp = sp.Matrix([
        lam * ef[0] - ef[1],
        lam * ef[1] - ef[2].subs(theta, -1),
        lam * ef[2] - sp.diff(ef[2], theta),
        ef[2].subs(theta, 1) + ef[2].subs(theta, -1) + mass**-1 * sp.integrate(ef[2], (theta, -1, 1), conds="none")
    ])
    sp.pprint(evp.subs(lam, eig_vals[3]).expand(complex=True))

    ef_r = real_eta
    ef_i = imag_eta
    eta_bc = sp.Matrix([ef_r[0], ef_r[1], ef_r[2], 0])
    A = -sp.Matrix([
        - ef_i[1],
        - ef_i[2].subs(theta, -1),
        - sp.diff(ef_i[2], theta),
        ef_i[2].subs(theta, 1) + ef_i[2].subs(theta, -1) + mass**-1 * sp.integrate(ef_i[2], (theta, -1, 1), conds="none")
    ])
    import numpy as np
    sp.pprint(sp.simplify((om*eta_bc - A).subs([(om, np.abs(eig_vals[4]))]).expand()))

    ef_r = real_eta
    ef_i = imag_eta
    eta_bc = sp.Matrix([ef_i[0], ef_i[1], ef_i[2], 0])
    A = -sp.Matrix([
        - ef_r[1],
        - ef_r[2].subs(theta, -1),
        - sp.diff(ef_r[2], theta),
        ef_r[2].subs(theta, 1) + ef_r[2].subs(theta, -1) + mass**-1 * sp.integrate(ef_r[2], (theta, -1, 1), conds="none")
    ])
    sp.pprint(sp.simplify((-om*eta_bc - A).subs([(om, np.abs(eig_vals[3]))]).expand()))

