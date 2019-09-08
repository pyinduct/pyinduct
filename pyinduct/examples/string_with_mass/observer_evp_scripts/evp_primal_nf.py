import sympy as sp


# functions and symbols
from pyinduct.examples.string_with_mass.utils import sym
m, lam, om, theta = sym.m, sym.lam, sym.om, sym.theta
eta = sp.Function("eta", real=True)(theta)
tau = sp.Function("tau", real=True)(theta)
epsilon = sp.Symbol("epsilon", real=True)

# eigenvector for lambda = 0
eta10 = 0
eta30 = eta.subs(theta, -1) - m**-1 * eta.subs(theta, 1) * (theta + 1)
assert sp.diff(eta30, theta) == - m**-1 * eta.subs(theta, 1)
eta30_tm1 = sp.solve(eta30.subs(theta, 1) - eta.subs(theta, 1), eta.subs(theta, -1))[0]
eta30 = eta30.subs(eta.subs(theta, -1), eta30_tm1)
eta20 = sp.simplify(eta30.subs(theta, -1) + eta30.subs(theta, 1))
eta0 = (sp.Matrix([eta10, eta20, eta30])).subs(eta.subs(theta, 1), 1)

# generalized eigenvector for lambda = 0
eta100 = -eta20
eta200 = 0
eta300 = (tau.subs(theta, -1)
          + sp.diff(tau, theta).subs(theta, -1) * (theta + 1)
          + sp.diff(tau, theta, theta).subs(theta, -1) / 2 * (theta + 1) ** 2)
d_eta300 = - m**-1 * tau.subs(theta, 1) + eta30
eta300 = eta300.subs(sp.diff(tau, theta).subs(theta, -1),
                     d_eta300.subs(theta, -1))
eta300 = eta300.subs(sp.diff(tau, theta, theta).subs(theta, -1),
                     sp.diff(d_eta300, theta).subs(theta, -1))
eta300 = eta300.subs(tau.subs(theta, -1), -tau.subs(theta, 1))
eta300_tp1 = sp.solve(eta300.subs(theta, 1) - tau.subs(theta, 1),
                      tau.subs(theta, 1))[0]
eta300 = eta300.subs(tau.subs(theta, 1), eta300_tp1)
eta00 = (sp.Matrix([eta100, eta200, eta300])).subs(eta.subs(theta, 1), 1)
# to meet the modes from the original primal coordinates
eta00 -= eta0
# to gain jordan form
eta00 *= -1

# eigenvectors for lambda \neq 0
eta1 = 0
eta2 = 0
eta3 = (sp.exp(-lam * (theta + 1)) * eta.subs(theta, -1)
        - m ** -1 * eta.subs(theta, 1) * sp.integrate(
                sp.exp(-lam * (theta - epsilon)),
                (epsilon, -1, theta), conds="none"))
eta3_tm1 = sp.solve(eta3.subs(theta, 1) - eta.subs(theta, 1), eta.subs(theta, -1))[0]
eta3 = eta3.subs(eta.subs(theta, -1), eta3_tm1)
eta = sp.Matrix([eta1, eta2, eta3]).subs(eta.subs(theta, 1), 1).subs(eta.subs(theta, 1), 1)

# decomposition
# real_eta, imag_eta = eta.subs(lam, 1j * om).expand(complex=True).as_real_imag()
# real_eta, imag_eta = sp.simplify(real_eta), sp.simplify(imag_eta)

# nice representation of the decompositions
real_eta = sp.Matrix([
    0, 0,
    sp.cos(om * (theta - 1)) - (m * om) ** -1 * sp.sin(om * (theta - 1))
])
imag_eta = sp.Matrix([
    0, 0,
    -sp.sin(om * (theta - 1)) - (m * om) ** -1 * (sp.cos(om * (theta - 1)) - 1)
])


if __name__ == "__main__":
    print("\n eigenvector for lambda = 0")
    sp.pprint(eta0)
    print("\n generalized eigenvector for lambda = 0")
    sp.pprint(eta00)
    print("\n eigenvector for lambda \\neq 0")
    sp.pprint(eta, num_columns=500)
    sp.pprint(real_eta, num_columns=500)
    sp.pprint(imag_eta, num_columns=500)

    # check solution of the eigenvalue problem
    from pyinduct.examples.swm_utils import get_eigenvalues
    mass = 1
    _, eig_vals = get_eigenvalues(3, mass)

    ef = eta
    eta_bc = sp.Matrix([ef[0], ef[1], ef[2], 0])
    A = -sp.Matrix([
        0,
        -ef[0],
        sp.diff(ef[2], theta) + m ** -1 * ef[2].subs(theta, 1),
        ef[1] - ef[2].subs(theta, 1) - ef[2].subs(theta, -1)
    ])
    sp.pprint((lam*eta_bc - A).subs([(lam, eig_vals[3]), (m, mass)]).expand(complex=True))

    # check (real part of the) solution of the eigenvalue problem
    ef_r = real_eta
    ef_i = imag_eta
    eta_bc = sp.Matrix([ef_r[0], ef_r[1], ef_r[2], 0])
    A = -sp.Matrix([
        0,
        -ef_i[0],
        sp.diff(ef_i[2], theta) + m ** -1 * ef_i[2].subs(theta, 1),
        ef_i[1] - ef_i[2].subs(theta, 1) - ef_i[2].subs(theta, -1)
    ])
    import numpy as np
    sp.pprint(sp.simplify((om*eta_bc - A).subs([(om, np.abs(eig_vals[4])), (m, mass)]).expand()))

    # check (imaginary part of the) solution of the eigenvalue problem
    ef_r = real_eta
    ef_i = imag_eta
    eta_bc = sp.Matrix([ef_i[0], ef_i[1], ef_i[2], 0])
    A = -sp.Matrix([
        0,
        -ef_r[0],
        sp.diff(ef_r[2], theta) + m ** -1 * ef_r[2].subs(theta, 1),
        ef_r[1] - ef_r[2].subs(theta, 1) - ef_r[2].subs(theta, -1)
    ])
    sp.pprint(sp.simplify((-om*eta_bc - A).subs([(om, np.abs(eig_vals[3])), (m, mass)]).expand()))
