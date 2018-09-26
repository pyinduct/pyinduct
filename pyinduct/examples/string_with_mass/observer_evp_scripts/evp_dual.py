import sympy as sp


# functions and symbols
from pyinduct.examples.string_with_mass.utils import sym
m, om, z, lam = sym.m, sym.om, sym.z, sym.lam

# eigenvector for lambda = 0
psi0 = sp.Matrix([0, 1, 0, 1])

# generalized eigenvector for lambda = 0
psi00 = sp.Matrix([-z ** 2 / 2 + z + m + 1, 0, m + 1, 0])

# eigenvector for lambda \neq 0
_psi = om ** -1 * sp.sin(om * z) - m ** -1 * om ** -2 * (sp.cos(om * z) - 1)
psi = sp.Matrix([_psi,
                 -lam ** -1 * sp.diff(_psi, z, z),
                 _psi.subs(z, 0),
                 -(lam * m) ** -1 * sp.diff(_psi, z).subs(z, 0)])
real_psi, imag_psi = psi.subs(lam, 1j * om).expand(complex=True).as_real_imag()
