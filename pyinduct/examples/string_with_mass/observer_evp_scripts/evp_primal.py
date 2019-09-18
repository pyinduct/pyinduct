import sympy as sp


# functions and symbols
from pyinduct.examples.string_with_mass.utils import sym
m, om, z, lam = sym.m, sym.om, sym.z, sym.lam

# eigenvector for lambda = 0
phi0 = sp.Matrix([1, 0, 1, 0])

# generalized eigenvector for lambda = 0
phi00 = sp.Matrix([0, 1, 0, 1])

# eigenvector for lambda \neq 0
_phi = sp.cos(om * z) - m * om * sp.sin(om * z)
phi = sp.Matrix([_phi, lam * _phi, _phi.subs(z, 0), lam * _phi.subs(z, 0)])
real_phi, imag_phi = phi.subs(lam, 1j * om).expand(complex=True).as_real_imag()
