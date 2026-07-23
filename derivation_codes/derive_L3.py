#!/usr/bin/env python3
"""
Derivation of the cubic even-parity near-horizon graviton vertex at strict
leading-soft order (longitudinal derivatives only, flat near-horizon limit).

Setup (per harmonic, Y -> const so the Gaunt factor is a spectator):
  background: eta = diag(-1, +1, +1, +1), coords (t, z, y1, y2)
  perturbation (RW gauge, even parity, toolbox 2.58):
    delta g_ab = kappa * H_ab(t,z),  a,b in {t,z}:  H_tt=a, H_tz=b, H_zz=c
    delta g_AB = kappa * K(t,z) * delta_AB          (h+_AB = K g_AB Y)
  action: S = 1/(2 kappa^2) Integral sqrt(-g) R   (convention of the paper;
          S2 = -1/4 h G1)
  matter: S_m = -1/2 Integral sqrt(-g) g^{mu nu} d phi d phi, phi = phi(t,z) Y

Outputs: quadratic Lagrangian L2 and cubic Lagrangian L3 as momentum-space
vertices (multilinear in plane-wave legs), reduced on the conservation surface.
"""
import sympy as sp
import itertools, pickle, sys

t, z = sp.symbols('t z', real=True)
kap = sp.symbols('kappa', positive=True)
X = (t, z)

# field functions
a = sp.Function('a')(t, z)   # H_tt
b = sp.Function('b')(t, z)   # H_tz
c = sp.Function('c')(t, z)   # H_zz
k = sp.Function('k')(t, z)   # K (transverse scalar)
ph = sp.Function('phi')(t, z)  # matter scalar (per harmonic)

eta = sp.diag(-1, 1, 1, 1)
h = sp.zeros(4, 4)
h[0,0], h[0,1], h[1,0], h[1,1] = a, b, b, c
h[2,2] = k
h[3,3] = k

g = eta + kap*h

coords = [t, z, sp.Symbol('y1'), sp.Symbol('y2')]  # fields do not depend on y1,y2

ORDER = 4  # keep through kappa^3

def series_trunc(expr, n=ORDER):
    return sp.expand(sp.series(sp.expand(expr), kap, 0, n).removeO())

# inverse metric to O(kappa^3): eta - kap h + kap^2 h.h - kap^3 h.h.h (indices via eta)
hu = eta.inv()*h  # h^mu_nu
ginv = (sp.eye(4) - kap*hu + kap**2*hu*hu - kap**3*hu*hu*hu)*eta.inv()

# sqrt(-g) via exp(1/2 tr log(1 + kap h^mu_nu))
trh  = sp.trace(hu)
trh2 = sp.trace(hu*hu)
trh3 = sp.trace(hu*hu*hu)
sqrtg = 1 + kap*trh/2 + kap**2*(trh**2/8 - trh2/4) + kap**3*(trh**3/48 - trh*trh2/8 + trh3/6)

def d(expr, mu):
    v = coords[mu]
    if v in (t, z):
        return sp.diff(expr, v)
    return sp.Integer(0)

print("building Christoffels...", flush=True)
Gam = [[[sp.Integer(0)]*4 for _ in range(4)] for _ in range(4)]
for rho in range(4):
    for mu in range(4):
        for nu in range(mu, 4):
            s = sp.Integer(0)
            for sig in range(4):
                s += ginv[rho, sig]*(d(g[sig, mu], nu) + d(g[sig, nu], mu) - d(g[mu, nu], sig))
            s = series_trunc(s/2)
            Gam[rho][mu][nu] = s
            Gam[rho][nu][mu] = s

print("building Ricci...", flush=True)
Ric = sp.zeros(4, 4)
for mu in range(4):
    for nu in range(mu, 4):
        s = sp.Integer(0)
        for rho in range(4):
            s += d(Gam[rho][mu][nu], rho) - d(Gam[rho][mu][rho], nu)
            for lam in range(4):
                s += Gam[rho][rho][lam]*Gam[lam][mu][nu] - Gam[rho][nu][lam]*Gam[lam][mu][rho]
        s = series_trunc(sp.expand(s))
        Ric[mu, nu] = s
        Ric[nu, mu] = s

Rs = sp.Integer(0)
for mu in range(4):
    for nu in range(4):
        Rs += ginv[mu, nu]*Ric[mu, nu]
Rs = series_trunc(Rs)

L = series_trunc(sqrtg*Rs)
Lc = sp.Poly(L, kap).all_coeffs()[::-1]  # coeffs [kap^0, kap^1, ...]
while len(Lc) < ORDER:
    Lc.append(sp.Integer(0))
L1, L2, L3 = sp.expand(Lc[1]), sp.expand(Lc[2]), sp.expand(Lc[3])
# gravitational action S = 1/(2 kap^2) * Integral sqrt(-g) R
# => S^(2) = (1/2) Int L2 ; S^(3) = (kap/2) Int L3
print("L1 (should be total derivative / zero on flat bg):")
print(sp.simplify(L1))

with open('L23.pkl','wb') as f:
    pickle.dump({'L2': L2, 'L3': L3}, f)
print("done; L2 terms:", len(L2.args), " L3 terms:", len(L3.args))
