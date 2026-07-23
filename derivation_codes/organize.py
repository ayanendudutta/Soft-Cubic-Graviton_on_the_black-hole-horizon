#!/usr/bin/env python3
"""Organisation of the surviving cubic vertex by trace/K content, and its exact
decomposition in a covariant basis of two-derivative three-point structures.

Input: V3c, the reduced momentum-space cubic vertex on the conservation
surface p3 = -p1 - p2 (produced by cubic.py).  The longitudinal amplitudes are
first traded for the traceless/trace split a = P - Hb/2, c = P + Hb/2, so that
each leg carries the traceless longitudinal polarisation (P, B), the
longitudinal trace Hb, and the transverse scalar K.

Two statements are established here:

  (i) the pure-traceless sector is empty -- every surviving monomial carries at
      least one factor of {Hb, K}, which is the vanishing theorem in the
      organised variables;

 (ii) the vertex is an EXACT rational linear combination of covariant
      structures of three types: three-scalar {Hb,K}^3 (p_i.p_j), one-tensor
      (p_i T_l p_j){Hb,K}^2, and two-tensor (T_a.T_b){Hb,K}(p_i.p_j), together
      with their parity-odd counterparts.

The parity-odd structures are indispensable.  In two dimensions the Levi-Civita
symbol supplies a second invariant of two momenta, the wedge
(p_i ^ p_j) = w_i q_j - q_i w_j, and a dual of the traceless polarisation,
~T_ab = eps_a{}^c T_cb, which for T_tt = T_zz = P, T_tz = B is again symmetric
and traceless with (P, B) -> (B, P).  A parity-even basis alone does NOT span
the vertex: it leaves a residual of relative size O(1), and rationalising an
unconverged least-squares solution then yields coefficients that are fit
artefacts rather than expansion data.  With the dual structures included the
fit closes to machine precision numerically and to an identical zero
symbolically, and the coefficients come out as small rationals.

The linear solve is done by sampling the multilinear vertex at random rational
points, selecting a maximal linearly independent subset of the candidate
structures by rank-revealing QR (so that the coefficients are unique), fitting
by least squares, rationalising, and then asserting that the exact symbolic
residual vanishes identically.
"""
import pickle
import random
from fractions import Fraction

import numpy as np
import sympy as sp
from scipy.linalg import qr

w = sp.symbols('w1:4'); q = sp.symbols('q1:4')
A = sp.symbols('A1:4'); B = sp.symbols('B1:4')
C = sp.symbols('C1:4'); Kk = sp.symbols('K1:4')

with open('V3.pkl', 'rb') as f:
    V3c = pickle.load(f)['V3c']

# change of variables: a = P - Hb/2, c = P + Hb/2
# (traceless longitudinal T_tt = T_zz = P, T_tz = B; longitudinal trace Hb)
P = sp.symbols('P1:4'); Hb = sp.symbols('Hb1:4')
subPT = {}
for j in range(3):
    subPT[A[j]] = P[j] - Hb[j] / 2
    subPT[C[j]] = P[j] + Hb[j] / 2
V = sp.expand(V3c.subs(subPT))

# ---------------------------------------------------------------- theorem
# sanity: no term free of K and Hb (the vanishing theorem in these variables)
pKH = sp.Poly(V, *Kk, *Hb)
pure = sum(cf for mono, cf in zip(pKH.monoms(), pKH.coeffs()) if sum(mono) == 0)
print("pure-T sector (must be 0):", sp.simplify(pure))
assert sp.simplify(pure) == 0

# ------------------------------------------------- covariant building blocks
# momenta: p1, p2 independent (p3 = -p1 - p2 already imposed); eta = diag(-1,1)
legs = [0, 1, 2]
mom_pairs = [(0, 0), (0, 1), (1, 1)]


def scal(i, which):
    """Scalar polarisation of leg i: the transverse scalar K or the trace Hb."""
    return Kk[i] if which == 'K' else Hb[i]


def dot(i, j):
    """p_i . p_j with eta = diag(-1, +1)."""
    return -w[i] * w[j] + q[i] * q[j]


def wedge(i, j):
    """Two-dimensional Levi-Civita invariant eps^{ab} p_ia p_jb (up to sign):
    the parity-odd partner of dot(i, j).  Antisymmetric, so only (1, 2)
    contributes on the conservation surface."""
    return w[i] * q[j] - q[i] * w[j]


def pTp(i, l, j):
    """p_i^a T_{l,ab} p_j^b with lower-index momenta p_a = (w, q) and
    p^t = -w, p^z = q."""
    return P[l] * (w[i] * w[j] + q[i] * q[j]) - B[l] * (w[i] * q[j] + q[i] * w[j])


def pTdualp(i, l, j):
    """p_i^a ~T_{l,ab} p_j^b, the same contraction on the Levi-Civita dual
    polarisation ~T (obtained from T by (P, B) -> (B, P))."""
    return B[l] * (w[i] * w[j] + q[i] * q[j]) - P[l] * (w[i] * q[j] + q[i] * w[j])


def TT(i, j):
    """T_i^{ab} T_{j,ab}."""
    return 2 * (P[i] * P[j] - B[i] * B[j])


def TTdual(i, j):
    """T_i^{ab} ~T_{j,ab}; antisymmetric in the two legs."""
    return 2 * (P[i] * B[j] - B[i] * P[j])


# ------------------------------------------------------- candidate structures
basis, names = [], []


def add(expr, name):
    basis.append(expr)
    names.append(name)


# --- three scalars, parity even and odd ---
for c1 in 'KH':
    for c2 in 'KH':
        for c3 in 'KH':
            s3 = scal(0, c1) * scal(1, c2) * scal(2, c3)
            for (i, j) in mom_pairs:
                add(s3 * dot(i, j), f"{c1}1 {c2}2 {c3}3 (p{i+1}.p{j+1})")
            add(s3 * wedge(0, 1), f"{c1}1 {c2}2 {c3}3 (p1^p2)")

# --- one tensor (or its dual), two scalars ---
for tl in legs:
    others = [x for x in legs if x != tl]
    for c1 in 'KH':
        for c2 in 'KH':
            sc = scal(others[0], c1) * scal(others[1], c2)
            tag = f"{c1}{others[0]+1} {c2}{others[1]+1}"
            for (i, j) in mom_pairs:
                add(pTp(i, tl, j) * sc, f"(p{i+1} T{tl+1} p{j+1}) {tag}")
                add(pTdualp(i, tl, j) * sc, f"(p{i+1} ~T{tl+1} p{j+1}) {tag}")

# --- two tensors (or a tensor and a dual), one scalar ---
for sl in legs:
    others = [x for x in legs if x != sl]
    for ch in 'KH':
        sc = scal(sl, ch)
        tag = f"{ch}{sl+1}"
        for (i, j) in mom_pairs:
            add(TT(others[0], others[1]) * sc * dot(i, j),
                f"(T{others[0]+1}.T{others[1]+1}) {tag} (p{i+1}.p{j+1})")
            add(TTdual(others[0], others[1]) * sc * dot(i, j),
                f"(T{others[0]+1}.~T{others[1]+1}) {tag} (p{i+1}.p{j+1})")
        add(TT(others[0], others[1]) * sc * wedge(0, 1),
            f"(T{others[0]+1}.T{others[1]+1}) {tag} (p1^p2)")
        add(TTdual(others[0], others[1]) * sc * wedge(0, 1),
            f"(T{others[0]+1}.~T{others[1]+1}) {tag} (p1^p2)")

print("candidate covariant structures:", len(basis))

# ------------------------------------------------------------- numerical fit
random.seed(1)
vars_all = (list(w[:2]) + list(q[:2]) + list(P) + list(B) + list(Hb) + list(Kk))
V_l = sp.lambdify(vars_all, V, 'math')
basis_l = [sp.lambdify(vars_all, bexpr, 'math') for bexpr in basis]

NS = len(basis) + 150
Msys = np.zeros((NS, len(basis)))
rvec = np.zeros(NS)
for r in range(NS):
    vals = [random.uniform(-1, 1) for _ in vars_all]
    for cidx, bl in enumerate(basis_l):
        Msys[r, cidx] = bl(*vals)
    rvec[r] = V_l(*vals)

# a maximal independent subset makes the coefficients unique
rank = int(np.linalg.matrix_rank(Msys, tol=1e-9))
_, _, piv = qr(Msys, mode='economic', pivoting=True)
keep = sorted(int(k) for k in piv[:rank])
Msub = Msys[:, keep]
sol, _, _, _ = np.linalg.lstsq(Msub, rvec, rcond=None)
err = np.max(np.abs(Msub @ sol - rvec))
print(f"independent structures: {rank} of {len(basis)}; "
      f"lstsq max residual: {err:.3e}")
assert err < 1e-9, "covariant basis does not span the vertex"

# ---------------------------------------------------------- rationalise/verify
sol_r = [Fraction(x).limit_denominator(10 ** 6) for x in sol]
Vfit = sum(sp.Rational(fr.numerator, fr.denominator) * basis[k]
           for fr, k in zip(sol_r, keep))
diff = sp.simplify(sp.expand(V - Vfit))
print("exact symbolic residual:", diff)
assert diff == 0, "rationalised fit is not exact"

print("\n=== Surviving cubic vertex: exact covariant decomposition ===")
for fr, k in zip(sol_r, keep):
    if fr != 0:
        print(f"  {str(fr):>8s}  *  {names[k]}")

with open('fit.pkl', 'wb') as f:
    pickle.dump({'sol': sol_r, 'keep': keep, 'names': names, 'basis': basis}, f)
print("\nwrote fit.pkl")
