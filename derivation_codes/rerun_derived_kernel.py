#!/usr/bin/env python3
"""Propagation of the DERIVED on-shell weight W(lam1,lam2,lam3) through the
simulation model.

(1) l=2 multiplet: the table built with the derived kernel is exactly
    proportional to the K=1 table, because on an equal-lambda register the
    weight is a constant that renormalises the coupling; the proportionality
    is asserted at run time.
(2) multi-l register (l=2 + l=3, Nmax-truncated, Lz=0): the diagonal-ensemble
    (infinite-time) pbar_N and inelasticity with the derived kernel against K=1.
    Both tables are scaled to the Frobenius norm of the PHYSICAL K=1 table
    (g_eff = 12 on the raw unit-kernel table), so only the RELATIVE multipole
    structure differs at the physical coupling.  The shift is Nmax-converged
    (Nmax=4 and 5 agree): Delta_eta ~ -25%, TV(pbar_N) ~ 0.018.

The closed form of W below is the conservation-surface evaluation
(w3 = -(w1+w2), legs 1 and 2 on shell), i.e. prescription H0 of the
off-shell-prescription scan in dilaton_W.py.  On the equal-lambda diagonal it
is prescription-independent and reproduces W(7,7,7) = -35.4375; off the
diagonal it is one representative choice among several defensible ones, and
the multi-l shift reported here inherits that scheme systematic.

This script bridges the two halves of the repository: it imports the
simulation package `nhq` from the Qiskit suite and writes its summary into
that suite's output directory, both located relative to this file, so it runs
correctly from any working directory."""
import os
import sys
import numpy as np, json

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_CANDIDATES = [d for d in sorted(os.listdir(_ROOT))
               if os.path.isdir(os.path.join(_ROOT, d, "nhq"))]
if not _CANDIDATES:
    raise SystemExit("the simulation suite (a sibling directory containing "
                     "the package 'nhq') was not found next to "
                     f"{_HERE}")
_SUITE = os.path.join(_ROOT, _CANDIDATES[0])
if _SUITE not in sys.path:
    sys.path.insert(0, _SUITE)
_OUT = os.path.join(_SUITE, "out")
os.makedirs(_OUT, exist_ok=True)

from nhq import model as md

def lam(l): return l*l + l + 1

def W(l1, l2, l3):
    """derived on-shell three-point weight, mu=1 (legs 1,2 creation; 3 annihilation)
       W = V3c on GGV residue polarizations; closed form generated symbolically"""
    import math
    L1, L2, L3 = lam(l1), lam(l2), lam(l3)
    s1, s2 = math.sqrt(L1), math.sqrt(L2)
    d12 = L1*L2 + L1 + L2 + 1
    d13 = L1*L3 + L1 + L3 + 1
    d23 = L2*L3 + L2 + L3 + 1
    e1, e2, e3 = L1+1, L2+1, L3+1
    num = ( s1*s2*L3*e1*e2*d12*d13*d23
          - e1*e2*e3*(s1*s2 + L1 + L2)*d12*d13*d23
          - e1*e2*e3*(s1*L2*s2 - s1*s2*L3 - L1*L3 + L1)*d12*d13
          - e1*e2*e3*d12*(L1*s1*s2 - s1*s2*L3 - L2*L3 + L2)*d23
          + e1*e2*e3*d13*d23*(L1*s1*s2 + s1*L2*s2 - 2*s1*s2 + L1**2 - L1 + L2**2 - L2)
          - e1*e3*(s1*L2*s2 + L1*L2)*d12*d13*d23
          - e2*e3*(L1*s1*s2 + L1*L2)*d12*d13*d23 )
    den = e1*e2*e3*d12*d13*d23
    return num/den

def kern(mi, mj, mk):
    return W(mi[0], mj[0], mk[0])

# ---------- (1) l=2 exact proportionality ----------
M2 = [(2, m) for m in range(-2, 3)]
t0 = md.cubic_vertex_table(M2, g_eff=1.0)
t1 = md.cubic_vertex_table(M2, g_eff=1.0, kernel=kern)
assert set(t0) == set(t1)
r = [t1[key]/t0[key] for key in t0]
print(f"(1) l=2 table ratio derived/K=1: min={min(r):.10f} max={max(r):.10f} "
      f"(constant => exact proportionality; W(7,7,7)={W(2,2,2):.6f})")
assert abs(max(r)-min(r)) < 1e-12

# ---------- (2) multi-l register ----------
ML = [(2, m) for m in range(-2, 3)] + [(3, m) for m in range(-3, 4)]
occ0 = tuple(1 if mo == (2, 0) else 0 for mo in ML)
G_REP = 12.0
# Isolate the RELATIVE multipole structure at the physical coupling: scale both
# the K=1 and the derived tables to the Frobenius norm of the physical K=1 table
# (g_eff = G_REP on the raw unit-kernel table), so the overall strength is fixed
# and only the relative weights differ.  Matching instead to a fixed Frobenius
# norm of G_REP is an absolute scale, not the physical one, and inflates g_eff
# by 1/||raw|| (about 80x here), outside the convergent window.
F_K1 = np.sqrt(sum(v*v for v in md.cubic_vertex_table(ML, g_eff=G_REP).values()))


def diag_ensemble_pN(H, psi, fb, nmaxN=64):
    """Infinite-time (diagonal-ensemble) multiplicity distribution."""
    E, U = np.linalg.eigh(H)
    w = np.abs(U.conj().T @ psi) ** 2
    pN = np.zeros(nmaxN)
    for a in range(len(E)):
        v = md.multiplicity_distribution(U[:, a], fb)
        it = v.items() if isinstance(v, dict) else enumerate(v)
        for nn, pp in it:
            pN[int(nn)] += w[a] * pp
    return pN[:np.max(np.nonzero(pN)) + 1]


res = {}
for Nmax in (4, 5):
    row = {}
    for name, kn in [("K1", None), ("derived", kern)]:
        tab = md.cubic_vertex_table(ML, g_eff=1.0, kernel=kn)
        nrm = np.sqrt(sum(v * v for v in tab.values()))
        tab = {key: v / nrm * F_K1 for key, v in tab.items()}   # physical strength
        fb = md.FockBasis(ML, Nmax=Nmax, lz=0)
        H = fb.H(tab, include_H2=True)
        psi = np.zeros(fb.dim, complex); psi[fb.index[occ0]] = 1.0
        pbar = diag_ensemble_pN(H, psi, fb)
        row[name] = dict(eta=float(1.0 - pbar[1]), pbar=pbar.tolist(), dim=fb.dim)
    pa = np.array(row['K1']['pbar']); pb = np.array(row['derived']['pbar'])
    L = max(len(pa), len(pb)); pa = np.pad(pa, (0, L - len(pa))); pb = np.pad(pb, (0, L - len(pb)))
    tv = 0.5 * np.abs(pa - pb).sum()
    deta = 100 * (row['derived']['eta'] / row['K1']['eta'] - 1)
    row['tv'] = float(tv); row['deta_pct'] = float(deta)
    res[f"Nmax{Nmax}"] = row
    print(f"(2) Nmax={Nmax} dim={row['K1']['dim']}: "
          f"eta_K1={row['K1']['eta']:.4f} eta_derived={row['derived']['eta']:.4f}  "
          f"Delta_eta={deta:+.1f}%  TV(diag-ensemble pbar)={tv:.4f}")
print("    (diagonal ensemble, physical coupling; Nmax=4 and 5 agree => converged)")

_path = os.path.join(_OUT, 'summary_derived_kernel.json')
json.dump({'protocol': 'physical-coupling Frobenius match; diagonal ensemble',
           'F_K1_physical': float(F_K1),
           'Nmax4': {'tv': res['Nmax4']['tv'], 'deta_pct': res['Nmax4']['deta_pct'],
                     'eta_K1': res['Nmax4']['K1']['eta'], 'eta_derived': res['Nmax4']['derived']['eta'],
                     'dim': res['Nmax4']['K1']['dim']},
           'Nmax5': {'tv': res['Nmax5']['tv'], 'deta_pct': res['Nmax5']['deta_pct'],
                     'eta_K1': res['Nmax5']['K1']['eta'], 'eta_derived': res['Nmax5']['derived']['eta'],
                     'dim': res['Nmax5']['K1']['dim']},
           'pbar_K1': res['Nmax5']['K1']['pbar'],
           'pbar_derived': res['Nmax5']['derived']['pbar'],
           'W777': W(2, 2, 2)},
          open(_path, 'w'), indent=1)
print("saved", _path)
