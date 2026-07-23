#!/usr/bin/env python3
"""Propagation of the DERIVED on-shell weight W(lam1,lam2,lam3) through the
simulation model.

(1) l=2 multiplet: the table built with the derived kernel is exactly
    proportional to the K=1 table, because on an equal-lambda register the
    weight is a constant that renormalises the coupling; the proportionality
    is asserted at run time.
(2) multi-l register (l=2 + l=3, Nmax-truncated, Lz=0): eta(t) and pbar_N with
    the derived kernel against K=1 at matched Frobenius norm, so that only the
    RELATIVE multipole structure differs.  This measurement replaces the swept
    kernel systematic of the earlier runs.

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
res = {}
for name, kn in [("K1", None), ("derived", kern)]:
    tab = md.cubic_vertex_table(ML, g_eff=1.0, kernel=kn)
    # norm-match: unit Frobenius norm of the magnitude multiset (paper protocol)
    nrm = np.sqrt(sum(v*v for v in tab.values()))
    tab = {key: v/nrm*G_REP for key, v in tab.items()}
    fb = md.FockBasis(ML, Nmax=4, lz=0)
    H = fb.H(tab, include_H2=True)
    psi = np.zeros(fb.dim); psi[fb.index[occ0]] = 1.0
    N0 = 1
    ts = np.linspace(0, 6.0, 25)
    etas = []
    from scipy.sparse.linalg import expm_multiply
    psi_t = expm_multiply(-1j*H, psi, start=0, stop=6.0, num=25, endpoint=True)
    def pvec(row):
        pN = md.multiplicity_distribution(row, fb)
        if isinstance(pN, dict):
            L = max(pN)+1
            v = np.zeros(L)
            for n, p in pN.items(): v[n] = p
            return v
        return np.asarray(pN)
    for row in psi_t:
        v = pvec(row)
        etas.append(1.0 - (v[N0] if N0 < len(v) else 0.0))
    acc = None
    for row in psi_t[-8:]:
        v = pvec(row)
        if acc is None: acc = v.copy()
        else:
            L = max(len(acc), len(v))
            acc = np.pad(acc,(0,L-len(acc))) + np.pad(v,(0,L-len(v)))
    pbar = acc/8
    res[name] = dict(eta_late=float(np.mean(etas[-8:])), pbar=pbar.tolist(),
                     dim=fb.dim)
    print(f"(2) {name:8s}: dim={fb.dim}  eta_late={res[name]['eta_late']:.5f}")

pa = np.array(res['K1']['pbar']); pb = np.array(res['derived']['pbar'])
L = max(len(pa), len(pb)); pa = np.pad(pa,(0,L-len(pa))); pb = np.pad(pb,(0,L-len(pb)))
tv = 0.5*np.abs(pa-pb).sum()
print(f"    TV(pbar_N | derived vs K=1) = {tv:.4f}")
print(f"    eta shift: {res['derived']['eta_late']-res['K1']['eta_late']:+.5f} "
      f"({100*(res['derived']['eta_late']/res['K1']['eta_late']-1):+.2f}%)")
_path = os.path.join(_OUT, 'summary_derived_kernel.json')
json.dump({'tv': tv, 'res': {k: {kk: vv for kk, vv in v.items() if kk!='pbar'}
           for k, v in res.items()},
           'pbar_K1': res['K1']['pbar'], 'pbar_derived': res['derived']['pbar'],
           'W777': W(2,2,2)},
          open(_path,'w'), indent=1)
print("saved", _path)
