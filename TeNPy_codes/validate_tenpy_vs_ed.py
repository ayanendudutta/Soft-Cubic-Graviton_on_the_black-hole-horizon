#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Two-engine validation of the TeNPy MPO build against the paper's reference
occupation-basis engine (nhq.model.FockBasis / H_dense_kron).

The per-mode occupation cutoff d is the SAME truncation in both engines, so at
the l = 2 register (small enough to diagonalize densely) the two Hamiltonians
must agree to machine precision -- eigenvalue-for-eigenvalue in the Lz = 0
sector -- and the real-time dynamics of the single-(2,0) graviton quench must
agree between exact block evolution (reference) and near-exact TDVP (TeNPy).

This is the engine-vs-engine discipline of the paper carried over to the
tensor-network method.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
import numpy as np

# Reference engine (nhq.model) from the Qiskit suite in this repository.
# It resolves to the sibling qiskit_codes/ directory by default; setting the
# environment variable NHQ_QISKIT_DIR overrides that location.
_qk = os.environ.get("NHQ_QISKIT_DIR",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "qiskit_codes"))
sys.path.insert(0, os.path.abspath(_qk))
from nhq.model import (FockBasis, cubic_vertex_table, BlockEvolver,
                       multiplicity_distribution, inelasticity,
                       entanglement_entropy)
from nhq_tenpy.graviton_mps import GravitonCubicModel, build_modes

from tenpy.algorithms.exact_diag import ExactDiag


def spectrum_check(l_max, d, g_eff=12.0):
    """Full Lz=0 spectrum: TeNPy ExactDiag vs FockBasis dense H."""
    mdl = GravitonCubicModel(l_max=l_max, d=d, g_eff=g_eff)
    modes_ord = mdl.modes                      # site-ordered modes

    # reference: FockBasis in the same site order, Lz = 0
    fb = FockBasis(modes_ord, d=d, lz=0)
    Href = fb.H(cubic_vertex_table(modes_ord, g_eff=g_eff))
    Eref = np.linalg.eigvalsh(Href)

    # TeNPy: exact diag of the MPO restricted to the Lz = 0 charge sector.
    # max_size guards on (d^M)^2 (full product space); raise it for the small
    # validation registers where the full-space pipe is affordable.
    ed = ExactDiag(mdl, charge_sector=[0], max_size=1e9)
    ed.build_full_H_from_mpo()
    ed.full_diagonalization()
    Emps = np.sort(ed.E.real)

    n = min(len(Eref), len(Emps))
    max_dev = np.max(np.abs(np.sort(Eref)[:n] - Emps[:n]))
    return dict(M=mdl.M, dim_ref=fb.dim, dim_mps=len(Emps),
                n_vertices=mdl.n_vertices, H_bond_dim=mdl.H_bond_dim,
                E0_ref=float(np.min(Eref)), E0_mps=float(np.min(Emps)),
                max_eig_dev=float(max_dev))


def dynamics_check(l_max, d, g_eff=12.0, t_final=20.0, n_snap=11, chi_max=200):
    """Single-(2,0) graviton quench: exact block evolution vs TeNPy 2-site TDVP.
    Compares P(N,t), inelasticity, and half-cut entanglement."""
    from tenpy.algorithms.tdvp import TwoSiteTDVPEngine

    mdl = GravitonCubicModel(l_max=l_max, d=d, g_eff=g_eff)
    modes_ord = mdl.modes

    # reference exact evolution in the Lz=0 sector
    fb = FockBasis(modes_ord, d=d, lz=0)
    Href = fb.H(cubic_vertex_table(modes_ord, g_eff=g_eff))
    ev = BlockEvolver(Href)
    # initial: one graviton in (2,0)
    occ0 = tuple(1 if modes_ord[s] == (2, 0) else 0 for s in range(mdl.M))
    psi0 = np.zeros(fb.dim, complex); psi0[fb.index[occ0]] = 1.0
    # index sets for half-cut entanglement (first half of sites)
    half = list(range(mdl.M // 2))

    ts = np.linspace(0, t_final, n_snap)
    ref = {"eta": [], "P": [], "S_half": []}
    for t in ts:
        psit = ev.psi(psi0, t)
        ref["eta"].append(inelasticity(psit, fb, 1))
        pN = multiplicity_distribution(psit, fb)
        ref["P"].append([pN.get(k, 0.0) for k in range(6)])
        ref["S_half"].append(entanglement_entropy(psit, fb, half))

    # TeNPy TDVP -- small fixed step, snapshots every (ts[1]-ts[0]).
    psi = mdl.single_graviton_state(2, 0)
    dt_snap = ts[1] - ts[0]
    dt_small = 0.05
    n_sub = int(round(dt_snap / dt_small))
    opts = {"trunc_params": {"chi_max": chi_max, "svd_min": 1e-12},
            "dt": dt_small, "N_steps": n_sub,
            "max_dt": 10.0}                       # omega ~ 2.6, so dt=0.05 is fine
    eng = TwoSiteTDVPEngine(psi, mdl, opts)
    mps = {"eta": [], "P": [], "S_half": [], "chi": []}

    def measure(psi):
        pN = number_distribution_from_mps(psi, mdl.d)
        mps["eta"].append(1.0 - pN.get(1, 0.0))
        mps["P"].append([pN.get(k, 0.0) for k in range(6)])
        S = psi.entanglement_entropy(bonds=[mdl.M // 2])
        mps["S_half"].append(float(S[0]))
        mps["chi"].append(int(max(psi.chi)))

    measure(psi)
    for _ in range(n_snap - 1):
        eng.run()                                 # advances n_sub * dt_small = dt_snap
        measure(psi)

    eta_dev = np.max(np.abs(np.array(ref["eta"]) - np.array(mps["eta"])))
    P_dev = np.max(np.abs(np.array(ref["P"]) - np.array(mps["P"])))
    S_dev = np.max(np.abs(np.array(ref["S_half"]) - np.array(mps["S_half"])))
    return dict(M=mdl.M, eta_max_dev=float(eta_dev), P_max_dev=float(P_dev),
                S_half_max_dev=float(S_dev),
                eta_final_ref=float(ref["eta"][-1]),
                eta_final_mps=float(mps["eta"][-1]),
                chi_max_reached=int(max(mps["chi"])))


def number_distribution_from_mps(psi, d):
    """Total-quanta distribution P(N) from an MPS by exact contraction of the
    projector onto each total-N sector.  For the small validation registers
    the full statevector is read directly (cheap); returns dict {N: prob}."""
    # For validation registers (l=2, few sites) the full vector is tiny.
    vec = psi.get_theta(0, psi.L).to_ndarray().reshape(-1)  # dense statevector
    # site order: TeNPy site 0 is the FIRST tensor; occupation index within site
    L = psi.L
    probs = np.abs(vec) ** 2
    # enumerate occupations in the same index convention as get_theta
    # get_theta(0,L) has legs (vL, p0, p1, ..., p_{L-1}, vR) with vL,vR trivial
    from itertools import product
    pN = {}
    for idx, occ in enumerate(product(range(d), repeat=L)):
        N = sum(occ)
        pN[N] = pN.get(N, 0.0) + probs[idx]
    return pN


if __name__ == "__main__":
    print("=" * 72)
    print("TeNPy MPO  vs  reference occupation-basis engine")
    print("=" * 72)
    print("\n[1] Full Lz=0 spectrum (machine-precision engine check)")
    for (lmax, d) in [(2, 4), (2, 6)]:
        r = spectrum_check(lmax, d)
        print(f"  l_max={lmax}, d={d}: M={r['M']}, dim={r['dim_ref']}, "
              f"{r['n_vertices']} vertices, H-MPO chi={r['H_bond_dim']}")
        print(f"      E0(ref)={r['E0_ref']:.10f}  E0(mps)={r['E0_mps']:.10f}")
        print(f"      max |E_ref - E_mps| over {r['dim_mps']} levels = "
              f"{r['max_eig_dev']:.3e}")

    print("\n[2] Single-(2,0) quench dynamics: exact evolution vs 2-site TDVP")
    for (lmax, d) in [(2, 4), (2, 6)]:
        r = dynamics_check(lmax, d, chi_max=200)
        print(f"  l_max={lmax}, d={d}: M={r['M']}, chi reached={r['chi_max_reached']}")
        print(f"      max|dP(N,t)| = {r['P_max_dev']:.3e}   "
              f"max|d eta| = {r['eta_max_dev']:.3e}   "
              f"max|dS_half| = {r['S_half_max_dev']:.3e}")
        print(f"      eta(t=20): ref={r['eta_final_ref']:.6f}  "
              f"mps={r['eta_final_mps']:.6f}")
    print("\nvalidation complete.")
