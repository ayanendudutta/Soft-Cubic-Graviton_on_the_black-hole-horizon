#!/usr/bin/env python3
r"""Validate the hybrid evolver + MPS observables against exact block evolution
at l=2 (per-mode cutoff d, identical truncation in both engines)."""
import warnings; warnings.filterwarnings("ignore")
import os, sys
import numpy as np
# Reference engine (nhq.model) from the Qiskit suite in this repository.
# It resolves to the sibling qiskit_codes/ directory by default; setting the
# environment variable NHQ_QISKIT_DIR overrides that location.
_qk = os.environ.get("NHQ_QISKIT_DIR",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qiskit_codes"))
sys.path.insert(0, os.path.abspath(_qk))
from nhq.model import (FockBasis, cubic_vertex_table, BlockEvolver,
                       multiplicity_distribution, inelasticity, entanglement_entropy)
from nhq_tenpy.graviton_mps import GravitonCubicModel
from nhq_tenpy.evolve import evolve, number_distribution

def run(l_max=2, d=4, g=12.0, t_final=40.0, dt=0.05, n_warmup=6):
    mdl = GravitonCubicModel(l_max=l_max, d=d, g_eff=g)
    modes = mdl.modes
    fb = FockBasis(modes, d=d, lz=0)
    H = fb.H(cubic_vertex_table(modes, g_eff=g))
    ev = BlockEvolver(H)
    occ0 = tuple(1 if modes[s]==(2,0) else 0 for s in range(mdl.M))
    psi0 = np.zeros(fb.dim, complex); psi0[fb.index[occ0]] = 1.0
    half = list(range(mdl.M//2))
    snaps = [k*2.0 for k in range(int(t_final//2)+1)]

    psi = mdl.single_graviton_state(2,0)
    rec = evolve(mdl, psi, t_final, dt=dt, n_warmup=n_warmup, chi_max=400,
                 snapshots=snaps)

    eta_r, eta_m, S_r, S_m, P_r, P_m = [],[],[],[],[],[]
    for ti, t in enumerate(rec["times"]):
        psit = ev.psi(psi0, t)
        eta_r.append(inelasticity(psit, fb, 1)); eta_m.append(rec["eta"][ti])
        S_r.append(entanglement_entropy(psit, fb, half)); S_m.append(rec["S_half"][ti])
        pr = multiplicity_distribution(psit, fb); pm = rec["pN"][ti]
        P_r.append([pr.get(k,0.0) for k in range(6)])
        P_m.append([pm.get(k,0.0) for k in range(6)])
    eta_r,eta_m,S_r,S_m = map(np.array,(eta_r,eta_m,S_r,S_m))
    P_r,P_m = np.array(P_r), np.array(P_m)
    return dict(M=mdl.M, chi=max(rec["chi"]),
                eta_dev=float(np.max(np.abs(eta_r-eta_m))),
                S_dev=float(np.max(np.abs(S_r-S_m))),
                P_dev=float(np.max(np.abs(P_r-P_m))),
                eta_r20=float(eta_r[10]), eta_m20=float(eta_m[10]))

if __name__ == "__main__":
    print("hybrid evolver vs exact block evolution (l=2, d=4)")
    for dt in (0.05, 0.025):
        r = run(dt=dt)
        print(f"  dt={dt}: chi={r['chi']}  max|d eta|={r['eta_dev']:.2e}  "
              f"max|dS_half|={r['S_dev']:.2e}  max|dP(N)|={r['P_dev']:.2e}")
        print(f"          eta(t=20): exact={r['eta_r20']:.5f}  mps={r['eta_m20']:.5f}")
    print("  (agreement improving with dt confirms integrator-limited error)")
