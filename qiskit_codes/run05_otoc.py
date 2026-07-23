#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
 run05_otoc.py -- out-of-time-order correlator: bounded, non-scrambling
================================================================================
The OTOC section of the paper.  MSS squared-commutator convention
(Maldacena-Shenker-Stanford, JHEP 08 (2016) 106, arXiv:1503.01409):
C(t) = <[W(t), V]^dag [W(t), V]>, growth exponent = 2x the trajectory rate
(Polchinski, arXiv:1505.08108).

What this script establishes -- and, deliberately, what it does not:
  * dense-engine OTOC between quadratures of different modes on the l = 2
    multiplet: BOUNDED, no clean exponential window at accessible sizes
    -> the reported quantity is max C(t), with NO Lyapunov exponent and
    NO MSS saturation claimed.
  * the hardware-ready measurement protocol: the ancilla Hadamard-test
    interferometer with Pauli probes, cross-validated on Aer (shots) against
    the dense squared commutator -- the two-engine standard for the OTOC.
  * a Lyapunov extraction is performed ONLY on the analytically solvable
    IHO cosh^2 elastic benchmark, where the exponential window exists by
    construction, to validate the fitting pipeline itself.

Outputs: fig_E1_otoc_dense.pdf, fig_E2_otoc_circuit.pdf, summary_otoc.json.
Runtime: full ~10-25 min (Hadamard tests at several times on 11 qubits);
QUICK ~2 min (3-mode register).
================================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

from nhq import model as md
from nhq import circuits as qc
from nhq import tracka as ta
from nhq.util import Timer, save_json, save_npz, save_fig, QUICK, \
    INK, RUST, TEAL, GOLD, GREY

G_UNIT = 1.0
G_REP = 12.0


def main():
    summary = {"claims": [
        "OTOC bounded, no clean exponential window at accessible sizes:",
        "no Lyapunov exponent extracted for H3; no MSS saturation claimed."]}

    modes = [(2, 1), (2, -1), (2, 0)] if QUICK else \
        [(2, m) for m in range(-2, 3)]
    d = 4
    tab = md.cubic_vertex_table(modes, g_eff=G_REP * G_UNIT)

    # ---------------- E1: dense OTOC on the multiplet ---------------------
    with Timer("E1: dense quadrature OTOC (thermal and single-graviton)"):
        fb = md.FockBasis(modes, d=d)
        H = fb.H(tab)
        i20 = modes.index((2, 0))
        i21 = modes.index((2, 1))
        W = fb.quadrature(i20, "u")
        V = fb.quadrature(i21, "p")
        ts = np.linspace(0, 30, 31 if QUICK else 121)
        occ0 = tuple(1 if s == i20 else 0 for s in range(len(modes)))
        psi0 = np.zeros(fb.dim, complex)
        psi0[fb.index[occ0]] = 1.0
        C_state = md.otoc_squared_commutator(H, W, V, ts, state=psi0)
        beta = 2 * np.pi              # kappa = 1 units: beta = 2 pi / kappa
        C_th = md.otoc_squared_commutator(H, W, V, ts, beta=beta)
        summary["E1_maxC_state"] = float(C_state.max())
        summary["E1_maxC_thermal"] = float(C_th.max())
        summary["E1_beta"] = beta

        fig, ax = plt.subplots(figsize=(4.4, 3.0))
        ax.plot(ts, C_state, color=INK,
                label=r"single $(2,0)$ graviton state")
        ax.plot(ts, C_th, color=TEAL, label=rf"thermal, $\beta=2\pi$")
        ax.set_xlabel("$t$")
        ax.set_ylabel(r"$C(t)=\langle|[W(t),V]|^2\rangle$")
        ax.set_title("quadrature OTOC: bounded, non-scrambling\n"
                     "(no Lyapunov fit performed; no MSS claim)")
        ax.legend()
        save_fig(fig, "fig_E1_otoc_dense.pdf")
        save_npz("data_E1_otoc.npz", ts=ts, C_state=C_state, C_th=C_th)

    # ---------------- E1b: fit-pipeline validation on the IHO -------------
    with Timer("E1b: Lyapunov-fit pipeline validated on the IHO cosh^2"):
        Omega = 1.0
        ts_i = np.linspace(0.05, 4.0, 80)
        C_iho = np.cosh(Omega * ts_i) ** 2
        # C - 1 = sinh^2(Omega t) ~ e^{2 Omega t}/4 asymptotically; the local
        # log-slope is 2 Omega coth(Omega t), so fit where coth ~ 1 (t >= 3)
        sl = ts_i >= 3.0
        lam = np.polyfit(ts_i[sl], np.log(C_iho[sl] - 1.0), 1)[0]
        summary["E1b_lambda_fit_on_cosh2"] = float(lam)
        summary["E1b_lambda_expected"] = 2 * Omega
        print(f"  IHO benchmark: fitted growth rate {lam:.4f} "
              f"(expected 2*Omega = {2*Omega}; squared-commutator factor 2)")

    # ---------------- E2: circuit OTOC (Hadamard test, Aer shots) ---------
    with Timer("E2: Hadamard-test OTOC circuit vs dense (Pauli probes)"):
        Hp, reg = qc.H_pauli(modes, tab, d, encoding="gray")
        prep = qc.prep_fock_circuit(occ0, reg)
        # Pauli probes on physical qubits: W = Z on mode i20 qubit 0,
        # V = X on mode i21 qubit 0 (hardware-native probes)
        Wq = [i20 * reg.nb]
        Vq = [i21 * reg.nb]
        n = reg.n_total

        def pauli_dense(p, q):
            out = np.array([[1]], complex)
            mats = {"Z": np.diag([1, -1]).astype(complex),
                    "X": np.array([[0, 1], [1, 0]], complex)}
            for k in range(n - 1, -1, -1):
                out = np.kron(out, mats[p] if k == q else np.eye(2))
            return out

        Wd = pauli_dense("Z", Wq[0])
        Vd = pauli_dense("X", Vq[0])
        Hm = Hp.to_matrix()
        psi_enc = np.zeros(2 ** n, complex)
        psi_enc[reg.basis_index(occ0)] = 1.0
        ts_c = np.linspace(0.5, 8.0, 5 if QUICK else 10)
        C_dense, C_circ, C_err = [], [], []
        shots = 4096 if QUICK else 16384
        for t in ts_c:
            C_dense.append(md.otoc_squared_commutator(
                Hm, Wd, Vd, [t], state=psi_enc)[0])
            reps = max(2, int(np.ceil((3 if QUICK else 8) * t)))
            evo = qc.trotter_circuit(Hp, float(t), reps)
            circ = qc.otoc_hadamard_test_circuit(prep, evo, Wq, Vq,
                                                 W_pauli="Z", V_pauli="X")
            ReF, sig = qc.run_otoc_test(circ, shots=shots, seed=17)
            C_circ.append(2 - 2 * ReF)
            C_err.append(2 * 2 * sig)     # 2 sigma on C = 2 - 2 ReF
        summary["E2_max_absdev"] = float(np.max(np.abs(
            np.array(C_dense) - np.array(C_circ))))
        summary["E2_shots"] = shots

        fig, ax = plt.subplots(figsize=(4.4, 3.0))
        ax.plot(ts_c, C_dense, "-", color=INK, label="dense engine")
        ax.errorbar(ts_c, C_circ, yerr=C_err, fmt="o", ms=4, color=RUST,
                    capsize=3, label=rf"Hadamard test, {shots} shots "
                    r"($\pm2\sigma$)")
        ax.set_xlabel("$t$")
        ax.set_ylabel("$C(t)$ (Pauli probes)")
        ax.set_title("circuit OTOC protocol vs dense (two-engine standard)")
        ax.legend()
        save_fig(fig, "fig_E2_otoc_circuit.pdf")
        save_npz("data_E2_otoc_circuit.npz", ts=ts_c,
                 C_dense=np.array(C_dense), C_circ=np.array(C_circ),
                 C_err=np.array(C_err))

    save_json("summary_otoc.json", summary)
    print("\nrun05 complete.")


if __name__ == "__main__":
    main()
