#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
 run07_noise_resources.py -- hardware resources and the depolarising-noise
 hazard (Aer density matrix)
================================================================================
The paper's hardware-feasibility section, written so the manuscript can state
"hardware runs are deferred" with quantitative honesty:

 (1) Resource table: transpiled cx count / depth per second-order Trotter step
     to the basis {cx, rz, sx, x} at optimisation level 1, for the 3-mode and
     5-mode l = 2 registers (Gray, d = 4) -- the transpiled-resource table of
     the paper.

 (2) Depolarising-noise study on the 3-mode register (6 qubits, exact density
     matrix, NO shot noise): state fidelity F(p2) AND measured inelasticity
     eta(p2) tracked JOINTLY versus the two-qubit error rate p2, with the
     spurious mixed-state value eta_mix = 1 - D_{N0}/2^n marked.  Depolarising
     noise drives rho toward the maximally mixed state, whose graviton number
     is not N0 -- reporting eta without fidelity would mistake decoherence for
     particle production.  The reported budget is the p2 at which the eta
     bias exceeds 10% of the signal.

Outputs: resource_table.json / .txt, fig_G1_noise.pdf, summary_noise.json.
Runtime: full ~20-60 min (density-matrix evolution of deep transpiled
circuits); QUICK ~3 min (fewer noise points, smaller r).
================================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from qiskit import transpile

from nhq import model as md
from nhq import circuits as qc
from nhq.util import Timer, save_json, save_fig, QUICK, \
    INK, RUST, TEAL, GOLD, GREY

G_UNIT = 1.0
G_REP = 12.0


def main():
    summary = {}

    # ---------------- G0: resource table ----------------------------------
    with Timer("G0: transpiled resource table (dt = 0.25 step, r = 1)"):
        rows = []
        registers = [
            ("l=2 subset (3 modes)", [(2, 1), (2, -1), (2, 0)], 4),
            ("l=2 multiplet (5 modes)", [(2, m) for m in range(-2, 3)], 4),
        ]
        for name, modes, d in registers:
            tab = md.cubic_vertex_table(modes, g_eff=G_REP * G_UNIT)
            Hp, reg = qc.H_pauli(modes, tab, d, encoding="gray")
            step = qc.trotter_circuit(Hp, 0.25, 1)
            res = qc.resource_counts(step)
            res.update(name=name, pauli_terms=len(Hp))
            rows.append(res)
            print(f"  {name}: {res['n_qubits']} qubits, "
                  f"{res['pauli_terms']} Pauli terms, "
                  f"{res['cx']} cx/step, depth {res['depth']}")
        summary["resource_table"] = rows
        import os
        from nhq.util import OUT
        with open(os.path.join(OUT, "resource_table.txt"), "w") as f:
            f.write("register | qubits | Pauli terms | cx/step | depth/step\n")
            for r in rows:
                f.write(f"{r['name']} | {r['n_qubits']} | {r['pauli_terms']}"
                        f" | {r['cx']} | {r['depth']}\n")

    # ---------------- G1: depolarising-noise study ------------------------
    with Timer("G1: depolarising noise, fidelity + eta jointly (6 qubits)"):
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error
        from qiskit.quantum_info import DensityMatrix, Statevector

        modes = [(2, 1), (2, -1), (2, 0)]
        d = 4
        occ0 = (0, 0, 1)
        N0 = 1
        t_phys = 2.0
        reps = 4 if QUICK else 8
        tab = md.cubic_vertex_table(modes, g_eff=G_REP * G_UNIT)
        Hp, reg = qc.H_pauli(modes, tab, d, encoding="gray")
        prep = qc.prep_fock_circuit(occ0, reg)
        evo = qc.trotter_circuit(Hp, t_phys, reps)
        circ = prep.compose(evo)
        n = reg.n_total

        # ideal reference state (noiseless statevector of the SAME circuit)
        psi_ideal = Statevector(circ)

        # graviton-number projectors in the ENCODED basis, for eta from rho
        fb_full = md.FockBasis(modes, d=d)
        Ns = fb_full.N_diag().astype(int)
        proj_N0 = np.zeros(2 ** n)
        for p, o in enumerate(fb_full.occs):
            if Ns[p] == N0:
                proj_N0[reg.basis_index(o)] = 1.0
        D_N0 = int(proj_N0.sum())
        eta_mix = 1.0 - D_N0 / 2 ** n
        summary["G1_eta_mix"] = eta_mix
        summary["G1_reps"] = reps

        # exact eta of the ideal circuit state
        pd_ideal = np.abs(psi_ideal.data) ** 2
        eta_ideal = 1.0 - float(np.dot(pd_ideal, proj_N0))
        summary["G1_eta_ideal"] = eta_ideal

        p2s = np.array([0, 1e-4, 3e-4] if QUICK else
                       [0, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3])
        fids, etas, cxs = [], [], None
        for p2 in p2s:
            backend_kwargs = dict(method="density_matrix")
            if p2 > 0:
                nm = NoiseModel()
                nm.add_all_qubit_quantum_error(
                    depolarizing_error(p2 / 10, 1), ["sx", "x"])
                nm.add_all_qubit_quantum_error(
                    depolarizing_error(p2, 2), ["cx"])
                backend_kwargs["noise_model"] = nm
            backend = AerSimulator(**backend_kwargs)
            tqc = transpile(circ, backend,
                            basis_gates=["cx", "rz", "sx", "x"],
                            optimization_level=1)
            if cxs is None:
                cxs = int(tqc.count_ops().get("cx", 0))
                summary["G1_cx_total"] = cxs
                print(f"  circuit: {cxs} cx total "
                      f"(t={t_phys}, r={reps}); eta_ideal={eta_ideal:.4f}, "
                      f"eta_mix={eta_mix:.4f}")
            tqc.save_density_matrix()
            rho = backend.run(tqc).result().data()["density_matrix"]
            rho = DensityMatrix(rho)
            F = float(np.real(psi_ideal.expectation_value(rho)))  # <psi|rho|psi>
            diag = np.real(np.diag(rho.data))
            eta = 1.0 - float(np.dot(diag, proj_N0))
            fids.append(F)
            etas.append(eta)
            print(f"    p2={p2:.0e}: F={F:.4f}, eta={eta:.4f}")
        fids, etas = np.array(fids), np.array(etas)
        # per-gate budget: p2 at which |eta - eta_ideal| > 0.1 * eta_ideal
        bias = np.abs(etas - eta_ideal)
        bad = np.where(bias > 0.1 * max(eta_ideal, 1e-9))[0]
        p2_budget = float(p2s[bad[0]]) if len(bad) else float(p2s[-1])
        summary["G1_p2_budget_10pct"] = p2_budget
        summary["G1_table"] = [dict(p2=float(a), F=float(b), eta=float(c))
                               for a, b, c in zip(p2s, fids, etas)]

        fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9))
        axes[0].semilogx(np.maximum(p2s, 1e-5), fids, "o-", color=INK)
        axes[0].set_xlabel("two-qubit depolarising error $p_2$")
        axes[0].set_ylabel(r"fidelity $\langle\psi_{\rm id}|\rho|"
                           r"\psi_{\rm id}\rangle$")
        axes[0].set_title(f"{cxs} cx total (t={t_phys:g}, r={reps})")
        axes[1].semilogx(np.maximum(p2s, 1e-5), etas, "o-", color=RUST,
                         label=r"$\eta(p_2)$")
        axes[1].axhline(eta_ideal, color=GREY, ls=":",
                        label=r"$\eta_{\rm ideal}$")
        axes[1].axhline(eta_mix, color=GOLD, ls="--",
                        label=rf"$\eta_{{\rm mix}}={eta_mix:.3f}$ (spurious)")
        axes[1].set_xlabel("two-qubit depolarising error $p_2$")
        axes[1].set_ylabel(r"measured inelasticity $\eta$")
        axes[1].set_title("noise manufactures $\\eta$: report with fidelity")
        axes[1].legend(fontsize=7)
        save_fig(fig, "fig_G1_noise.pdf")

    save_json("summary_noise.json", summary)
    print("\nrun07 complete.  The 'hardware deferred' statement of the paper "
          "rests on G1_p2_budget_10pct and the resource table.")


if __name__ == "__main__":
    main()
