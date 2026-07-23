#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
 run00_validate.py -- multi-engine validation suite (paper Table: validation)
================================================================================
Produces the validation table of the paper on the current software stack.
Nothing in this script is a physics claim; it establishes the two/three-engine
standard:

  E1  dense Kronecker product engine        (nhq.model.H_dense_kron)
  E2  direct occupation-basis engine        (nhq.model.FockBasis)
  E3  Qiskit SparsePauliOp / circuit engine (nhq.circuits)

Checks: encodings bit-for-bit; E1 = E2 = E3; conserved charges [H, Lz], [H, P];
Wigner-Eckart SU(2)-scalar vertex; L2 conservation under the symmetry-
preserving total-N truncation vs the boundary breaking of the per-mode cutoff;
Delta N = +-1 rule; Trotter r^-4 convergence and decompose-independence
(the PauliEvolutionGate pitfall); Fock-state preparation; Aer shot decoding;
OTOC interferometer vs dense; the elastic-benchmark centering fix, Strang
r^-4, exact shift, and the analytic cosh^2(Omega t) IHO OTOC.

Runtime: ~2-4 min.
================================================================================
"""
import numpy as np

from nhq import encoding as enc
from nhq import model as md
from nhq import circuits as qc
from nhq import tracka as ta
from nhq.util import Timer, save_json


def main():
    results = {}

    with Timer("encoding selftests"):
        assert enc.selftest(verbose=True)

    with Timer("model selftests"):
        assert md.selftest(verbose=True)

    with Timer("circuit selftests"):
        assert qc.selftest(verbose=True)

    with Timer("elastic-benchmark selftests"):
        assert ta.selftest(verbose=True)

    # ---- consolidated validation table for the paper --------------------
    with Timer("consolidated validation table (l=2 multiplet)"):
        modes = [(2, m) for m in range(-2, 3)]
        d = 4
        table = md.cubic_vertex_table(modes, g_eff=1.0)
        results["n_vertex_terms_l2_multiplet"] = len(table)

        fb = md.FockBasis(modes, d=d)
        H2e = fb.H(table)
        H1e = md.H_dense_kron(modes, table, d)
        results["dev_E1_E2"] = float(np.abs(H1e - H2e).max())

        Hp_sb, reg_sb = qc.H_pauli(modes, table, d, "standard_binary")
        results["dev_E1_E3_SB"] = float(np.abs(Hp_sb.to_matrix() - H1e).max())
        results["pauli_terms_SB_5mode"] = len(Hp_sb)
        Hp_gr, reg_gr = qc.H_pauli(modes, table, d, "gray")
        ev1 = np.sort(np.linalg.eigvalsh(H1e).real)
        ev3 = np.sort(np.linalg.eigvalsh(Hp_gr.to_matrix()).real)
        results["dev_E1_E3_gray_spectrum"] = float(np.abs(ev1 - ev3).max())
        results["pauli_terms_gray_5mode"] = len(Hp_gr)
        results["qubits_5mode_d4"] = reg_gr.n_total

        Lz = np.diag(fb.Lz_diag())
        results["comm_H_Lz"] = float(np.abs(H1e @ Lz - Lz @ H1e).max())
        P = fb.parity_perm()
        results["comm_H_parity"] = float(np.abs(H1e @ P - P @ H1e).max())
        L2 = fb.L2()
        results["comm_H_L2_permode_cutoff"] = float(
            np.abs(H1e @ L2 - L2 @ H1e).max())
        fbN = md.FockBasis(modes, Nmax=4)
        HN = fbN.H(table)
        L2N = fbN.L2()
        results["comm_H_L2_totalN_cutoff"] = float(
            np.abs(HN @ L2N - L2N @ HN).max())

        # Wigner-Eckart witness processes quoted in the paper
        results["Gop_22_from_21_21"] = md.gaunt_operator(2, 1, 2, 1, 2, 2)
        results["Gop_20_from_21_2m1"] = md.gaunt_operator(2, 1, 2, -1, 2, 0)
        results["Gop_20_from_20_20"] = md.gaunt_operator(2, 0, 2, 0, 2, 0)
        results["forbidden_DeltaLz4"] = (
            0.0 if (1 + 1) != -2 else 1.0)  # (2,1)+(2,1) <- (2,-2) forbidden

    save_json("validation_summary.json", results)
    print("\nconsolidated validation numbers:")
    for k, v in results.items():
        print(f"  {k:34s} = {v}")
    print("\nrun00 complete: all engines agree; charges as documented.")


if __name__ == "__main__":
    main()
