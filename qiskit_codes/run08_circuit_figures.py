#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
 run08_circuit_figures.py -- the three circuit figures of the manuscript
================================================================================
Regenerates the circuit diagrams that illustrate the quantum-simulation
pipeline.  Each figure is drawn from the production objects of `nhq`, not
redrawn schematically: the drawn circuit is built by the same constructors the
dynamical runs use, and a unitary-equality assertion is executed for every
figure whose sub-blocks are boxed for readability.

  fig_Q1_trotter_step.pdf  The smallest instance of the derived model: the
                           single mode (2,0) at per-mode cutoff d = 4 in Gray
                           encoding (two qubits, seven Pauli strings), drawn as
                           the X-layer Fock preparation followed by ONE fully
                           decomposed second-order Suzuki-Trotter step at the
                           resource-table step size dt = 0.25.  Reference
                           counts: 7 Pauli terms, depth 25, 10 cx, global phase
                           5.952466393296513 -- asserted at run time.

  fig_Q2_snapshot.pdf      The sequential-segment snapshot protocol of
                           nhq.circuits.trotter_snapshot_states on a two-mode
                           register (4 qubits): X-layer Fock preparation, the
                           transpiled segment U_T(dt) applied repeatedly with
                           the state recorded after each segment, and
                           computational-basis multiplicity readout.

  fig_Q3_otoc_circuit.pdf  The ancilla Hadamard-test interferometer of
                           nhq.circuits.otoc_hadamard_test_circuit on a
                           three-mode register at d = 2 (3 system qubits + 1
                           ancilla).  Controlled time evolution is avoided by
                           conjugation, so only the single-qubit Pauli probes
                           are controlled and the sequence [U_T, cW, U_T^dag,
                           cV] is applied twice.

Boxing convention: for Q2 and Q3 the segment / evolution blocks are wrapped
into labelled composite gates so the diagram stays readable.  Wrapping is
unitary-preserving, and the script asserts that the boxed circuit and the raw
production circuit implement the same operator before the figure is written.

Outputs: fig_Q1_trotter_step.pdf, fig_Q2_snapshot.pdf, fig_Q3_otoc_circuit.pdf,
summary_circuit_figures.json.  Runtime: ~1-2 min (dominated by the Q3 operator
cross-check); NHQ_QUICK=1 skips that cross-check and finishes in seconds.
================================================================================
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

from nhq import model as md
from nhq import circuits as qc
from nhq.util import Timer, save_json, OUT, QUICK

import os

G_REP = 12.0          # representative reduced coupling of the dynamical runs
DT_STEP = 0.25        # Trotter step size of the resource table
DT_SEG = 2.0          # segment length of the snapshot protocol
T_OTOC = 1.0          # evolution time drawn in the interferometer figure


def _boxed(circ, label):
    """Wrap a circuit into a single labelled composite gate on the same number
    of qubits.  The operation is unitary-preserving; it only changes how the
    block is rendered."""
    box = QuantumCircuit(circ.num_qubits)
    box.append(circ.to_gate(label=label), range(circ.num_qubits))
    return box


def _save(fig_circuit, name, fold=-1):
    path = os.path.join(OUT, name)
    fig = fig_circuit.draw(output="mpl", fold=fold, style={"dpi": 300})
    fig.savefig(path, bbox_inches="tight")
    print(f"  wrote {path}")


def figure_Q1(summary):
    """Single mode (2,0), d = 4, Gray: prep + one decomposed Trotter step."""
    modes = [(2, 0)]
    d = 4
    table = md.cubic_vertex_table(modes, g_eff=G_REP)
    Hp, reg = qc.H_pauli(modes, table, d, encoding="gray")
    prep = qc.prep_fock_circuit((1,), reg)          # one graviton in (2,0)
    step = qc.trotter_circuit(Hp, DT_STEP, reps=1, order=2)
    circ = prep.compose(step)

    ops = circ.count_ops()
    summary["Q1"] = dict(modes=[list(m) for m in modes], d=d, g_tilde=G_REP,
                         dt=DT_STEP, reps=1, qubits=reg.n_total,
                         pauli_terms=len(Hp), depth=int(circ.depth()),
                         cx=int(ops.get("cx", 0)),
                         global_phase=float(circ.global_phase),
                         vertex=float(list(table.values())[0]))
    # reference counts quoted in the manuscript
    assert reg.n_total == 2 and len(Hp) == 7, (reg.n_total, len(Hp))
    assert int(circ.depth()) == 25 and int(ops.get("cx", 0)) == 10, ops
    assert abs(float(circ.global_phase) - 5.952466393296513) < 1e-12
    print(f"  Q1: {reg.n_total} qubits, {len(Hp)} Pauli strings, "
          f"depth {circ.depth()}, {ops.get('cx', 0)} cx, "
          f"global phase {float(circ.global_phase):.15f}")
    _save(circ, "fig_Q1_trotter_step.pdf", fold=14)   # wraps onto two rows


def figure_Q2(summary):
    """Two-mode register (4 qubits): the sequential-segment snapshot protocol
    exactly as implemented in nhq.circuits.trotter_snapshot_states."""
    modes = [(2, 1), (2, 0)]
    d, n_seg, reps_seg = 4, 3, 2
    occ0 = (0, 1)                                   # one graviton in (2,0)
    table = md.cubic_vertex_table(modes, g_eff=G_REP)
    Hp, reg = qc.H_pauli(modes, table, d, encoding="gray")

    backend = AerSimulator(method="statevector")
    prep = transpile(qc.prep_fock_circuit(occ0, reg), backend,
                     optimization_level=0)
    seg = transpile(qc.trotter_circuit(Hp, DT_SEG, reps_seg), backend,
                    optimization_level=0)          # the production segment
    seg_box = _boxed(seg, r"$U_T(\Delta t)$")

    # the boxed segment must implement the transpiled production segment
    dev = np.abs(Operator(seg_box).data - Operator(seg).data).max()
    assert dev < 1e-12, dev

    n = reg.n_total
    circ = QuantumCircuit(n, n)
    circ.compose(prep, inplace=True)
    for k in range(n_seg):
        circ.compose(seg_box, inplace=True)
        circ.barrier(label=rf"$|\psi_{k + 1}\rangle$")
    circ.measure(range(n), range(n))

    summary["Q2"] = dict(modes=[list(m) for m in modes], d=d, occ0=list(occ0),
                         qubits=n, pauli_terms=len(Hp), n_segments=n_seg,
                         dt_segment=DT_SEG, reps_per_segment=reps_seg,
                         segment_depth=int(seg.depth()),
                         segment_cx=int(seg.count_ops().get("cx", 0)),
                         boxed_vs_raw_maxdev=float(dev))
    print(f"  Q2: {n} qubits, segment depth {seg.depth()}, "
          f"{seg.count_ops().get('cx', 0)} cx, {n_seg} segments; "
          f"boxed-vs-raw segment deviation {dev:.1e}")
    _save(circ, "fig_Q2_snapshot.pdf", fold=-1)


def figure_Q3(summary):
    """Three-mode register at d = 2 (3 system qubits + 1 ancilla): the OTOC
    Hadamard-test interferometer, built by the production constructor."""
    modes = [(2, 1), (2, -1), (2, 0)]
    d = 2
    occ0 = (0, 0, 1)                                # one graviton in (2,0)
    table = md.cubic_vertex_table(modes, g_eff=G_REP)
    Hp, reg = qc.H_pauli(modes, table, d, encoding="gray")
    prep = qc.prep_fock_circuit(occ0, reg)
    evo = qc.trotter_circuit(Hp, T_OTOC, reps=1)
    i20, i21 = modes.index((2, 0)), modes.index((2, 1))
    Wq, Vq = [i20 * reg.nb], [i21 * reg.nb]         # W = Z probe, V = X probe

    # production circuit, and the same circuit with its blocks boxed
    prod = qc.otoc_hadamard_test_circuit(prep, evo, Wq, Vq,
                                         W_pauli="Z", V_pauli="X")
    fig_circ = qc.otoc_hadamard_test_circuit(
        _boxed(prep, r"prep $|\psi_0\rangle$"), _boxed(evo, r"$U_T(t)$"),
        Wq, Vq, W_pauli="Z", V_pauli="X")

    dev = None
    if not QUICK:
        u_prod = Operator(prod.remove_final_measurements(inplace=False)).data
        u_fig = Operator(fig_circ.remove_final_measurements(inplace=False)).data
        dev = float(np.abs(u_prod - u_fig).max())
        assert dev < 1e-12, dev

    summary["Q3"] = dict(modes=[list(m) for m in modes], d=d, occ0=list(occ0),
                         system_qubits=reg.n_total, ancilla=reg.n_total,
                         pauli_terms=len(Hp), t=T_OTOC,
                         W_probe=dict(pauli="Z", qubit=Wq[0]),
                         V_probe=dict(pauli="X", qubit=Vq[0]),
                         production_depth=int(prod.depth()),
                         boxed_vs_production_maxdev=dev)
    print(f"  Q3: {reg.n_total} system qubits + 1 ancilla, {len(Hp)} Pauli "
          f"strings, production depth {prod.depth()}; boxed-vs-production "
          f"deviation {'skipped in QUICK mode' if dev is None else f'{dev:.1e}'}")
    _save(fig_circ, "fig_Q3_otoc_circuit.pdf", fold=-1)


def main():
    summary = {"note": "circuit diagrams drawn from the production "
                       "constructors of nhq.circuits; boxed blocks are "
                       "verified against the raw production circuits by "
                       "unitary comparison."}
    with Timer("Q1: single-mode Trotter step (2 qubits)"):
        figure_Q1(summary)
    with Timer("Q2: sequential-segment snapshot protocol (4 qubits)"):
        figure_Q2(summary)
    with Timer("Q3: OTOC Hadamard-test interferometer (3 + 1 qubits)"):
        figure_Q3(summary)
    save_json("summary_circuit_figures.json", summary)
    print("\nrun08 complete.")


if __name__ == "__main__":
    main()
