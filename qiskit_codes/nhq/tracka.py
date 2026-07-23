#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
 nhq.tracka -- the elastic backbone: grid IHO and the Dray-'t Hooft shift
================================================================================
The KNOWN elastic near-horizon physics, used as the internal validation
backbone: the elastic-benchmark calibration appendix of the paper.

  A.1  Inverted harmonic oscillator / dilatation generator per partial wave
       (BGP, SciPost Phys. Core 4, 032 (2021), arXiv:2004.09523; GGV
       arXiv:2012.02357):  H = (Omega/2)(p^2 - u^2)  on a symmetric grid of
       N = 2^nq points, delta_u = delta_p = sqrt(2 pi / N), evolved by the
       second-order split-operator (Strang) step
           U(dt) ~ e^{+i(O/2)u^2 dt/2} e^{-i(O/2)p^2 dt} e^{+i(O/2)u^2 dt/2},
       with the p^2 phase conjugated by the QFT *with the centering fix*:

       Qiskit's QFTGate is the un-centered +2pi transform; the symmetric grid
       needs the centered one.  The verified operator identity is
           p^2 = D Q diag(p_k^2) Q^dag D,   D = Z on qubit 0,
       and, because a Qiskit circuit's unitary is the REVERSE product of
       appended gates, realising D Q (.) Q^dag D appends: D, Q^dag, phase, Q, D.

  A.2  The Dray-'t Hooft gravitational shift per partial wave,
       S_l = exp(i c_l p_in p_out), c_l = 8 pi G_N / (l^2 + l + 1)
       ('t Hooft; GGV arXiv:2012.02357), realised EXACTLY (no Trotter error)
       as a cross-register diagonal phase conjugated by centered QFTs.

  The analytic OTOC check: for the bare Gaussian IHO the Heisenberg
  commutator [u(t), p] = i cosh(Omega t) is a c-number, so
  C(t) = cosh^2(Omega t) exactly -- temperature-independent, and NOT an MSS
  statement.
================================================================================
"""
import numpy as np
import scipy.linalg as sla
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate, DiagonalGate
from qiskit.quantum_info import Statevector, Operator


# ---------------------------------------------------------------- grid setup
def grid(nq):
    """Symmetric grid: N = 2^nq points, u_j = (j - N/2) du, du = dp =
    sqrt(2 pi / N); p_k = (k - N/2) dp."""
    N = 2 ** nq
    du = np.sqrt(2 * np.pi / N)
    j = np.arange(N)
    u = (j - N / 2) * du
    p = (j - N / 2) * du
    return N, du, u, p


def dense_u2_p2(nq):
    """Dense u^2 and p^2 on the centered grid (p^2 via the centered DFT)."""
    N, du, u, p = grid(nq)
    U2 = np.diag(u ** 2).astype(complex)
    # centered transform F_{kj} = N^{-1/2} e^{-i p_k u_j}
    F = np.exp(-1j * np.outer(p, u)) / np.sqrt(N)
    P2 = F.conj().T @ np.diag(p ** 2).astype(complex) @ F
    return U2, P2


def iho_hamiltonian_dense(nq, Omega=1.0):
    U2, P2 = dense_u2_p2(nq)
    return 0.5 * Omega * (P2 - U2)


# ------------------------------------------------------- centered QFT blocks
def _append_centered_p_phase(qc, qubits, phases):
    """Append D Q diag(e^{i phases_k}) Q^dag D on `qubits` (momentum-diagonal
    phase on the centered grid).  Appended order (reverse-product rule):
    D, Q^dag, phase, Q, D."""
    nq = len(qubits)
    qc.z(qubits[0])                              # D = Z on qubit 0
    qc.append(QFTGate(nq).inverse(), qubits)     # Q^dag
    qc.append(DiagonalGate(np.exp(1j * phases)), qubits)
    qc.append(QFTGate(nq), qubits)               # Q
    qc.z(qubits[0])


def iho_strang_step(nq, Omega, dt):
    """One second-order (Strang) split-operator step of e^{-i H dt},
    H = (Omega/2)(p^2 - u^2)."""
    N, du, u, p = grid(nq)
    qc = QuantumCircuit(nq)
    half_u = np.exp(+1j * 0.5 * Omega * u ** 2 * dt / 2)
    qc.append(DiagonalGate(half_u), range(nq))
    _append_centered_p_phase(qc, list(range(nq)), -0.5 * Omega * p ** 2 * dt)
    qc.append(DiagonalGate(half_u), range(nq))
    return qc


def iho_trotter_circuit(nq, Omega, t, r):
    step = iho_strang_step(nq, Omega, t / r)
    qc = QuantumCircuit(nq)
    for _ in range(r):
        qc.compose(step, inplace=True)
    return qc


# ----------------------------------------------------------- 't Hooft shift
def shift_circuit(nq, c_l):
    """Exact circuit for S_l = exp(i c_l p_in (x) p_out) on two nq-qubit
    registers (in = qubits 0..nq-1, out = qubits nq..2nq-1)."""
    N, du, u, p = grid(nq)
    qc = QuantumCircuit(2 * nq)
    for q0 in (0, nq):
        qc.z(q0)
    qc.append(QFTGate(nq).inverse(), range(nq))
    qc.append(QFTGate(nq).inverse(), range(nq, 2 * nq))
    # cross-register diagonal phase e^{i c_l p_kin p_kout}
    phase = np.exp(1j * c_l * np.outer(p, p))    # [k_out, k_in] after kron
    diag = phase.reshape(-1)                     # index = k_out * N + k_in
    qc.append(DiagonalGate(diag), range(2 * nq))
    qc.append(QFTGate(nq), range(nq))
    qc.append(QFTGate(nq), range(nq, 2 * nq))
    for q0 in (0, nq):
        qc.z(q0)
    return qc


def shift_dense(nq, c_l):
    N, du, u, p = grid(nq)
    F = np.exp(-1j * np.outer(p, u)) / np.sqrt(N)
    F2 = np.kron(F, F)                           # [out (x) in], out = high
    phase = np.exp(1j * c_l * np.outer(p, p)).reshape(-1)
    return F2.conj().T @ (phase[:, None] * F2)


# ------------------------------------------------------------------ states
def gaussian_packet(nq, u0=0.0, p0=0.0, sigma=1.0):
    N, du, u, p = grid(nq)
    psi = np.exp(-(u - u0) ** 2 / (4 * sigma ** 2) + 1j * p0 * u)
    return psi / np.linalg.norm(psi)


# ------------------------------------------------------------------ selftest
def selftest(verbose=True):
    ok = True
    nq = 5
    # (a) centered-p^2 circuit identity vs dense
    N, du, u, p = grid(nq)
    qc = QuantumCircuit(nq)
    _append_centered_p_phase(qc, list(range(nq)), -0.7 * p ** 2)
    Uc = Operator(qc).data
    _, P2 = dense_u2_p2(nq)
    Ud = sla.expm(-1j * 0.7 * P2)
    dev = np.abs(Uc - Ud).max()
    ok &= dev < 1e-10
    if verbose:
        print(f"  [centering fix] circuit e^(-i 0.7 p^2) vs dense: {dev:.2e}")
    # (b) Strang step converges as r^-4 in global infidelity
    Omega, t = 1.0, 1.0
    H = iho_hamiltonian_dense(nq, Omega)
    psi0 = gaussian_packet(nq)
    psi_ex = sla.expm(-1j * H * t) @ psi0
    infid = []
    for r in (2, 4, 8, 16):
        psi_c = Statevector(psi0).evolve(iho_trotter_circuit(nq, Omega, t, r)).data
        infid.append(1 - abs(np.vdot(psi_ex, psi_c)) ** 2)
    slope = np.polyfit(np.log([2, 4, 8, 16]), np.log(infid), 1)[0]
    ok &= infid[-1] < 1e-6 and slope < -3.5
    if verbose:
        print(f"  [IHO Strang] infidelity r=2..16: "
              + ", ".join(f"{x:.1e}" for x in infid) + f"; slope {slope:.2f}")
    # (c) exact shift circuit vs dense
    c_l = 8 * np.pi * 0.01 / 7.0
    dev = np.abs(Operator(shift_circuit(4, c_l)).data
                 - shift_dense(4, c_l)).max()
    ok &= dev < 1e-10
    if verbose:
        print(f"  ['t Hooft shift] circuit vs dense: {dev:.2e}")
    # (d) OTOC of the bare IHO = cosh^2(Omega t) on a low-energy packet
    U2, P2 = dense_u2_p2(nq)
    uop = np.diag(u).astype(complex)
    F = np.exp(-1j * np.outer(p, u)) / np.sqrt(N)
    pop = F.conj().T @ np.diag(p).astype(complex) @ F
    psi0 = gaussian_packet(nq, sigma=1 / np.sqrt(2))
    t_ = 0.6
    Ue = sla.expm(-1j * H * t_)
    ut = Ue.conj().T @ uop @ Ue
    comm = ut @ pop - pop @ ut
    C = np.real(np.vdot(psi0, comm.conj().T @ comm @ psi0))
    ok &= abs(C - np.cosh(Omega * t_) ** 2) < 0.02
    if verbose:
        print(f"  [IHO OTOC] C({t_}) = {C:.4f} vs cosh^2 = "
              f"{np.cosh(Omega*t_)**2:.4f}")
    return ok


if __name__ == "__main__":
    print("nhq.tracka selftest")
    assert selftest()
    print("all elastic-benchmark checks passed.")
