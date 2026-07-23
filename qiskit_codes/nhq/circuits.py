#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
 nhq.circuits -- Qiskit engine: Pauli Hamiltonians, Trotter circuits, Aer runs
================================================================================
The quantum engine of the two-engine standard: every object built here is
cross-validated against the dense/occupation-basis engines of nhq.model.

Critical Qiskit conventions honoured throughout:
  * PauliEvolutionGate applied UN-decomposed silently uses the exact matrix
    exponential in statevector simulation, masking all Trotter error.  Every
    circuit here calls .decompose(reps=...) so the Trotter error is genuine
    and set solely by the Suzuki repetitions r (verified in selftest()).
  * Qiskit little-endian: mode i on qubits [i*nb, (i+1)*nb), matching the
    Kronecker convention (mode 0 least significant).
  * Fock product states are computational-basis states in SB/Gray/unary, so
    state preparation is a layer of X gates (no `initialize` needed).
================================================================================
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter

from . import encoding as enc
from .model import omega_l


# ============================================================================
# 1. Hamiltonians as SparsePauliOp
# ============================================================================
def H_pauli(modes, table, d, encoding="gray", R=1.0):
    """H2^grav + H3 as a SparsePauliOp on the encoded register."""
    reg = enc.BosonRegister(len(modes), d, encoding=encoding)
    terms = []
    for i, (l, _) in enumerate(modes):
        terms.append(omega_l(l, R) * reg.num(i))
    for (i, j, k), V in table.items():
        T = reg.adag(i) @ reg.adag(j) @ reg.a(k)
        terms.append((V * (T + T.adjoint())).simplify())
    H = terms[0]
    for t in terms[1:]:
        H = (H + t).simplify()
    return H.simplify(), reg


def H_pauli_mixed(g_modes, s_modes, g_table, m_table, d, encoding="gray",
                  R=1.0):
    """Full H2 + H3 + H_int on a mixed graviton+scalar register (gravitons
    first, then scalars; m_table keys (g, i, j) with local scalar indices)."""
    modes = list(g_modes) + list(s_modes)
    reg = enc.BosonRegister(len(modes), d, encoding=encoding)
    n_g = len(g_modes)
    terms = []
    for i, (l, _) in enumerate(modes):
        terms.append(omega_l(l, R) * reg.num(i))
    for (i, j, k), V in g_table.items():
        T = reg.adag(i) @ reg.adag(j) @ reg.a(k)
        terms.append((V * (T + T.adjoint())).simplify())
    for (g, i, j), C in m_table.items():
        T = reg.adag(g) @ reg.a(n_g + i) @ reg.a(n_g + j)
        terms.append((C * (T + T.adjoint())).simplify())
    H = terms[0]
    for t in terms[1:]:
        H = (H + t).simplify()
    return H.simplify(), reg


# ============================================================================
# 2. circuits
# ============================================================================
def prep_fock_circuit(occ, reg):
    """State-preparation circuit for the product Fock state |occ>: X gates on
    the set bits of each mode's code word (exact, depth 1)."""
    qc = QuantumCircuit(reg.n_total)
    for i, n in enumerate(occ):
        bits = enc.code_bits(n, reg.d, reg.encoding)
        for k, b in enumerate(bits):
            if b:
                qc.x(i * reg.nb + k)
    return qc


def trotter_circuit(H, t, reps, order=2, decompose_reps=2):
    """Second-order Suzuki-Trotter circuit for exp(-i H t), DECOMPOSED to
    elementary Pauli rotations.  `reps` sets the (genuine) Trotter error;
    `decompose_reps` only sets gate granularity and does not change the
    unitary (verified in selftest)."""
    evo = PauliEvolutionGate(H, time=t,
                             synthesis=SuzukiTrotter(order=order, reps=reps))
    qc = QuantumCircuit(H.num_qubits)
    qc.append(evo, range(H.num_qubits))
    return qc.decompose(reps=decompose_reps)


def evolve_statevector(prep, evo):
    """Exact statevector of prep;evo (little-endian numpy array)."""
    return Statevector(prep.compose(evo)).data


def exact_unitary_reference(H, t):
    """Dense e^{-iHt} of the SparsePauliOp (the exact reference in the SAME
    encoded basis; small registers only)."""
    import scipy.linalg as sla
    return sla.expm(-1j * H.to_matrix() * t)


def trotter_snapshot_states(H, prep, dt, n_segments, reps_per_segment,
                            order=2, seed=11):
    """Sequential Suzuki-Trotter evolution on AerSimulator(statevector):
    the segment circuit U_T(dt) (reps_per_segment Suzuki repetitions,
    decomposed and transpiled ONCE) is applied n_segments times, feeding the
    saved statevector of segment k back in as the initial state of segment
    k+1 via Aer's set_statevector.  Returns the statevectors at times
    dt, 2 dt, ..., n_segments*dt (little-endian numpy arrays).

    This is exactly the Trotterised dynamics -- per-segment error set by
    reps_per_segment, global error accumulating linearly as on hardware --
    with memory bounded by ONE segment circuit."""
    import qiskit_aer  # noqa: F401  (adds set_/save_statevector methods)
    from qiskit_aer import AerSimulator
    backend = AerSimulator(method="statevector", seed_simulator=seed)
    n = H.num_qubits
    seg = trotter_circuit(H, dt, reps_per_segment, order=order)
    tseg = transpile(seg, backend, optimization_level=0)
    tprep = transpile(prep, backend, optimization_level=0)
    states = []
    psi = None
    for k in range(n_segments):
        qck = QuantumCircuit(n)
        if k == 0:
            qck.compose(tprep, inplace=True)
        else:
            qck.set_statevector(psi)
        qck.compose(tseg, inplace=True)
        qck.save_statevector()
        psi = backend.run(qck).result().get_statevector()
        states.append(np.asarray(psi))
    return states


# ---------------------------- measurement decoding --------------------------
def counts_to_occupations(counts, reg):
    """Decode Aer counts into Fock occupation tuples; returns
    (dict occ -> counts, n_valid, n_leaked)."""
    occs, n_valid, n_leak = {}, 0, 0
    for bitstring, c in counts.items():
        occ = reg.decode_bitstring(bitstring.replace(" ", ""))
        if occ is None:
            n_leak += c
            continue
        occs[occ] = occs.get(occ, 0) + c
        n_valid += c
    return occs, n_valid, n_leak


def pN_from_counts(counts, reg):
    """Shot-estimated multiplicity distribution with binomial standard errors:
    returns dict N -> (p, sigma_p)."""
    occs, n_valid, _ = counts_to_occupations(counts, reg)
    tot = {}
    for occ, c in occs.items():
        N = sum(occ)
        tot[N] = tot.get(N, 0) + c
    out = {}
    for N, c in tot.items():
        p = c / n_valid
        out[N] = (p, np.sqrt(max(p * (1 - p), 1e-12) / n_valid))
    return out


# ---------------------------- OTOC Hadamard test ----------------------------
def otoc_hadamard_test_circuit(prep, U_evo, W_qubits, V_qubits,
                               W_pauli="Z", V_pauli="Z"):
    """Real-part Hadamard-test interferometer for
        F(t) = <psi| W V(t) W V(t) |psi>,  V(t) = U^dag V U,
    with Pauli (Hermitian-unitary) probes; then C(t) = 2 - 2 Re F.
    Controlled-(U^dag V U) is realised as U ; c-V ; U^dag (conjugation), so
    only the Pauli probes are controlled.  Appended gate sequence
        [U, cV, U^dag, cW] x 2, final H, measure <Z_anc> = Re F.
    prep acts on the system register; the ancilla is the LAST qubit."""
    n = U_evo.num_qubits
    qc = QuantumCircuit(n + 1, 1)
    anc = n
    qc.compose(prep, qubits=range(n), inplace=True)
    qc.h(anc)
    Udag = U_evo.inverse()

    def c_pauli(paulis, qubits):
        for p, q in zip(paulis, qubits):
            if p == "Z":
                qc.cz(anc, q)
            elif p == "X":
                qc.cx(anc, q)
            elif p == "Y":
                qc.cy(anc, q)
            else:
                raise ValueError(p)

    for _ in range(2):
        qc.compose(U_evo, qubits=range(n), inplace=True)
        c_pauli(V_pauli, V_qubits)
        qc.compose(Udag, qubits=range(n), inplace=True)
        c_pauli(W_pauli, W_qubits)
    qc.h(anc)
    qc.measure(anc, 0)
    return qc


def run_otoc_test(qc, shots=4096, seed=11, noise_model=None):
    """Run the Hadamard-test circuit on Aer; returns (ReF, sigma)."""
    from qiskit_aer import AerSimulator
    backend = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    tqc = transpile(qc, backend, optimization_level=1)
    res = backend.run(tqc, shots=shots).result().get_counts()
    n0 = res.get("0", 0)
    n1 = res.get("1", 0)
    ReF = (n0 - n1) / shots
    sig = np.sqrt(max(1 - ReF ** 2, 1e-12) / shots)
    return ReF, sig


# ---------------------------- Aer sampling ----------------------------------
def sample_counts(prep, evo, shots=8192, seed=11, noise_model=None,
                  method=None):
    """Measure all qubits after prep;evo on AerSimulator."""
    from qiskit_aer import AerSimulator
    kw = dict(seed_simulator=seed)
    if noise_model is not None:
        kw["noise_model"] = noise_model
    if method is not None:
        kw["method"] = method
    backend = AerSimulator(**kw)
    qc = prep.compose(evo)
    qc.measure_all()
    tqc = transpile(qc, backend, optimization_level=0)
    return backend.run(tqc, shots=shots).result().get_counts()


def resource_counts(qc, basis_gates=("cx", "rz", "sx", "x"),
                    optimization_level=1):
    """Transpiled resource estimate: (n_qubits, cx count, depth, total ops)."""
    tqc = transpile(qc, basis_gates=list(basis_gates),
                    optimization_level=optimization_level)
    ops = tqc.count_ops()
    return dict(n_qubits=tqc.num_qubits, cx=int(ops.get("cx", 0)),
                depth=int(tqc.depth()), total=int(sum(ops.values())))


# ============================================================================
# selftest
# ============================================================================
def selftest(verbose=True):
    import scipy.linalg as sla
    from .model import cubic_vertex_table, FockBasis, multiplicity_distribution
    ok = True
    modes = [(2, 1), (2, -1), (2, 0)]
    d, g = 4, 0.6
    table = cubic_vertex_table(modes, g_eff=g)

    # (a) two-engine Hamiltonian agreement, SB bit-for-bit
    from .model import H_dense_kron
    Hp_sb, reg_sb = H_pauli(modes, table, d, encoding="standard_binary")
    Hd = H_dense_kron(modes, table, d)
    dev = np.abs(Hp_sb.to_matrix() - Hd).max()
    ok &= dev < 1e-12
    if verbose:
        print(f"  [two-engine SB] ||H_pauli - H_kron|| = {dev:.2e} "
              f"({reg_sb.n_total} qubits, {len(Hp_sb)} Pauli terms)")

    # (b) Gray spectrum agreement
    Hp, reg = H_pauli(modes, table, d, encoding="gray")
    evd = np.sort(np.linalg.eigvalsh(Hd).real)
    evp = np.sort(np.linalg.eigvalsh(Hp.to_matrix()).real)
    dev = np.abs(evd - evp).max()
    ok &= dev < 1e-11
    if verbose:
        print(f"  [two-engine Gray] spectrum dev = {dev:.2e}")

    # (c) prep circuit lands on the right computational basis state
    occ0 = (0, 0, 1)
    prep = prep_fock_circuit(occ0, reg)
    psi = Statevector(prep).data
    idx = reg.basis_index(occ0)
    ok &= abs(abs(psi[idx]) - 1) < 1e-12
    if verbose:
        print(f"  [prep] |{occ0}> -> basis index {idx}: pass")

    # (d) Trotter convergence is genuine (r^-4) and decompose-independent
    t = 1.0
    Uex = exact_unitary_reference(Hp, t)
    psi_ex = Uex @ psi
    infids = []
    for reps in (1, 2, 4, 8):
        qc = trotter_circuit(Hp, t, reps)
        psi_c = evolve_statevector(prep, qc)
        infids.append(1 - abs(np.vdot(psi_ex, psi_c)) ** 2)
    slopes = np.diff(np.log(infids)) / np.log(2)
    ok &= infids[-1] < 1e-4 and slopes[-1] < -3.4
    qc1 = trotter_circuit(Hp, t, 4, decompose_reps=1)
    qc3 = trotter_circuit(Hp, t, 4, decompose_reps=3)
    dev = np.abs(evolve_statevector(prep, qc1)
                 - evolve_statevector(prep, qc3)).max()
    ok &= dev < 1e-10
    if verbose:
        print(f"  [Trotter] infidelity r=1..8: "
              + ", ".join(f"{x:.1e}" for x in infids)
              + f" (last slope {slopes[-1]:.2f}); decompose-independence "
              f"{dev:.1e}")

    # (e) Aer shot decoding reproduces the exact p_N
    fb = FockBasis(modes, d=d)
    Hb = fb.H(table)
    psi0 = np.zeros(fb.dim, complex)
    psi0[fb.index[occ0]] = 1.0
    psi_t = sla.expm(-1j * Hb * 2.0) @ psi0
    pN_exact = multiplicity_distribution(psi_t, fb)
    qc = trotter_circuit(Hp, 2.0, 16)
    counts = sample_counts(prep, qc, shots=20000, seed=5)
    pN_shots = pN_from_counts(counts, reg)
    dev = max(abs(pN_exact.get(N, 0) - pN_shots.get(N, (0, 0))[0])
              for N in set(pN_exact) | set(pN_shots))
    ok &= dev < 0.02
    if verbose:
        print(f"  [Aer shots] p_N shot-estimate vs exact: max dev = {dev:.3f} "
              f"(20k shots)")

    # (f) OTOC Hadamard test vs dense squared commutator
    from .model import otoc_squared_commutator
    # Pauli probes: W = Z on qubit 0, V = Z on qubit 2 (dense references)
    def z_on(q, n):
        out = np.array([[1]], complex)
        for k in range(n - 1, -1, -1):
            out = np.kron(out, np.diag([1, -1]) if k == q else np.eye(2))
        return out
    n = reg.n_total
    Wd, Vd = z_on(0, n), z_on(2, n)
    Hm = Hp.to_matrix()
    t = 1.2
    Cd = otoc_squared_commutator(Hm, Wd, Vd, [t], state=psi)[0]
    qc_evo = trotter_circuit(Hp, t, 12)
    qc_ot = otoc_hadamard_test_circuit(prep, qc_evo, [0], [2])
    ReF, sig = run_otoc_test(qc_ot, shots=20000, seed=9)
    Cc = 2 - 2 * ReF
    ok &= abs(Cc - Cd) < 5 * max(sig * 2, 0.02)
    if verbose:
        print(f"  [OTOC test] circuit C = {Cc:.3f} +- {2*sig:.3f} "
              f"vs dense C = {Cd:.3f}")
    return ok


if __name__ == "__main__":
    print("nhq.circuits selftest")
    assert selftest()
    print("all circuit checks passed.")
