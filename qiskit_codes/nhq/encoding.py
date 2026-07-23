#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
 nhq.encoding -- qubit encodings of truncated bosonic modes
================================================================================
Standard-binary (SB), Gray, and unary (one-hot) encodings of a d-level bosonic
mode, following Sawaya et al., npj Quantum Inf. 6, 49 (2020), arXiv:1909.12847,
and Di Matteo et al., PRA 103, 042405 (2021), returned as Qiskit SparsePauliOp.

Conventions:
  * Qiskit little-endian: qubit 0 is the LEAST significant / rightmost label
    character in a Pauli string of length n_total.
  * Mode i occupies global qubits [i*nb, (i+1)*nb); mode 0 is therefore the
    least-significant block, matching a dense Kronecker embedding in which
    mode 0 is the least-significant Kronecker factor.  The two orderings are
    easy to transpose silently, which is why the agreement of the encoded and
    dense engines is asserted at run time.
  * For SB with d a power of two, the computational-basis index of the product
    Fock state |n_{M-1} ... n_0> is  sum_i n_i d^i  -- bit-for-bit equal to the
    dense Fock index.  This is what makes SB the bit-for-bit cross-check engine.

Every builder is validated against the dense d x d matrix in selftest().
================================================================================
"""
import numpy as np
from qiskit.quantum_info import SparsePauliOp

# single-qubit outer products as (label, coeff) Pauli pieces
_P1 = {
    (0, 0): [("I", 0.5), ("Z", 0.5)],    # |0><0| = (I+Z)/2
    (1, 1): [("I", 0.5), ("Z", -0.5)],   # |1><1| = (I-Z)/2
    (0, 1): [("X", 0.5), ("Y", 0.5j)],   # |0><1| = sigma^+
    (1, 0): [("X", 0.5), ("Y", -0.5j)],  # |1><0| = sigma^-
}


# ---------------------------------------------------------------- code words
def n_qubits(d, encoding):
    """Qubits per mode for the given encoding."""
    if encoding == "unary":
        return d
    return max(1, int(np.ceil(np.log2(d))))


def code_int(level, d, encoding):
    """Integer code word of Fock level `level` in {0,...,d-1}."""
    if encoding == "standard_binary":
        return level
    if encoding == "gray":
        return level ^ (level >> 1)
    if encoding == "unary":
        return 1 << level                       # one-hot
    raise ValueError(encoding)


def code_bits(level, d, encoding):
    """Bit tuple (bit k = qubit k, little-endian) of the code word."""
    nb = n_qubits(d, encoding)
    c = code_int(level, d, encoding)
    return tuple((c >> k) & 1 for k in range(nb))


def decode_level(bits_int, d, encoding):
    """Inverse of code_int; returns Fock level or None if not a code word."""
    if encoding == "standard_binary":
        return bits_int if bits_int < d else None
    if encoding == "gray":
        # inverse Gray: prefix XOR
        n, x = 0, bits_int
        while x:
            n ^= x
            x >>= 1
        return n if n < d else None
    if encoding == "unary":
        if bits_int != 0 and (bits_int & (bits_int - 1)) == 0:
            lvl = bits_int.bit_length() - 1
            return lvl if lvl < d else None
        return None
    raise ValueError(encoding)


# ----------------------------------------------- outer product -> Pauli terms
def _outer_to_pauli(bits_m, bits_n):
    """|g(m)><g(n)| over nb qubits as [(label, coeff)], label leftmost char =
    highest qubit (Qiskit label order)."""
    nb = len(bits_m)
    terms = [("", 1.0 + 0j)]
    # build label from highest local qubit (leftmost char) down to qubit 0
    for k in range(nb - 1, -1, -1):
        pieces = _P1[(bits_m[k], bits_n[k])]
        terms = [(lab + p, c * cp) for lab, c in terms for p, cp in pieces]
    return terms


def single_mode_op_pauli(mat, d, encoding):
    """Encode a dense d x d single-mode operator `mat` as a SparsePauliOp on
    n_qubits(d, encoding) qubits."""
    nb = n_qubits(d, encoding)
    labels, coeffs = [], []
    for m in range(d):
        bm = code_bits(m, d, encoding)
        for n in range(d):
            if mat[m, n] == 0:
                continue
            bn = code_bits(n, d, encoding)
            for lab, c in _outer_to_pauli(bm, bn):
                labels.append(lab)
                coeffs.append(mat[m, n] * c)
    return SparsePauliOp(labels, coeffs=coeffs).simplify()


# ---------------------------------------------------------- dense references
def dense_a(d):
    return np.diag(np.sqrt(np.arange(1, d)), 1).astype(complex)


def dense_adag(d):
    return dense_a(d).conj().T


def dense_num(d):
    return np.diag(np.arange(d)).astype(complex)


# --------------------------------------------------------------- register
class BosonRegister:
    """M truncated bosonic modes on a qubit register.

    Mode i lives on global qubits [i*nb, (i+1)*nb).  a(i), adag(i), num(i)
    return SparsePauliOp on the full n_total = M*nb qubits.
    """

    def __init__(self, M, d, encoding="gray"):
        self.M, self.d, self.encoding = M, d, encoding
        self.nb = n_qubits(d, encoding)
        self.n_total = M * self.nb
        # cache single-mode Pauli ops
        self._a1 = single_mode_op_pauli(dense_a(d), d, encoding)
        self._ad1 = single_mode_op_pauli(dense_adag(d), d, encoding)
        self._n1 = single_mode_op_pauli(dense_num(d), d, encoding)

    def _pad(self, op, mode):
        """Embed a single-mode SparsePauliOp at `mode` (identities elsewhere)."""
        left = "I" * ((self.M - 1 - mode) * self.nb)   # higher qubits
        right = "I" * (mode * self.nb)                 # lower qubits
        labels = [left + lab + right for lab in op.paulis.to_labels()]
        return SparsePauliOp(labels, coeffs=op.coeffs)

    def a(self, i):
        return self._pad(self._a1, i)

    def adag(self, i):
        return self._pad(self._ad1, i)

    def num(self, i):
        return self._pad(self._n1, i)

    def identity(self):
        return SparsePauliOp("I" * self.n_total)

    def basis_index(self, occ):
        """Computational-basis integer of the product Fock state |occ>."""
        idx = 0
        for i, n in enumerate(occ):
            idx |= code_int(n, self.d, self.encoding) << (i * self.nb)
        return idx

    def decode_bitstring(self, bitstring):
        """Decode a measured bitstring (Qiskit order: leftmost char = highest
        qubit) into a Fock occupation tuple, or None if unphysical (possible
        only for non-power-of-two d or unary leakage)."""
        val = int(bitstring, 2)
        occ = []
        for i in range(self.M):
            block = (val >> (i * self.nb)) & ((1 << self.nb) - 1)
            lvl = decode_level(block, self.d, self.encoding)
            if lvl is None:
                return None
            occ.append(lvl)
        return tuple(occ)


# ------------------------------------------------------------------ selftest
def selftest(verbose=True):
    """Bit-for-bit validation of every encoding against dense matrices."""
    ok = True
    for d in (2, 3, 4, 8):
        for enc in ("standard_binary", "gray", "unary"):
            nb = n_qubits(d, enc)
            a_p = single_mode_op_pauli(dense_a(d), d, enc).to_matrix()
            # dense embedding of a into the code space of the nb-qubit space
            dim = 2 ** nb
            a_ref = np.zeros((dim, dim), complex)
            for n in range(1, d):
                a_ref[code_int(n - 1, d, enc), code_int(n, d, enc)] = np.sqrt(n)
            dev = np.abs(a_p - a_ref).max()
            ok &= dev < 1e-13
            if verbose:
                print(f"  encoding selftest d={d:2d} {enc:16s}: "
                      f"|a_pauli - a_dense| = {dev:.2e}")
            # round trip decode
            for n in range(d):
                assert decode_level(code_int(n, d, enc), d, enc) == n
    # register: SB computational index == dense Fock index (d power of 2)
    reg = BosonRegister(2, 4, "standard_binary")
    for occ in [(0, 0), (3, 1), (2, 3)]:
        assert reg.basis_index(occ) == occ[0] + 4 * occ[1]
    # Gray two-mode commutator spectrum {+1 x12, -3 x4}: the truncated
    # algebra [a, a^dag] = 1 - d|d-1><d-1| on each mode
    regg = BosonRegister(2, 4, "gray")
    comm = (regg.a(0) @ regg.adag(0) - regg.adag(0) @ regg.a(0)).to_matrix()
    ev = np.sort(np.linalg.eigvalsh(comm).real)
    ok &= np.allclose(ev[:4], -3) and np.allclose(ev[4:], 1)
    if verbose:
        print(f"  [a0, a0^dag] Gray 2-mode spectrum check: "
              f"{'pass' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("nhq.encoding selftest")
    assert selftest()
    print("all encoding checks passed.")
