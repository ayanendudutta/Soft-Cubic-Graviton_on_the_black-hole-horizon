#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
 nhq.model -- the derived near-horizon Hamiltonian, engines, and observables
================================================================================
Implements the derived near-horizon physics of the paper
(units hbar = 1, kappa = 1, R = R_S = 1):

  H = H2^grav + [H2^matter + H_int^(h phi phi)] + H3^(hhh),

  H2^grav   = sum_i omega_{l_i} b_i^dag b_i,   omega_l = sqrt(l^2+l+1)/R,
  H3        = sum_{123} V_123 (b_1^dag b_2^dag b_3 + h.c.),  [unrestricted sum]
  V_123     = g_eff * G_op[123] * K(1,2,3) * N_123,
  G_op      = (-1)^{m3} G[l1 m1; l2 m2; l3, -m3]   (operator-level rule of
              the m-convention/Hermiticity remark of the paper: m1 + m2 = m3),
  N_123     = 1/sqrt(8 w1 w2 w3),
  fold i<=j : distinct creation pairs carry the combinatorial factor 2
              (unrestricted-sum fold; gravitationally fixed relative structure),
  H_int     = (g_m/2) sum C_op (b_g^dag a_1 a_2 + h.c.)  with the same
              operator-level conjugation rule (m1 + m2 = m_g), the GGV matter
              vertex after sphere reduction and RWA (toolbox arXiv:2207.11277).

Engines (the multi-engine discipline):
  1. dense Kronecker engine on the full d^M product space (mode 0 = least
     significant factor);
  2. FockBasis engine: H built directly in an occupation basis restricted by
     (a) per-mode cutoff n_i <= d-1 (reproduces engine 1 exactly), and/or
     (b) a GLOBAL total-quanta cutoff N <= Nmax, which -- unlike the per-mode
     cutoff -- commutes with the full su(2) mode-relabelling algebra
     (L_z, L_+/-, L^2) and therefore yields a SYMMETRY-PRESERVING truncation
     P H P whose fixed-(L^2, L_z) spectral statistics can be attributed to the
     physical model once Nmax-stable;
  3. the Qiskit SparsePauliOp engine (nhq.circuits) built from nhq.encoding.

Conserved charges of the corrected H3: L_z = sum m_s n_s (exact), the m -> -m
mode-relabelling parity, and -- because the leading soft vertex is an exact
Wigner-Eckart SU(2) scalar -- total L^2 up to truncation effects (exact under
the total-N truncation; broken only at the occupation boundary of the per-mode
cutoff).  All verified numerically in selftest().

References: GGV JHEP 01 (2022) 023 [arXiv:2012.02357]; GGV JHEP 01 (2022) 146
[arXiv:2110.14673]; toolbox arXiv:2207.11277; DeWitt PR 162 1239 (1967);
Sannan PRD 34 1749 (1986); MSS JHEP 08 (2016) 106 [arXiv:1503.01409];
Polchinski arXiv:1505.08108; Sawaya et al. npj QI 6 49 (2020); Atas et al.
PRL 110, 084101 (2013) (<r> values: Poisson 0.3863, GOE 0.5307, GUE 0.5996).
================================================================================
"""
import itertools
import numpy as np
import scipy.linalg as sla
from sympy.physics.wigner import gaunt as _gaunt
from sympy import N as _symN

# <r> reference values, Atas-Bogomolny-Giraud-Roux, PRL 110, 084101 (2013)
R_POISSON, R_GOE, R_GUE = 0.38629, 0.53070, 0.59950


# ============================================================================
# 1. modes, frequencies, and the corrected gravitational vertex table
# ============================================================================
def omega_l(l, R=1.0):
    """Near-horizon frequency, anchored to the GGV longitudinal propagator
    pole: mass^2 = lambda/R^2, lambda = l^2 + l + 1 (modeling input, stated)."""
    return np.sqrt(l * l + l + 1) / R


def gaunt_field(l1, m1, l2, m2, l3, m3):
    """Field-label Gaunt overlap  G = int Y_{l1m1} Y_{l2m2} Y_{l3m3} dOmega."""
    return float(_symN(_gaunt(l1, l2, l3, m1, m2, m3)))


def gaunt_operator(l1, m1, l2, m2, l3, m3):
    """OPERATOR-level angular coefficient of b^dag_{l1m1} b^dag_{l2m2} b_{l3m3}
    (m-convention/Hermiticity remark of the paper): creation operators enter
    with conjugated harmonics, Y*_{lm} = (-1)^m Y_{l,-m}, hence

        V ~ int dOmega Y*_{l1m1} Y*_{l2m2} Y_{l3m3}
          = (-1)^{m3} G[l1 m1; l2 m2; l3, -m3]        (real),

    nonzero iff m1 + m2 = m3 (Lz of the created pair = Lz of the destroyed
    graviton), triangle rule, and l1+l2+l3 even.  Transcribing the field-label
    rule m1+m2+m3 = 0 onto operator labels silently violates Lz; the
    conservation is asserted at run time for that reason."""
    return ((-1) ** m3) * gaunt_field(l1, m1, l2, m2, l3, -m3)


def cubic_vertex_table(modes, g_eff=1.0, R=1.0, kernel=None, tol=1e-12):
    """Cubic-vertex table {(i, j, k): V} over UNORDERED creation pairs i <= j
    and annihilation index k, for H3 = sum V (b_i^dag b_j^dag b_k + h.c.).

    modes  : list of (l, m) graviton partial waves (l >= 2 only; l = 0, 1 are
             gauge-special and excluded, per the low-multipole modeling choice
             of the paper).
    kernel : None -> leading-soft single-p model, K = 1 (K then absorbed
             exactly into g_eff for single-l registers; for multi-l registers
             this is the stated leading-soft modeling input); or a callable
             K(mode_i, mode_j, mode_k).
    The distinct-pair factor 2 folds the unrestricted sum of Eq. (24) onto
    i <= j -- part of the gravitationally fixed relative structure."""
    w = [omega_l(l, R) for (l, _) in modes]
    M = len(modes)
    table = {}
    for i in range(M):
        for j in range(i, M):
            sym = 2.0 if i < j else 1.0
            for k in range(M):
                (l1, m1), (l2, m2), (l3, m3) = modes[i], modes[j], modes[k]
                if m1 + m2 != m3:
                    continue
                G = gaunt_operator(l1, m1, l2, m2, l3, m3)
                if abs(G) < tol:
                    continue
                K = 1.0 if kernel is None else kernel(modes[i], modes[j], modes[k])
                V = sym * g_eff * G * K / np.sqrt(8.0 * w[i] * w[j] * w[k])
                if abs(V) > tol:
                    table[(i, j, k)] = V
    return table


def matter_vertex_table(g_modes, s_modes, g_m=1.0, R=1.0, tol=1e-12):
    """Matter vertex table {(g, (i<=j)): C} for
    H_int = sum C (b_g^dag a_i a_j + h.c.)   [RWA of the GGV h phi phi vertex]
    with the same operator-level conjugation rule: coefficient
    (-1)^{m_g} G[l_i m_i; l_j m_j; l_g, -m_g], nonzero iff m_i + m_j = m_g,
    distinct scalar pairs folded with factor 2, and normalisation
    1/sqrt(8 w_g w_i w_j).  The overall matter coupling g_m plays the role of
    (gamma/2) x contraction constants; only its relative structure is used."""
    wg = [omega_l(l, R) for (l, _) in g_modes]
    ws = [omega_l(l, R) for (l, _) in s_modes]
    table = {}
    for g, (lg, mg) in enumerate(g_modes):
        for i in range(len(s_modes)):
            for j in range(i, len(s_modes)):
                (l1, m1), (l2, m2) = s_modes[i], s_modes[j]
                if m1 + m2 != mg:
                    continue
                G = ((-1) ** mg) * gaunt_field(l1, m1, l2, m2, lg, -mg)
                if abs(G) < tol:
                    continue
                sym = 2.0 if i < j else 1.0
                C = sym * g_m * G / np.sqrt(8.0 * wg[g] * ws[i] * ws[j])
                if abs(C) > tol:
                    table[(g, i, j)] = C
    return table


# ------------------- structure ablations (the non-tautology test) -----------
def ablate_uniform(table):
    """Same allowed triples, all |V| replaced by the mean |V| (signs kept):
    kills the gravitationally fixed Gaunt weighting but keeps the sparsity."""
    mean = np.mean([abs(v) for v in table.values()])
    return {key: np.sign(v) * mean for key, v in table.items()}


def ablate_shuffled(table, seed=0):
    """Permute the vertex magnitudes among the allowed triples (signs kept):
    keeps the value multiset and sparsity, destroys the structural assignment."""
    rng = np.random.default_rng(seed)
    keys = list(table.keys())
    mags = np.array([abs(table[k]) for k in keys])
    rng.shuffle(mags)
    return {k: np.sign(table[k]) * m for k, m in zip(keys, mags)}


# ============================================================================
# 2. FockBasis: occupation-basis engine with per-mode and/or total-N cutoffs
# ============================================================================
class FockBasis:
    """Occupation-number basis for M modes, restricted by:
        n_i <= d - 1        (per-mode Fock cutoff; d = None -> no per-mode cap)
        sum_i n_i <= Nmax   (global total-quanta cutoff; None -> no cap)
        Lz fixed            (optional symmetry restriction; None -> all)
    Provides H = H2 + H3 (+ matter), N, Lz, L+-, L^2, parity, all as dense
    matrices in the restricted basis, plus index maps to the full d^M product
    space when a per-mode cutoff d is given.
    """

    def __init__(self, modes, d=None, Nmax=None, lz=None, R=1.0):
        if d is None and Nmax is None:
            raise ValueError("need at least one of d, Nmax")
        self.modes, self.d, self.Nmax, self.lz, self.R = modes, d, Nmax, lz, R
        self.M = len(modes)
        self.w = np.array([omega_l(l, R) for (l, _) in modes])
        self.ms = np.array([m for (_, m) in modes])
        self.ls = np.array([l for (l, _) in modes])
        cap = (d - 1) if d is not None else (Nmax if Nmax is not None else 0)
        # Direct enumeration of the admissible occupations.  Filtering the
        # full (cap+1)^M product space instead is catastrophic under a total-N
        # cut: at M = 12, Nmax = 5 that is 2.2e9 tuples for a 564-state basis.
        # The recursion below emits EXACTLY the same tuples in EXACTLY the same
        # (lexicographic) order as itertools.product-then-filter, so all matrix
        # indices are unchanged; it is a pure performance fix.
        Ncap = Nmax if Nmax is not None else cap * self.M
        occs = []
        buf = [0] * self.M
        M = self.M

        def _rec(pos, remaining):
            if pos == M:
                occs.append(tuple(buf))
                return
            for n in range(min(cap, remaining) + 1):
                buf[pos] = n
                _rec(pos + 1, remaining - n)
            buf[pos] = 0

        _rec(0, Ncap)
        if lz is not None:
            occs = [o for o in occs if int(np.dot(self.ms, o)) == lz]
        # deterministic order: by dense index when d given, else lexicographic
        if d is not None:
            occs.sort(key=lambda o: sum(n * d ** s for s, n in enumerate(o)))
        self.occs = occs
        self.index = {o: p for p, o in enumerate(occs)}
        self.dim = len(occs)

    # ------------------------------------------------------------ helpers
    def dense_indices(self):
        """Indices of the basis states in the full d^M product space
        (mode 0 least significant); requires a per-mode cutoff d."""
        assert self.d is not None
        d = self.d
        return np.array([sum(n * d ** s for s, n in enumerate(o))
                         for o in self.occs])

    def embed(self, psi, d=None):
        """Embed a basis-restricted state into the full d^M product space."""
        d = d or self.d
        assert d is not None
        full = np.zeros(d ** self.M, complex)
        for p, o in enumerate(self.occs):
            if max(o) <= d - 1:
                full[sum(n * d ** s for s, n in enumerate(o))] = psi[p]
        return full

    def _apply_bd_bd_b(self, occ, i, j, k):
        """b_i^dag b_j^dag b_k |occ> -> (amp, occ') or None if annihilated or
        outside the basis (per-mode cap and/or total-N cap and/or Lz)."""
        if occ[k] == 0:
            return None
        amp = np.sqrt(occ[k])
        o = list(occ)
        o[k] -= 1
        for s in (j, i):                        # sequential: repeated indices exact
            if self.d is not None and o[s] + 1 > self.d - 1:
                return None
            amp *= np.sqrt(o[s] + 1)
            o[s] += 1
        o = tuple(o)
        p = self.index.get(o)
        return None if p is None else (amp, p)

    # ------------------------------------------------------------ operators
    def H2_diag(self):
        return np.array([np.dot(self.w, o) for o in self.occs])

    def N_diag(self):
        return np.array([sum(o) for o in self.occs], dtype=float)

    def Lz_diag(self):
        return np.array([int(np.dot(self.ms, o)) for o in self.occs],
                        dtype=float)

    def H(self, table, include_H2=True):
        """H2 + H3 in the restricted basis (dense, Hermitian).  For a per-mode
        cutoff this equals the truncated Kronecker engine exactly; for a
        total-N cutoff it is the symmetry-preserving projection P H P."""
        H = np.zeros((self.dim, self.dim), complex)
        if include_H2:
            H[np.diag_indices(self.dim)] = self.H2_diag()
        for (i, j, k), V in table.items():
            for p, occ in enumerate(self.occs):
                hit = self._apply_bd_bd_b(occ, i, j, k)
                if hit is None:
                    continue
                amp, q = hit
                H[q, p] += V * amp
                H[p, q] += np.conj(V * amp)
        return H

    def ladder(self, i, dagger=False):
        """Single-mode ladder operator as a dense matrix in the basis
        (transitions leaving the basis dropped -- the truncation)."""
        A = np.zeros((self.dim, self.dim), complex)
        for p, occ in enumerate(self.occs):
            o = list(occ)
            if dagger:
                if self.d is not None and o[i] + 1 > self.d - 1:
                    continue
                amp = np.sqrt(o[i] + 1)
                o[i] += 1
            else:
                if o[i] == 0:
                    continue
                amp = np.sqrt(o[i])
                o[i] -= 1
            q = self.index.get(tuple(o))
            if q is not None:
                A[q, p] = amp
        return A

    def quadrature(self, i, which="u"):
        a = self.ladder(i)
        ad = self.ladder(i, dagger=True)
        if which == "u":
            return (a + ad) / np.sqrt(2)
        return -1j * (a - ad) / np.sqrt(2)

    def Lp(self):
        """L_+ = sum_{l,m} sqrt(l(l+1) - m(m+1)) b^dag_{l,m+1} b_{l,m}
        (mode-relabelling su(2); requires the register to contain (l, m+1))."""
        L = np.zeros((self.dim, self.dim), complex)
        midx = {mm: s for s, mm in enumerate(self.modes)}
        for s, (l, m) in enumerate(self.modes):
            t = midx.get((l, m + 1))
            if t is None:
                continue
            c = np.sqrt(l * (l + 1) - m * (m + 1))
            for p, occ in enumerate(self.occs):
                if occ[s] == 0:
                    continue
                o = list(occ)
                amp = np.sqrt(o[s]); o[s] -= 1
                if self.d is not None and o[t] + 1 > self.d - 1:
                    continue
                amp *= np.sqrt(o[t] + 1); o[t] += 1
                q = self.index.get(tuple(o))
                if q is not None:
                    L[q, p] += c * amp
        return L

    def L2(self):
        """L^2 = L_+ L_- + Lz^2 - Lz, with the composite L_+ L_- applied at
        the OCCUPATION level through intermediate configurations that may lie
        OUTSIDE an Lz-restricted basis (L_- maps lz -> lz - 1; the product
        returns).  Building L_+ L_- as a product of sector-restricted ladder
        matrices instead gives identically zero -- the bug this replaces.
        Per-mode cutoff d is enforced on intermediates (the standard
        truncated algebra: this is exactly the cutoff-induced SU(2)
        breaking); the total-N truncation needs no intermediate cut (L_+-
        conserve N), so there L^2 is exact."""
        # single-hop terms of L_- : coeff * b†_(l,m-1) b_(l,m)
        hops_minus = []
        for s, (l, m) in enumerate(self.modes):
            if (l, m - 1) in self.modes:
                t = self.modes.index((l, m - 1))
                c = np.sqrt(l * (l + 1) - m * (m - 1))
                if c != 0.0:
                    hops_minus.append((t, s, c))
        L2 = np.zeros((self.dim, self.dim), complex)
        dmax = None if self.d is None else self.d - 1
        for p, occ in enumerate(self.occs):
            # accumulate L_-|p> as {intermediate occ: amplitude}
            inter = {}
            for (t, s, c) in hops_minus:
                if occ[s] == 0:
                    continue
                o = list(occ)
                amp = c * np.sqrt(o[s])
                o[s] -= 1
                if dmax is not None and o[t] + 1 > dmax:
                    continue
                amp *= np.sqrt(o[t] + 1)
                o[t] += 1
                key = tuple(o)
                inter[key] = inter.get(key, 0.0) + amp
            # apply L_+ = (L_-)^dagger : coeff * b†_(l,m) b_(l,m-1)
            for o1, a1 in inter.items():
                for (t, s, c) in hops_minus:
                    # L_+ hop: annihilate at t=(l,m-1), create at s=(l,m)
                    if o1[t] == 0:
                        continue
                    o = list(o1)
                    amp = c * np.sqrt(o[t])
                    o[t] -= 1
                    if dmax is not None and o[s] + 1 > dmax:
                        continue
                    amp *= np.sqrt(o[s] + 1)
                    o[s] += 1
                    q = self.index.get(tuple(o))
                    if q is not None:
                        L2[q, p] += a1 * amp
        Lzd = self.Lz_diag()
        L2 += np.diag(Lzd ** 2 - Lzd)
        return L2

    def parity_perm(self):
        """Permutation matrix of the mode relabelling (l, m) -> (l, -m);
        requires the register (and the basis restriction) closed under it."""
        perm = [self.modes.index((l, -m)) for (l, m) in self.modes]
        P = np.zeros((self.dim, self.dim))
        for p, occ in enumerate(self.occs):
            occ_p = tuple(occ[perm[s]] for s in range(self.M))
            q = self.index.get(occ_p)
            if q is None:
                raise ValueError("basis not closed under m -> -m")
            P[q, p] = 1.0
        return P

    # ----------------------------------------------------- mixed g + s register
    @staticmethod
    def mixed(g_modes, s_modes, d, Nmax=None, lz=None, R=1.0):
        """Basis for a graviton (+) scalar register: modes are concatenated
        [gravitons..., scalars...]; returns (basis, n_g) with n_g the number of
        graviton modes (graviton indices 0..n_g-1)."""
        basis = FockBasis(list(g_modes) + list(s_modes), d=d, Nmax=Nmax,
                          lz=lz, R=R)
        return basis, len(g_modes)

    def H_mixed(self, g_table, m_table, n_g, include_H2=True):
        """H2(all modes) + H3(gravitons) + H_int(b_g^dag a_i a_j + h.c.),
        graviton mode indices 0..n_g-1, scalar indices n_g..M-1 (m_table keys
        (g, i, j) use LOCAL scalar indices i, j)."""
        H = self.H(g_table, include_H2=include_H2)      # H2 + H3 (g_table on globals)
        for (g, i, j), C in m_table.items():
            gi, si, sj = g, n_g + i, n_g + j
            for p, occ in enumerate(self.occs):
                # b_g^dag a_i a_j |occ>
                if occ[sj] == 0:
                    continue
                o = list(occ); amp = np.sqrt(o[sj]); o[sj] -= 1
                if o[si] == 0:
                    continue
                amp *= np.sqrt(o[si]); o[si] -= 1
                if self.d is not None and o[gi] + 1 > self.d - 1:
                    continue
                amp *= np.sqrt(o[gi] + 1); o[gi] += 1
                q = self.index.get(tuple(o))
                if q is None:
                    continue
                H[q, p] += C * amp
                H[p, q] += np.conj(C * amp)
        return H


# ============================================================================
# 3. dense Kronecker engine (independent cross-check of FockBasis)
# ============================================================================
def _embed_dense(op1, site, M, d):
    """Mode 0 = least-significant Kronecker factor (mode-ordering fix)."""
    mats = [np.eye(d, dtype=complex)] * M
    mats[site] = op1.astype(complex)
    out = mats[M - 1]
    for s in range(M - 2, -1, -1):
        out = np.kron(out, mats[s])
    return out


def dense_ladders(M, d):
    a1 = np.diag(np.sqrt(np.arange(1, d)), 1).astype(complex)
    a = [_embed_dense(a1, i, M, d) for i in range(M)]
    ad = [x.conj().T for x in a]
    n = [x.conj().T @ x for x in a]
    return a, ad, n


def H_dense_kron(modes, table, d, R=1.0):
    """H2 + H3 on the full d^M product space via Kronecker products."""
    M = len(modes)
    a, ad, n = dense_ladders(M, d)
    H = np.zeros((d ** M, d ** M), complex)
    for i, (l, _) in enumerate(modes):
        H += omega_l(l, R) * n[i]
    for (i, j, k), V in table.items():
        T = ad[i] @ ad[j] @ a[k]
        H += V * (T + T.conj().T)
    return H


# ============================================================================
# 4. observables
# ============================================================================
def multiplicity_distribution(psi, basis):
    """p_N over total quanta, from a basis-restricted state."""
    probs = np.abs(psi) ** 2
    Ns = basis.N_diag().astype(int)
    pN = {}
    for p, N in enumerate(Ns):
        pN[N] = pN.get(N, 0.0) + probs[p]
    return pN


def sector_number_distribution(psi, basis, mode_subset):
    """p_n of the total occupation of a subset of modes (e.g. scalar sector)."""
    probs = np.abs(psi) ** 2
    pN = {}
    for p, occ in enumerate(basis.occs):
        n = sum(occ[s] for s in mode_subset)
        pN[n] = pN.get(n, 0.0) + probs[p]
    return pN


def inelasticity(psi, basis, N0):
    return 1.0 - multiplicity_distribution(psi, basis).get(N0, 0.0)


def shannon(p):
    p = np.asarray([v for v in p if v > 1e-15])
    return float(-np.sum(p * np.log(p)))


def production_entropy(psi, basis):
    return shannon(list(multiplicity_distribution(psi, basis).values()))


def entanglement_entropy(psi, basis, partA):
    """Von Neumann entropy (nats) of the mode bipartition A|rest, computed
    directly in the restricted occupation basis (no full-space embedding):
    amplitudes are scattered into a (#occ_A) x (#occ_B) matrix and SVD'd."""
    partA = sorted(partA)
    partB = [s for s in range(basis.M) if s not in partA]
    occA = {}
    occB = {}
    entries = []
    for p, occ in enumerate(basis.occs):
        oa = tuple(occ[s] for s in partA)
        ob = tuple(occ[s] for s in partB)
        ia = occA.setdefault(oa, len(occA))
        ib = occB.setdefault(ob, len(occB))
        entries.append((ia, ib, psi[p]))
    mat = np.zeros((len(occA), len(occB)), complex)
    for ia, ib, amp in entries:
        mat[ia, ib] = amp
    s = np.linalg.svd(mat, compute_uv=False)
    return shannon(s ** 2)


def page_profile(psi, basis, max_subsets=200, seed=1):
    """Mean entanglement entropy versus subsystem size k = 1..M-1, averaged
    over all (or up to max_subsets random) k-mode subsets."""
    rng = np.random.default_rng(seed)
    M = basis.M
    prof = {}
    for k in range(1, M):
        subs = list(itertools.combinations(range(M), k))
        if len(subs) > max_subsets:
            subs = [tuple(sorted(rng.choice(M, size=k, replace=False)))
                    for _ in range(max_subsets)]
        vals = [entanglement_entropy(psi, basis, list(A)) for A in subs]
        prof[k] = (float(np.mean(vals)), float(np.std(vals)))
    return prof


def random_state_reference(basis, n_samples=32, seed=7, max_subsets=60,
                           want_page=True):
    """Ergodic (Haar-random within the restricted basis) references:
    the mean multiplicity distribution and the mean Page profile.  The p_N
    reference is analytic: <P_N>_Haar = dim(N-sector within basis)/dim(basis);
    the Page profile is sampled."""
    Ns = basis.N_diag().astype(int)
    pN_ref = {int(N): float(np.mean(Ns == N)) for N in np.unique(Ns)}
    prof = None
    if want_page:
        rng = np.random.default_rng(seed)
        acc = {}
        for _ in range(n_samples):
            v = rng.normal(size=basis.dim) + 1j * rng.normal(size=basis.dim)
            v /= np.linalg.norm(v)
            pr = page_profile(v, basis, max_subsets=max_subsets, seed=seed)
            for k, (m, _) in pr.items():
                acc.setdefault(k, []).append(m)
        prof = {k: (float(np.mean(v)), float(np.std(v))) for k, v in acc.items()}
    return pN_ref, prof


def diagonal_ensemble_pN(H, psi0, basis):
    """Infinite-time-averaged multiplicity distribution
    p̄_N = sum_n |<n|psi0>|^2 <n|P_N|n>  from one eigendecomposition (assumes
    non-degenerate spectrum contributions; degeneracies within a symmetry
    block are handled by working inside the block)."""
    E, V = np.linalg.eigh(H)
    c2 = np.abs(V.conj().T @ psi0) ** 2
    Ns = basis.N_diag().astype(int)
    pbar = {}
    for N in np.unique(Ns):
        PN = (Ns == N).astype(float)
        w = np.einsum("in,i->n", np.abs(V) ** 2, PN)
        pbar[int(N)] = float(np.dot(c2, w))
    return pbar


# ------------------------ reference distributions & distances ---------------
def microcanonical_pN(H, psi0, basis, width_sigmas=1.0):
    """Microcanonical / ETH-window reference: the eigenstate-averaged
    multiplicity distribution over the energy shell actually explored by the
    initial state, [E0 - w, E0 + w] with E0 = <psi0|H|psi0> and
    w = width_sigmas * sqrt(<H^2> - <H>^2).  This is the fair ergodic
    baseline for the diagonal ensemble (energy conservation confines the
    dynamics to this shell; comparing against Haar over the whole block is
    trivially distinguishable).  Returns (pN, E0, w, n_states)."""
    E, V = np.linalg.eigh(H)
    E0 = float(np.real(np.vdot(psi0, H @ psi0)))
    E2 = float(np.real(np.vdot(psi0, H @ (H @ psi0))))
    w = width_sigmas * np.sqrt(max(E2 - E0 ** 2, 1e-30))
    sel = np.where(np.abs(E - E0) <= w)[0]
    if len(sel) == 0:
        sel = np.array([int(np.argmin(np.abs(E - E0)))])
    Ns = basis.N_diag().astype(int)
    pN = {}
    W2 = np.abs(V[:, sel]) ** 2                    # (dim, n_sel)
    for N in np.unique(Ns):
        mask = (Ns == N).astype(float)
        pN[int(N)] = float(np.mean(mask @ W2))
    return pN, E0, float(w), int(len(sel))


def thermal_pN(basis, target_meanN, beta_bracket=(1e-4, 60.0)):
    """Gibbs p_N of the free H2 restricted to the basis, with beta tuned so
    <N> matches target_meanN.  Returns (pN, beta)."""
    E = basis.H2_diag()
    Ns = basis.N_diag()

    def meanN(beta):
        w = np.exp(-beta * (E - E.min()))
        w /= w.sum()
        return float(np.dot(w, Ns))

    lo, hi = beta_bracket
    if not (meanN(hi) <= target_meanN <= meanN(lo)):
        # fall back: clamp
        beta = hi if target_meanN < meanN(hi) else lo
    else:
        from scipy.optimize import brentq
        beta = brentq(lambda b: meanN(b) - target_meanN, lo, hi)
    w = np.exp(-beta * (E - E.min()))
    w /= w.sum()
    pN = {}
    for p, N in enumerate(Ns.astype(int)):
        pN[int(N)] = pN.get(int(N), 0.0) + float(w[p])
    return pN, beta


def poisson_pN(target_meanN, support):
    """Coherent-state (Poisson) p_N at matched mean, renormalised on the
    accessible support."""
    from math import factorial, exp
    lam = max(target_meanN, 1e-12)
    p = {N: exp(-lam) * lam ** N / factorial(N) for N in support}
    Z = sum(p.values())
    return {N: v / Z for N, v in p.items()}


def dist_TV(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def dist_KL(p, q, eps=1e-15):
    keys = set(p) | set(q)
    return sum(p.get(k, 0.0) * np.log((p.get(k, 0.0) + eps) /
                                      (q.get(k, 0.0) + eps)) for k in keys)


# ------------------------ spectral diagnostics ------------------------------
def r_ratio(E, mid_fraction=(0.1, 0.9), degeneracy_tol=1e-10):
    """Mean adjacent-gap ratio <r> in a mid-spectrum window."""
    E = np.sort(np.asarray(E).real)
    n = len(E)
    E = E[int(mid_fraction[0] * n):int(mid_fraction[1] * n)]
    s = np.diff(E)
    s = s[s > degeneracy_tol]
    if len(s) < 3:
        return np.nan, 0
    r = np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])
    return float(np.mean(r)), len(s)


def unfolded_spacings(E, mid_fraction=(0.1, 0.9)):
    """Locally-unfolded nearest-neighbour spacings (mean 1) for histograms."""
    E = np.sort(np.asarray(E).real)
    n = len(E)
    E = E[int(mid_fraction[0] * n):int(mid_fraction[1] * n)]
    s = np.diff(E)
    s = s[s > 1e-12]
    return s / np.mean(s)


def simultaneous_blocks(H, symmetries, tol=1e-7):
    """Split H into blocks of the simultaneous eigenspaces of a list of
    commuting Hermitian `symmetries` (each must commute with H).  Returns a
    list of (labels_tuple, H_block)."""
    blocks = [(tuple(), np.arange(H.shape[0]), np.eye(H.shape[0]))]
    for S in symmetries:
        new = []
        for labels, idx, U in blocks:
            Sb = U.conj().T @ S @ U
            ev, W = np.linalg.eigh((Sb + Sb.conj().T) / 2)
            # group eigenvalues
            order = np.argsort(ev)
            ev, W = ev[order], W[:, order]
            start = 0
            for t in range(1, len(ev) + 1):
                if t == len(ev) or abs(ev[t] - ev[start]) > tol:
                    lab = float(np.round(np.mean(ev[start:t]), 6))
                    new.append((labels + (lab,), idx, U @ W[:, start:t]))
                    start = t
        blocks = new
    out = []
    for labels, _, U in blocks:
        Hb = U.conj().T @ H @ U
        out.append((labels, (Hb + Hb.conj().T) / 2))
    return out


def spectral_form_factor(E, ts, beta=0.0):
    E = np.asarray(E).real
    Z = np.array([np.sum(np.exp(-beta * E - 1j * E * t)) for t in ts])
    sff = np.abs(Z) ** 2
    return sff / sff[0]


# ------------------------ dynamics & OTOC -----------------------------------
class BlockEvolver:
    """Exact evolution inside a basis via one eigendecomposition."""

    def __init__(self, H):
        self.E, self.V = np.linalg.eigh(H)

    def psi(self, psi0, t):
        c = self.V.conj().T @ psi0
        return self.V @ (np.exp(-1j * self.E * t) * c)

    def U(self, t):
        return (self.V * np.exp(-1j * self.E * t)) @ self.V.conj().T


def evolve_krylov(H, psi0, t):
    """Single-time e^{-iHt} psi0 via scipy's Krylov expm_multiply, WITHOUT
    diagonalising H.  For one (or a few) snapshot times on a large block this
    is far cheaper than BlockEvolver's full eigh: it needs only the sparse
    action of H.  BlockEvolver is the cheaper choice when many times on a
    single H are required (the eigh is then amortised); this routine is the
    cheaper choice when few times but many different H are required."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import expm_multiply
    Hs = csr_matrix(H)
    return expm_multiply(-1j * t * Hs, psi0.astype(complex))


def otoc_squared_commutator(H, W, V, ts, state=None, beta=None):
    """C(t) = <[W(t), V]^dag [W(t), V]>, MSS squared-commutator convention
    (arXiv:1503.01409); the growth rate of C is 2x the trajectory Lyapunov
    rate (Polchinski arXiv:1505.08108)."""
    ev = BlockEvolver(H)
    if state is None and beta is not None:
        w = np.exp(-beta * (ev.E - ev.E.min()))
        w /= w.sum()
        rho = (ev.V * w) @ ev.V.conj().T
    out = []
    for t in ts:
        U = ev.U(t)
        Wt = U.conj().T @ W @ U
        comm = Wt @ V - V @ Wt
        Msq = comm.conj().T @ comm
        if state is not None:
            out.append(float(np.real(np.vdot(state, Msq @ state))))
        elif beta is not None:
            out.append(float(np.real(np.trace(rho @ Msq))))
        else:
            out.append(float(np.real(np.trace(Msq)) / Msq.shape[0]))
    return np.array(out)


# ============================================================================
# selftest
# ============================================================================
def selftest(verbose=True):
    ok = True
    modes = [(2, m) for m in range(-2, 3)]          # full l = 2 multiplet
    d = 4
    table = cubic_vertex_table(modes, g_eff=0.6)

    # (a) operator-level selection rule and phase-form cross-check
    assert abs(gaunt_operator(2, 1, 2, 1, 2, 2)) > 1e-6     # allowed splitting
    assert abs(gaunt_field(2, 1, 2, 1, 2, -2)) > 1e-6       # field rule differs
    f1 = gaunt_operator(2, 2, 2, 0, 2, 2)
    f2 = ((-1) ** 2) * gaunt_field(2, -2, 2, 0, 2, 2)
    ok &= abs(f1 - f2) < 1e-12

    # (b) FockBasis (per-mode cutoff) vs Kronecker engine, full space
    fb = FockBasis(modes, d=d)
    H1 = fb.H(table)
    H2 = H_dense_kron(modes, table, d)
    dev = np.abs(H1 - H2).max()
    ok &= dev < 1e-12
    if verbose:
        print(f"  [engines] FockBasis vs Kronecker: max dev = {dev:.2e}")

    # (c) conserved charges: Lz exact; parity exact; L2 exact under total-N cut
    Lz = np.diag(fb.Lz_diag())
    devLz = np.abs(H1 @ Lz - Lz @ H1).max()
    P = fb.parity_perm()
    devP = np.abs(H1 @ P - P @ H1).max()
    ok &= devLz < 1e-12 and devP < 1e-12
    fbN = FockBasis(modes, Nmax=3)                   # symmetry-preserving cut
    HN = fbN.H(table)
    L2 = fbN.L2()
    devL2 = np.abs(HN @ L2 - L2 @ HN).max()
    ok &= devL2 < 1e-10
    LzN = np.diag(fbN.Lz_diag())
    ok &= np.abs(L2 @ LzN - LzN @ L2).max() < 1e-10
    if verbose:
        print(f"  [charges] ||[H,Lz]||={devLz:.1e} ||[H,P]||={devP:.1e} "
              f"||[H_Ncut,L2]||={devL2:.1e}")

    # (d) per-mode cutoff breaks L2 only at the occupation boundary
    L2d = fb.L2()
    devL2d = np.abs(H1 @ L2d - L2d @ H1).max()
    ok &= devL2d > 1e-6                              # boundary breaking present
    if verbose:
        print(f"  [cutoff systematic] per-mode d={d}: ||[H,L2]|| = "
              f"{devL2d:.2e} (nonzero boundary artefact, as documented)")

    # (e) Lz-block engine consistency with full-space projection
    fb0 = FockBasis(modes, d=d, lz=0)
    H0 = fb0.H(table)
    idx = [fb.index[o] for o in fb0.occs]
    dev = np.abs(H0 - H1[np.ix_(idx, idx)]).max()
    ok &= dev < 1e-12
    if verbose:
        print(f"  [Lz block] block engine vs projected full H: {dev:.2e}")

    # (f) H3 changes N by exactly +-1
    Ns = fb.N_diag()
    H3 = H1 - np.diag(fb.H2_diag())
    bad = 0
    nz = np.argwhere(np.abs(H3) > 1e-9)
    for p, q in nz:
        if abs(Ns[p] - Ns[q]) != 1:
            bad += 1
    ok &= bad == 0
    if verbose:
        print(f"  [number rule] all H3 elements connect |Delta N| = 1: "
              f"{'pass' if bad == 0 else 'FAIL'}")

    # (g) SU(2)-scalar (Wigner-Eckart) structure of the unfolded vertex:
    # V_unfolded / 3j-symbol constant across the l = 2 multiplet
    from sympy.physics.wigner import wigner_3j
    ratios = []
    for (i, j, k), V in table.items():
        (l1, m1), (l2, m2), (l3, m3) = modes[i], modes[j], modes[k]
        sym = 2.0 if i < j else 1.0
        w3 = float(_symN(wigner_3j(l1, l2, l3, m1, m2, -m3)))
        if abs(w3) > 1e-12:
            ratios.append((V / sym) / (((-1) ** m3) * w3))
    spread = np.ptp(ratios) / abs(np.mean(ratios))
    ok &= spread < 1e-12
    if verbose:
        print(f"  [Wigner-Eckart] unfolded V / 3j ratio spread = {spread:.1e}")

    # (h) entanglement entropy: basis-restricted vs full-space reshape
    rng = np.random.default_rng(0)
    psi = rng.normal(size=fb.dim) + 1j * rng.normal(size=fb.dim)
    psi /= np.linalg.norm(psi)
    S1 = entanglement_entropy(psi, fb, [0, 1])
    psi_t = psi.reshape([d] * fb.M)                 # axis 0 = mode M-1
    axes = [fb.M - 1 - s for s in [0, 1]]
    rest = [ax for ax in range(fb.M) if ax not in axes]
    mat = np.transpose(psi_t, axes + rest).reshape(d ** 2, -1)
    sv = np.linalg.svd(mat, compute_uv=False)
    S2 = shannon(sv ** 2)
    ok &= abs(S1 - S2) < 1e-10
    if verbose:
        print(f"  [entanglement] basis-SVD vs reshape-SVD: "
              f"|dS| = {abs(S1 - S2):.1e}")

    # (i) diagonal ensemble equals long-time average (numerical check)
    fbs = FockBasis(modes, d=3, lz=0)
    Hs = fbs.H(cubic_vertex_table(modes, g_eff=1.0))
    psi0 = np.zeros(fbs.dim, complex)
    psi0[fbs.index[(0, 0, 1, 0, 0)]] = 1.0          # one graviton in (2,0)
    pbar = diagonal_ensemble_pN(Hs, psi0, fbs)
    ev = BlockEvolver(Hs)
    ts = np.linspace(200, 1200, 601)
    acc = {}
    for t in ts:
        for N, v in multiplicity_distribution(ev.psi(psi0, t), fbs).items():
            acc[N] = acc.get(N, 0.0) + v / len(ts)
    dev = max(abs(pbar.get(N, 0) - acc.get(N, 0)) for N in set(pbar) | set(acc))
    ok &= dev < 5e-3
    if verbose:
        print(f"  [diag ensemble] vs long-time average: max dev = {dev:.1e}")
    return ok


if __name__ == "__main__":
    print("nhq.model selftest")
    assert selftest()
    print("all model checks passed.")
