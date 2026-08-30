#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
nhq_tenpy.vertex -- the near-horizon cubic-graviton vertex table.

These four functions are transcribed VERBATIM from the paper's Qiskit reference
module `nhq.model` (the single source of truth for the physics), so the MPS
build uses byte-for-byte identical couplings.  Only numpy and sympy are needed,
so the tensor-network runs carry no Qiskit dependency.

Physics (units hbar = 1, kappa = 1, R = R_S = 1):
    omega_l   = sqrt(l^2 + l + 1)/R
    H3        = sum_{(i<=j),k} V_{ijk} (b_i^dag b_j^dag b_k + h.c.)
    V_{ijk}   = sym * g_eff * G_op[123] * K / sqrt(8 w_i w_j w_k)
    G_op      = (-1)^{m3} G[l1 m1; l2 m2; l3, -m3],  nonzero iff m1+m2 = m3,
                triangle(l1,l2,l3), l1+l2+l3 even  (Lz-conserving operator rule).
    sym       = 2 (i<j) or 1 (i=j)                  (unrestricted-sum fold).
"""
import numpy as np
from sympy.physics.wigner import gaunt as _gaunt
from sympy import N as _symN


def omega_l(l, R=1.0):
    """Near-horizon frequency, anchored to the GGV longitudinal propagator
    pole: mass^2 = lambda/R^2, lambda = l^2 + l + 1 (modeling input, stated)."""
    return np.sqrt(l * l + l + 1) / R


def gaunt_field(l1, m1, l2, m2, l3, m3):
    """Field-label Gaunt overlap  G = int Y_{l1m1} Y_{l2m2} Y_{l3m3} dOmega."""
    return float(_symN(_gaunt(l1, l2, l3, m1, m2, m3)))


def gaunt_operator(l1, m1, l2, m2, l3, m3):
    """OPERATOR-level angular coefficient of b^dag_{l1m1} b^dag_{l2m2} b_{l3m3}:
    creation operators enter with conjugated harmonics, Y*_{lm}=(-1)^m Y_{l,-m},
    hence V ~ int Y*_{l1m1} Y*_{l2m2} Y_{l3m3} = (-1)^{m3} G[l1 m1; l2 m2; l3,-m3]
    (real), nonzero iff m1 + m2 = m3, triangle rule, and l1+l2+l3 even."""
    return ((-1) ** m3) * gaunt_field(l1, m1, l2, m2, l3, -m3)


def cubic_vertex_table(modes, g_eff=1.0, R=1.0, kernel=None, tol=1e-12):
    """Cubic-vertex table {(i, j, k): V} over UNORDERED creation pairs i <= j
    and annihilation index k, for H3 = sum V (b_i^dag b_j^dag b_k + h.c.).

    modes  : list of (l, m) graviton partial waves (l >= 2 only).
    kernel : None -> leading-soft single-p model, K = 1; or callable K(i,j,k).
    The distinct-pair factor 2 folds the unrestricted sum onto i <= j."""
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
