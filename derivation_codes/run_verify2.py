#!/usr/bin/env python3
"""Multi-seed driver for the exact-nonlinear-GR cross-check of verify_gr.py.

The definitions of verify_gr.py (spectral derivatives, the exact nonlinear
sqrt(-g)R evaluator, the Richardson extraction, and the grid evaluator for the
symbolic L2/L3) are re-used verbatim: the source is split on the BEGIN REPORT
marker and the prefix is executed, so there is a single implementation of the
physics and no possibility of the two files drifting apart.

What this driver adds is per-seed regeneration of the random field
configurations with an explicit generator, and a report that quotes the
relative deviation of BOTH extracted coefficients (quadratic and cubic)
against the symbolic Lagrangians, followed by the traceless-configuration
check of the vanishing theorem.

Prerequisite: L23.pkl (produced by derive_L3.py) in the working directory.
"""
import numpy as np

_src = open('verify_gr.py').read()
_prefix = _src.split('# --- BEGIN REPORT')[0]
assert _prefix != _src, "BEGIN REPORT marker not found in verify_gr.py"
exec(_prefix)


def random_field(rng):
    """Random low-wavevector field on the grid defined by verify_gr.py."""
    f = np.zeros((N, N))
    for (kw, kq) in kvecs:
        amp = rng.normal(scale=0.5)
        phase = rng.uniform(0, 2 * np.pi)
        f += amp * np.cos(kw * tg + kq * zg + phase)
    return f


print(f"{'seed':>4} {'s2_exact':>13} {'s2_L2':>13} {'rel2':>9} "
      f"{'s3_exact':>13} {'s3_L3':>13} {'rel3':>9}")
for seed in range(4):
    rng = np.random.default_rng(seed)
    a, b, c, k = (random_field(rng) for _ in range(4))
    s2, s3 = taylor_coeffs(a, b, c, k)
    L2v = eval_L(D['L2'], a, b, c, k)
    L3v = eval_L(D['L3'], a, b, c, k)
    print(f"{seed:>4} {s2:>13.8f} {L2v:>13.8f} {abs(s2-L2v)/abs(s2):>9.2e} "
          f"{s3:>13.8f} {L3v:>13.8f} {abs(s3-L3v)/abs(s3):>9.2e}")

print("\nTHEOREM (traceless: c = a, K = 0) -> both coefficients vanish:")
print(f"{'seed':>4} {'s2_exact':>14} {'s3_exact':>14}")
for seed in range(4):
    rng = np.random.default_rng(100 + seed)
    a, b = random_field(rng), random_field(rng)
    c = a.copy()
    k = np.zeros_like(a)
    s2, s3 = taylor_coeffs(a, b, c, k)
    print(f"{seed:>4} {s2:>14.6e} {s3:>14.6e}")
