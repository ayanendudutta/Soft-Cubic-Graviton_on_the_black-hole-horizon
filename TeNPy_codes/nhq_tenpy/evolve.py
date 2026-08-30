#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
nhq_tenpy.evolve -- real-time evolution and MPS observables for the multipole
                    ladder extension of the paper.

Two facts about the model dictate the method:

  * H3 is a genuinely LONG-RANGE, 3-body bosonic vertex (b_i^dag b_j^dag b_k),
    coupling non-adjacent sites via the Gaunt/Wigner-3j rule.  Two-site TDVP
    cannot generate such a term from a product (chi = 1) initial state -- the
    local two-site projector sees no matrix element to the split channel, the
    well-known cold-start failure of TDVP for long-range Hamiltonians.
  * Forming exp(-iH dt) as an MPO (W^II / ExpMPOEvolution) DOES seed the
    entanglement from a product state, but its memory cost blows up at large M
    because the intermediate bond is chi * chi_MPO.

We therefore evolve HYBRID: a short ExpMPO warm-up (cheap while chi is still
small) seeds the physical entanglement, after which two-site TDVP (memory
O(chi^2 chi_MPO), so affordable at large M) carries the bulk of the trajectory
and grows the bond only as far as the physics demands.  The bond dimension it
needs is itself the primary diagnostic: bounded chi <=> perturbative rigidity;
chi growing toward the Page value <=> onset of scrambling and breakdown of the
matrix-product description.

Observables are computed WITHOUT ever forming the d^M statevector:

  * P(N, t): characteristic function chi(theta) = <psi| e^{i theta N_tot} |psi>
    of the total-number operator (a bond-1 diagonal product operator, contracted
    as a transfer network), inverse-DFT'd on N = 0..(d-1)M.  Exact.
  * inelasticity eta = 1 - P(N0).
  * contiguous-cut entanglement profile S(k) and half-cut S_half, read directly
    from the MPS Schmidt spectra.
  * <N>, <N^2>, per-site occupations, max bond dimension.
"""
import numpy as np
import tenpy.linalg.np_conserved as npc
from tenpy.algorithms.mpo_evolution import ExpMPOEvolution
from tenpy.algorithms.tdvp import TwoSiteTDVPEngine


# ---------------------------------------------------------------- observables
def char_function(psi, thetas, d):
    """chi(theta) = <psi| exp(i theta N_tot) |psi> for an MPS, by contracting
    the diagonal product operator prod_i diag(e^{i theta n}) as a transfer
    network.  Returns a complex array over `thetas`."""
    L = psi.L
    out = np.empty(len(thetas), complex)
    Bs = [psi.get_B(i, form='B') for i in range(L)]
    for a, th in enumerate(thetas):
        phase = np.exp(1j * th * np.arange(d))
        E = None
        for i in range(L):
            B = Bs[i]
            Bp = B.copy().iscale_axis(phase, axis='p')       # O_i acting on ket
            if E is None:
                E = npc.tensordot(B.conj(), Bp,
                                  axes=[['vL*', 'p*'], ['vL', 'p']])  # -> vR*, vR
            else:
                E = npc.tensordot(E, Bp, axes=['vR', 'vL'])           # vR*, p, vR
                E = npc.tensordot(B.conj(), E,
                                  axes=[['vL*', 'p*'], ['vR*', 'p']]) # -> vR*, vR
        out[a] = complex(E.to_ndarray().reshape(()))
    return out


def number_distribution(psi, d, Nmax_support=None):
    """Exact P(N) of the total quanta from an MPS via the characteristic
    function and an inverse DFT on N = 0..(d-1)L."""
    L = psi.L
    Nmax = (d - 1) * L if Nmax_support is None else Nmax_support
    K = Nmax + 1
    thetas = 2 * np.pi * np.arange(K) / K
    chi = char_function(psi, thetas, d)
    # P(N) = (1/K) sum_k e^{-i theta_k N} chi(theta_k)
    pN = {}
    for N in range(K):
        val = np.real(np.mean(np.exp(-1j * thetas * N) * chi))
        if val > 1e-12:
            pN[N] = float(val)
    return pN


def mps_snapshot(psi, d, N0=1):
    """Bundle of the observables we track along a trajectory."""
    Nsite = psi.expectation_value("N")
    NN = float(np.sum(Nsite))
    S_profile = psi.entanglement_entropy()          # per bond (contiguous cut)
    L = psi.L
    pN = number_distribution(psi, d)
    return dict(
        chi=int(max(psi.chi)),
        Ntot=NN,
        eta=float(1.0 - pN.get(N0, 0.0)),
        pN=pN,
        S_half=float(S_profile[L // 2 - 1]) if L >= 2 else 0.0,
        S_profile=[float(x) for x in S_profile],
        S_max=float(np.max(S_profile)) if len(S_profile) else 0.0,
        occ=[float(x) for x in Nsite],
    )


# ---------------------------------------------------------------- evolver
def evolve(model, psi, t_final, dt=0.05, n_warmup=3, chi_max=64,
           svd_min=1e-10, snapshots=None, verbose=False, measure_N0=1,
           warmup_chi=6):
    """Hybrid ExpMPO-warmup + two-site TDVP real-time evolution.

    The ExpMPO warm-up only needs to nudge the state OFF the product manifold so
    that two-site TDVP -- whose cost is linear (not quadratic) in the MPO bond
    dimension, hence far cheaper at large multipole depth -- can take over and
    grow the bond itself.  The warm-up therefore runs at a deliberately tight
    bond cap `warmup_chi` (so its chi*chi_MPO intermediate stays small and cheap)
    for just `n_warmup` steps; TDVP then runs at the full `chi_max`.
    """
    import logging
    logging.getLogger("tenpy").setLevel(logging.WARNING)

    steps = int(round(t_final / dt))
    if snapshots is None:
        snapshots = [k * dt for k in range(0, steps + 1, 20)]
    snap_steps = sorted(set(int(round(t / dt)) for t in snapshots) | {0, steps})

    trunc = {"chi_max": chi_max, "svd_min": svd_min}
    warm_trunc = {"chi_max": warmup_chi, "svd_min": svd_min}
    rec = {"times": [], "chi": [], "Ntot": [], "eta": [], "S_half": [],
           "S_max": [], "S_profile": [], "pN": [], "occ": []}
    hit = False
    d = model.d

    def record(step):
        s = mps_snapshot(psi, d, N0=measure_N0)
        rec["times"].append(float(step * dt))
        for k in ("chi", "Ntot", "eta", "S_half", "S_max"):
            rec[k].append(s[k])
        rec["S_profile"].append(s["S_profile"])
        rec["pN"].append(s["pN"])
        rec["occ"].append(s["occ"])
        if verbose:
            print(f"    t={step*dt:6.2f}  chi={s['chi']:4d}  Ntot={s['Ntot']:.4f}"
                  f"  eta={s['eta']:.4f}  S_half={s['S_half']:.4f}", flush=True)
        return s

    record(0)
    cur = 0
    n_warmup = min(n_warmup, steps)

    # --- ExpMPO warm-up (seed entanglement off the product manifold) -------
    if n_warmup > 0:
        exp_opts = {"trunc_params": warm_trunc, "dt": dt, "N_steps": 1,
                    "order": 2, "approximation": "II",
                    "compression_method": "SVD",
                    "start_time": 0.0, "max_N_sites_per_ring": 100000}
        exp_eng = ExpMPOEvolution(psi, model, exp_opts)
        # advance to each snapshot boundary that lies within the warm-up window
        warm_bounds = [s for s in snap_steps if 0 < s <= n_warmup] + [n_warmup]
        for b in sorted(set(warm_bounds)):
            if b <= cur:
                continue
            exp_eng.options['N_steps'] = b - cur
            exp_eng.run()
            cur = b
            if max(psi.chi) >= chi_max:
                hit = True
            if b in snap_steps:
                record(b)

    # --- two-site TDVP for the bulk ----------------------------------------
    if cur < steps:
        tdvp_opts = {"trunc_params": trunc, "dt": dt, "N_steps": 1,
                     "max_dt": 100.0, "start_time": cur * dt,
                     "max_N_sites_per_ring": 100000}
        tdvp_eng = TwoSiteTDVPEngine(psi, model, tdvp_opts)
        bounds = [s for s in snap_steps if s > cur]
        for b in bounds:
            tdvp_eng.options['N_steps'] = b - cur
            tdvp_eng.run()
            cur = b
            if max(psi.chi) >= chi_max:
                hit = True
            record(b)

    rec["hit_chi_max"] = hit
    return rec
