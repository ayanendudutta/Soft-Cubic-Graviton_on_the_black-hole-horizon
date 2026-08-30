#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
 run02_multiplicity.py -- THE HEADLINE: the multiplicity distribution p_N of
 the first-principles cubic graviton vertex
================================================================================
Produces the paper's central figures and quotable numbers:

 (A) 5-mode l = 2 multiplet, per-mode cutoff d = 4 (with d = 5, 6 truncation
     overlays computed exactly inside the Lz = 0 block):
       fig_B1_pN_time.pdf     P(N, t) at the Fock-converged g~ = 12, exact
                              block evolution + Qiskit Trotter cross-check
                              + Aer shot estimates with binomial error bars
                              (the actual measurement protocol).
       fig_B2_onset.pdf       eta(g~) and S_mult(g~) at t = 20, d = 4/5/6
                              truncation overlays.
       fig_B3_pbarN.pdf       the infinite-time (diagonal-ensemble) p_bar_N at
                              g~ = 12 against three references: the ergodic
                              (Haar within the Lz block) prediction, the
                              thermal (Gibbs of H2, <N>-matched) distribution,
                              and the Poisson (<N>-matched) distribution --
                              with total-variation / KL distances printed.
                              THIS is the candidate quotable headline number.
       fig_B4_ablation.pdf    structure ablation (the non-tautology test):
                              the gravitationally fixed Gaunt table vs
                              uniform-|V| vs magnitude-shuffled tables (5
                              seeds): p_bar_N and eta(g~) differences quantify
                              how much of the signal is structure-specific.

 (B) multi-l register (l = 2 (+) l = 3, 12 modes) under the SYMMETRY-PRESERVING
     total-N truncation (N <= Nmax, exact SU(2)); Lz = 0 block:
       fig_B5_multil.pdf      p_bar_N and eta with Nmax = 3, 4, 5 convergence
                              overlays; leading-soft kernel K = 1 (stated
                              modeling input) plus a kernel-sensitivity
                              overlay, reported as a modeling systematic.

Scope guards (recorded verbatim in the JSON summary):
  * N counts occupation of the GGV longitudinal (non-radiative) sector.
  * g~ sweeps the magnitude of the gravitationally FIXED structure; the
    physical coupling gamma = kappa/R_S is Planck-suppressed -- mechanism
    structure, not absolute rates.
  * No chaos claim, no MSS saturation claim is made anywhere in this script.

Runtime: full run ~30-90 min on a laptop (the d = 6 sweep and the 10-qubit
Trotter evolution dominate).  NHQ_QUICK=1 finishes in ~3 min at reduced sizes.
================================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

from nhq import model as md
from nhq import circuits as qc
from nhq.util import Timer, save_json, save_npz, save_fig, pdict_to_arrays, \
    QUICK, INK, RUST, TEAL, GOLD, GREY, PALETTE

# ----------------------------------------------------------- configuration
MODES_L2 = [(2, m) for m in range(-2, 3)]
OCC0_L2 = (0, 0, 1, 0, 0)          # one graviton in (2, 0); Lz = 0, N0 = 1
G_REP = 12.0                       # Fock-converged representative coupling
T_SNAP = 20.0                      # snapshot time of the onset figures
G_SWEEP = np.linspace(0.0, 15.0, 6 if QUICK else 21)
D_LIST = (4, 5) if QUICK else (4, 5, 6)
TS = np.linspace(0.0, 40.0, 21 if QUICK else 81)

# reduced-coupling convention (the working window reproduces
# eta ~ 0.10-0.13 and P(N=2) ~ 0.10-0.12 at g~ = 12, t = 20 on the d = 4
# multiplet): g~ multiplies the UNIT-NORMALISED gravitationally fixed table
# (Gaunt x normalisation x combinatorial weights), i.e. g_eff = g~ * G_UNIT
# with G_UNIT = 1.  g~ replaces the Planck-suppressed physical magnitude
# gamma * c_g * K; the RELATIVE structure is never dialed.
G_UNIT = 1.0


def table_l2(gt, kernel=None):
    return md.cubic_vertex_table(MODES_L2, g_eff=gt * G_UNIT, kernel=kernel)


def psi0_in(basis, occ0):
    psi = np.zeros(basis.dim, complex)
    psi[basis.index[occ0]] = 1.0
    return psi


# ============================================================ part A helpers
def block_evolution_pN(d, gt, ts, occ0=OCC0_L2):
    """Exact P(N, t), eta(t), S_mult(t) inside the Lz block of occ0."""
    lz = int(np.dot([m for (_, m) in MODES_L2], occ0))
    fb = md.FockBasis(MODES_L2, d=d, lz=lz)
    H = fb.H(table_l2(gt))
    ev = md.BlockEvolver(H)
    psi0 = psi0_in(fb, occ0)
    N0 = sum(occ0)
    pNs, etas, smults = [], [], []
    for t in ts:
        psi = ev.psi(psi0, t)
        pN = md.multiplicity_distribution(psi, fb)
        pNs.append(pN)
        etas.append(1.0 - pN.get(N0, 0.0))
        smults.append(md.shannon(list(pN.values())))
    return fb, H, ev, psi0, pNs, np.array(etas), np.array(smults)


def onset_curves(d, gts, t_snap, occ0=OCC0_L2, table_fn=table_l2):
    """eta and S_mult at fixed t_snap versus coupling, exact in the block."""
    lz = int(np.dot([m for (_, m) in MODES_L2], occ0))
    fb = md.FockBasis(MODES_L2, d=d, lz=lz)
    psi0 = psi0_in(fb, occ0)
    N0 = sum(occ0)
    etas, smults = [], []
    for gt in gts:
        H = fb.H(table_fn(gt))
        psi = md.BlockEvolver(H).psi(psi0, t_snap)
        pN = md.multiplicity_distribution(psi, fb)
        etas.append(1.0 - pN.get(N0, 0.0))
        smults.append(md.shannon(list(pN.values())))
    return np.array(etas), np.array(smults)


# ============================================================ main
def main():
    summary = {"scope": [
        "N counts occupation of the GGV longitudinal (non-radiative) sector;"
        " the map to radiative Regge-Wheeler-Zerilli multiplicity is open.",
        "g~ sweeps the magnitude of the gravitationally fixed structure at"
        f" g_eff = g~ * {G_UNIT}; the physical coupling gamma = kappa/R_S is"
        " Planck-suppressed: mechanism structure, not absolute rates.",
        "No chaos claim and no MSS-saturation claim is made in this script.",
    ]}

    # ---------------- A1: P(N, t) with Qiskit / Aer cross-checks ----------
    with Timer("A1: P(N,t) exact block evolution, d = 4, g~ = 12"):
        fb4, H4, ev4, psi0, pNs, etas, smults = block_evolution_pN(4, G_REP, TS)
        Nvals = sorted({N for pN in pNs for N in pN})
        pN_t = {N: np.array([pN.get(N, 0.0) for pN in pNs]) for N in Nvals}
        summary["A1_eta_max_d4"] = float(etas.max())
        summary["A1_P2_max_d4"] = float(pN_t.get(2, np.zeros(1)).max())

    with Timer("A1b: Qiskit Gray Trotter cross-check (10 qubits, Aer)"):
        d = 4
        table = table_l2(G_REP)
        Hp, reg = qc.H_pauli(MODES_L2, table, d, encoding="gray")
        summary["A1b_pauli_terms"] = len(Hp)
        prep = qc.prep_fock_circuit(OCC0_L2, reg)
        # sequential Trotter segments with Aer statevector snapshots:
        # dt = 2.0 per segment, reps_per_segment sets the (genuine,
        # linearly-accumulating) Trotter error, as it would on hardware
        dt_seg = 2.0
        n_seg = int(TS[-1] / dt_seg)
        reps_seg = 2 if QUICK else 16
        states = qc.trotter_snapshot_states(Hp, prep, dt_seg, n_seg, reps_seg)
        ts_c = dt_seg * np.arange(1, n_seg + 1)
        fb_full = md.FockBasis(MODES_L2, d=d)     # full space for decoding
        perm = np.array([reg.basis_index(o) for o in fb_full.occs])
        pN_circ = [md.multiplicity_distribution(psi_q[perm], fb_full)
                   for psi_q in states]
        summary["A1b_reps_per_segment"] = reps_seg
        # Aer shot-based estimate at the snapshot time (measurement protocol)
        # r = 60 keeps the 2nd-order Trotter error at the 1e-3 level at t = 20
        # while bounding the transpiled circuit size for a laptop
        reps_snap = 10 if QUICK else 60
        circ = qc.trotter_circuit(Hp, T_SNAP, reps_snap)
        counts = qc.sample_counts(prep, circ, shots=4096 if QUICK else 16384,
                                  seed=11)
        pN_shots = qc.pN_from_counts(counts, reg)
        summary["A1b_shots_pN_t20"] = {str(N): [v[0], v[1]]
                                       for N, v in pN_shots.items()}

    with Timer("A1: figure fig_B1_pN_time.pdf"):
        # plot only N whose probability ever exceeds 1e-3 (readability)
        N_show = [N for N in Nvals if pN_t[N].max() > 1e-3]
        fig, ax = plt.subplots(figsize=(4.6, 3.1))
        for i, N in enumerate(N_show):
            c = PALETTE[i % len(PALETTE)]
            ax.plot(TS, pN_t[N], "-", color=c, label=rf"$P(N={N})$ exact")
        for i, N in enumerate(N_show):
            c = PALETTE[i % len(PALETTE)]
            vals = [pN.get(N, 0.0) for pN in pN_circ]
            ax.plot(ts_c, vals, "o", ms=4, mfc="none", color=c,
                    label="Qiskit Trotter" if i == 0 else None)
        for i, N in enumerate(sorted(pN_shots)):
            if N not in N_show:
                continue
            p, s = pN_shots[N]
            ax.errorbar([T_SNAP], [p], yerr=[2 * s], fmt="s", ms=5,
                        color=PALETTE[N_show.index(N) % len(PALETTE)],
                        capsize=3,
                        label="Aer shots ($\\pm2\\sigma$)" if i == 0 else None)
        ax.set_xlabel("$t$")
        ax.set_ylabel("multiplicity probability")
        ax.set_title(rf"graviton multiplicity, $\tilde g={G_REP:g}$ "
                     rf"($\ell=2$ multiplet, $d=4$)")
        ax.legend(ncol=2)
        save_fig(fig, "fig_B1_pN_time.pdf")
        save_npz("data_B1_pN_time.npz", ts=TS,
                 **{f"P{N}": pN_t[N] for N in Nvals},
                 ts_circ=ts_c,
                 pN_circ=np.array([[pN.get(N, 0.0) for N in Nvals]
                                   for pN in pN_circ]))

    # ---------------- A2: onset curves with truncation overlays -----------
    with Timer("A2: eta(g~), S_mult(g~) onset, d overlays"):
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
        onset = {}
        for d, c in zip(D_LIST, (INK, RUST, TEAL)):
            etas_g, smults_g = onset_curves(d, G_SWEEP, T_SNAP)
            onset[d] = (etas_g, smults_g)
            lab = f"$d={d}$" + ("" if d == 4 else " (truncation check)")
            axes[0].plot(G_SWEEP, etas_g, "o-" if d == 4 else "--", ms=3,
                         color=c, label=lab)
            axes[1].plot(G_SWEEP, smults_g, "o-" if d == 4 else "--", ms=3,
                         color=c, label=lab)
        summary["A2_trunc_dev_eta_d4_vs_dmax"] = float(np.max(np.abs(
            onset[D_LIST[0]][0] - onset[D_LIST[-1]][0])))
        axes[0].set_xlabel(r"reduced coupling $\tilde g$")
        axes[0].set_ylabel(r"inelasticity $\eta = 1 - P(N_0)$")
        axes[1].set_xlabel(r"reduced coupling $\tilde g$")
        axes[1].set_ylabel(r"$S_{\rm mult}$ (nats)")
        for a in axes:
            a.axvline(G_REP, color=GREY, lw=0.7, ls=":")
            a.legend()
        fig.suptitle(rf"onset at $t={T_SNAP:g}$ (initial single $(2,0)$ "
                     "graviton)", y=1.02)
        save_fig(fig, "fig_B2_onset.pdf")
        save_npz("data_B2_onset.npz", g=G_SWEEP,
                 **{f"eta_d{d}": onset[d][0] for d in D_LIST},
                 **{f"smult_d{d}": onset[d][1] for d in D_LIST})

    # ---------------- A3: diagonal ensemble vs references (HEADLINE) ------
    with Timer("A3: diagonal-ensemble p_bar_N vs ergodic/thermal/Poisson"):
        d_head = D_LIST[-1]
        lz = 0
        fbh = md.FockBasis(MODES_L2, d=d_head, lz=lz)
        Hh = fbh.H(table_l2(G_REP))
        psi0h = psi0_in(fbh, OCC0_L2)
        pbar = md.diagonal_ensemble_pN(Hh, psi0h, fbh)
        meanN = sum(N * p for N, p in pbar.items())
        pN_erg, _ = md.random_state_reference(fbh, want_page=False)
        pN_mc, E0, wE, n_mc = md.microcanonical_pN(Hh, psi0h, fbh)
        pN_mc2, _, wE2, n_mc2 = md.microcanonical_pN(Hh, psi0h, fbh,
                                                     width_sigmas=2.0)
        # effective dimension of the initial state over eigenstates:
        # d_eff = 1 / sum_n |c_n|^4.  The diagonal-ensemble-vs-microcanonical
        # comparison is meaningful evidence for ETH-grade equilibration ONLY
        # if d_eff >> 1 and the window holds many states; both quantities are
        # recorded, and belong alongside the TV/KL distances wherever those
        # are reported.
        Ew, Vw = np.linalg.eigh(Hh)
        cn2 = np.abs(Vw.conj().T @ psi0h) ** 2
        d_eff = float(1.0 / np.sum(cn2 ** 2))
        pN_th, beta = md.thermal_pN(fbh, meanN)
        support = sorted(pbar.keys())
        pN_po = md.poisson_pN(meanN, support)
        dists = {
            "TV_microcanonical": md.dist_TV(pbar, pN_mc),
            "TV_microcanonical_2sig": md.dist_TV(pbar, pN_mc2),
            "TV_ergodic_block": md.dist_TV(pbar, pN_erg),
            "TV_thermal": md.dist_TV(pbar, pN_th),
            "TV_poisson": md.dist_TV(pbar, pN_po),
            "KL_microcanonical": md.dist_KL(pbar, pN_mc),
            "KL_ergodic_block": md.dist_KL(pbar, pN_erg),
            "KL_thermal": md.dist_KL(pbar, pN_th),
            "KL_poisson": md.dist_KL(pbar, pN_po),
        }
        summary["A3_pbarN"] = pbar
        summary["A3_meanN"] = meanN
        summary["A3_beta_matched"] = beta
        summary["A3_d_eff_initial_state"] = d_eff
        summary["A3_microcanonical"] = dict(E0=E0, width=wE, n_states=n_mc,
                                            n_states_2sig=n_mc2,
                                            width_2sig=wE2, pN=pN_mc)
        summary["A3_distances"] = dists
        print(f"  d_eff of initial state = {d_eff:.1f}; MC window: "
              f"{n_mc} states (1 sigma), {n_mc2} (2 sigma)")
        print("  headline distances:", {k: f"{v:.4f}" for k, v in dists.items()})

        fig, ax = plt.subplots(figsize=(4.6, 3.1))
        Ns, pb = pdict_to_arrays(pbar)
        width = 0.16
        ax.bar(Ns - 2 * width, pb, width, color=INK,
               label=r"$\bar p_N$ (diag. ensemble, GR-fixed $H_3$)")
        _, pm = pdict_to_arrays(pN_mc, Ns[-1])
        ax.bar(Ns - width, pm, width, color="#5B4B8A",
               label=rf"microcanonical (ETH window, {n_mc} states)")
        _, pe = pdict_to_arrays(pN_erg, Ns[-1])
        ax.bar(Ns, pe, width, color=GOLD,
               label="ergodic (Haar in $L_z$ block)")
        _, pt = pdict_to_arrays(pN_th, Ns[-1])
        ax.bar(Ns + width, pt, width, color=TEAL,
               label=rf"thermal, $\beta={beta:.2f}$ matched")
        _, pp = pdict_to_arrays(pN_po, Ns[-1])
        ax.bar(Ns + 2 * width, pp, width, color=RUST,
               label=r"Poisson, $\langle N\rangle$ matched")
        ax.set_xlabel("total graviton number $N$")
        ax.set_ylabel(r"$\bar p_N$")
        ax.set_yscale("log")
        ax.set_title(rf"late-time multiplicity, $\tilde g={G_REP:g}$, "
                     rf"$d={d_head}$, $L_z=0$")
        ax.legend(fontsize=7,loc='lower left')
        save_fig(fig, "fig_B3_pbarN.pdf")

    # ---------------- A4: structure ablation (non-tautology test) ---------
    with Timer("A4: structure ablation (GR vs uniform vs shuffled)"):
        d_ab = 4
        gr_table = table_l2(G_REP)
        variants = {"GR-fixed": gr_table,
                    "uniform": md.ablate_uniform(gr_table)}
        n_sh = 2 if QUICK else 5
        for s in range(n_sh):
            variants[f"shuffled-{s}"] = md.ablate_shuffled(gr_table, seed=s)
        fb_ab = md.FockBasis(MODES_L2, d=d_ab, lz=0)
        psi0a = psi0_in(fb_ab, OCC0_L2)
        pbar_ab, eta_ab, tv_own_mc = {}, {}, {}
        for name, tab in variants.items():
            Hab = fb_ab.H(tab)
            pbar_ab[name] = md.diagonal_ensemble_pN(Hab, psi0a, fb_ab)
            # does THIS variant match its OWN microcanonical window?
            # (if all variants do, ETH-window agreement is generic to the
            # model class; if only some do, it is structure-dependent)
            pN_mc_v, _, _, _ = md.microcanonical_pN(Hab, psi0a, fb_ab)
            tv_own_mc[name] = md.dist_TV(pbar_ab[name], pN_mc_v)
            psi = md.BlockEvolver(Hab).psi(psi0a, T_SNAP)
            eta_ab[name] = 1.0 - md.multiplicity_distribution(
                psi, fb_ab).get(1, 0.0)
        summary["A4_eta_ablation"] = eta_ab
        summary["A4_TV_to_own_microcanonical"] = tv_own_mc
        print("  TV(pbar, own ETH window) per variant:",
              {k: f"{v:.4f}" for k, v in tv_own_mc.items()})
        summary["A4_TV_pbar_GR_vs_uniform"] = md.dist_TV(
            pbar_ab["GR-fixed"], pbar_ab["uniform"])
        tv_sh = [md.dist_TV(pbar_ab["GR-fixed"], pbar_ab[f"shuffled-{s}"])
                 for s in range(n_sh)]
        summary["A4_TV_pbar_GR_vs_shuffled_mean"] = float(np.mean(tv_sh))
        summary["A4_TV_pbar_GR_vs_shuffled_all"] = tv_sh

        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        for (name, pb), c in zip(pbar_ab.items(), PALETTE * 2):
            Ns, p = pdict_to_arrays(pb)
            ax.plot(Ns, p, "o-", ms=4, color=c,
                    label=name if "shuffled" not in name or name.endswith("0")
                    else None,
                    alpha=0.5 if "shuffled" in name else 1.0)
        ax.set_yscale("log")
        ax.set_xlabel("total graviton number $N$")
        ax.set_ylabel(r"$\bar p_N$")
        ax.set_title("structure ablation: is the signal gravitationally "
                     "specific?")
        ax.legend(fontsize=7)
        save_fig(fig, "fig_B4_ablation.pdf")

    # ---------------- B: multi-l register, symmetry-preserving cut --------
    with Timer("B: multi-l register (l = 2 (+) l = 3), total-N truncation"):
        modes_ml = [(2, m) for m in range(-2, 3)] + \
                   [(3, m) for m in range(-3, 4)]
        occ0_ml = tuple(1 if mo == (2, 0) else 0 for mo in modes_ml)
        Nmax_list = (3, 4) if QUICK else (3, 4, 5)
        curves = {}
        for Nmax in Nmax_list:
            fbm = md.FockBasis(modes_ml, Nmax=Nmax, lz=0)
            tab = md.cubic_vertex_table(modes_ml, g_eff=G_REP * G_UNIT)
            Hm = fbm.H(tab)
            # exact SU(2) of the truncation, recorded every run:
            L2m = fbm.L2()
            summary[f"B_commHL2_Nmax{Nmax}"] = float(
                np.abs(Hm @ L2m - L2m @ Hm).max())
            psi0m = psi0_in(fbm, occ0_ml)
            pbar = md.diagonal_ensemble_pN(Hm, psi0m, fbm)
            psi = md.BlockEvolver(Hm).psi(psi0m, T_SNAP)
            eta = 1.0 - md.multiplicity_distribution(psi, fbm).get(1, 0.0)
            curves[Nmax] = (pbar, eta, fbm.dim)
            summary[f"B_eta_Nmax{Nmax}"] = eta
            summary[f"B_dim_Nmax{Nmax}"] = fbm.dim
        # kernel sensitivity at the largest Nmax (modeling systematic)
        Nmax = Nmax_list[-1]
        fbm = md.FockBasis(modes_ml, Nmax=Nmax, lz=0)

        def K_soft(mi, mj, mk):
            wi, wj, wk = (md.omega_l(l) for (l, _) in (mi, mj, mk))
            return (3.0 / (wi + wj + wk)) ** 0.5   # illustrative soft kernel

        tabK = md.cubic_vertex_table(modes_ml, g_eff=G_REP * G_UNIT,
                                     kernel=K_soft)
        HmK = fbm.H(tabK)
        psi0m = psi0_in(fbm, occ0_ml)
        pbarK = md.diagonal_ensemble_pN(HmK, psi0m, fbm)
        summary["B_TV_kernel_sensitivity"] = md.dist_TV(curves[Nmax][0], pbarK)

        fig, ax = plt.subplots(figsize=(4.4, 3.0))
        for (Nmax, (pb, eta, dim)), c in zip(curves.items(), PALETTE):
            Ns, p = pdict_to_arrays(pb)
            ax.plot(Ns, p, "o-", ms=4, color=c,
                    label=rf"$N_{{\max}}={Nmax}$ (dim {dim}), "
                          rf"$\eta={eta:.3f}$")
        Ns, p = pdict_to_arrays(pbarK)
        ax.plot(Ns, p, "s--", ms=4, color=GREY, label="soft-kernel overlay")
        ax.set_yscale("log")
        ax.set_xlabel("total graviton number $N$")
        ax.set_ylabel(r"$\bar p_N$")
        ax.set_title(r"multi-$\ell$ register ($\ell=2\oplus3$), "
                     r"SU(2)-preserving $N\leq N_{\max}$ truncation, $L_z=0$")
        ax.legend(fontsize=7)
        save_fig(fig, "fig_B5_multil.pdf")

    save_json("summary_multiplicity.json", summary)
    print("\nrun02 complete. The headline quantities are in "
          "summary_multiplicity.json (A3_distances, A4_*, B_*).")


if __name__ == "__main__":
    main()
