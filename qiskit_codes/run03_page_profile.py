#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
 run03_page_profile.py -- the entanglement (Page-like) profile of the produced
 state across mode bipartitions
================================================================================
The second headline figure: the mode-bipartition entanglement profile
S_ent(k) versus subsystem size k, of the state produced by the gravitationally
fixed H3, compared against the ergodic (Haar-random within the accessible
Lz block) reference profile.

SCOPE (recorded verbatim in the JSON summary):
  This is a statement about the entanglement STRUCTURE of the longitudinal
  near-horizon sector on a static Schwarzschild background.  It is NOT a Page
  curve of black-hole evaporation: the fixed classical background captures
  multiplicity and entanglement structure but not secular mass loss.  The
  phrase "Page-like profile" is meant strictly in the sense of the
  subsystem-size dependence of S_ent compared with the random-state
  (Page/ergodic) reference.

Outputs:
  fig_C1_page_time.pdf   S_ent(k) at several times, g~ = 12, l = 2 multiplet,
                         d = 4, with the ergodic reference band.
  fig_C2_page_g.pdf      late-time S_ent(k) at several couplings.
  fig_C3_sent_smult.pdf  the correlated growth of S_ent, S_mult and eta,
                         all on one panel.
  fig_C4_page_multil.pdf multi-l register profile under the SU(2)-preserving
                         total-N truncation, Nmax convergence overlay.

Quotable numbers written to summary_page.json: the maximum profile deficit
max_k [S_Haar(k) - S_H3(k)], the time-averaged profile, and its distance to
the ergodic reference.

Runtime: ~10-30 min full, ~2 min QUICK.
================================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

from nhq import model as md
from nhq.util import Timer, save_json, save_npz, save_fig, QUICK, \
    INK, RUST, TEAL, GOLD, GREY, PALETTE

MODES = [(2, m) for m in range(-2, 3)]
OCC0 = (0, 0, 1, 0, 0)
G_UNIT = 1.0
G_REP = 12.0


def main():
    summary = {"scope": "Page-like = subsystem-size profile vs random-state "
                        "reference on static Schwarzschild; NOT evaporation."}
    d = 4
    fb = md.FockBasis(MODES, d=d, lz=0)
    psi0 = np.zeros(fb.dim, complex)
    psi0[fb.index[OCC0]] = 1.0

    # ---------------- C1: profile vs time --------------------------------
    with Timer("C1: S_ent(k) vs time + ergodic reference"):
        H = fb.H(md.cubic_vertex_table(MODES, g_eff=G_REP * G_UNIT))
        ev = md.BlockEvolver(H)
        times = (5.0, 20.0) if QUICK else (5.0, 10.0, 20.0, 40.0)
        prof_t = {}
        for t in times:
            psi = ev.psi(psi0, t)
            prof_t[t] = md.page_profile(psi, fb)
        pN_ref, prof_ref = md.random_state_reference(
            fb, n_samples=8 if QUICK else 32, want_page=True)
        # time-averaged profile from late-time samples
        ts_avg = np.linspace(30, 120, 6 if QUICK else 24)
        acc = {}
        for t in ts_avg:
            pr = md.page_profile(ev.psi(psi0, t), fb)
            for k, (m, _) in pr.items():
                acc.setdefault(k, []).append(m)
        prof_avg = {k: float(np.mean(v)) for k, v in acc.items()}
        deficit = {k: prof_ref[k][0] - prof_avg[k] for k in prof_avg}
        summary["C1_profile_timeavg"] = prof_avg
        summary["C1_profile_ergodic"] = {k: v[0] for k, v in prof_ref.items()}
        summary["C1_max_deficit"] = float(max(deficit.values()))
        summary["C1_deficit_by_k"] = deficit

        fig, ax = plt.subplots(figsize=(4.4, 3.1))
        ks = sorted(prof_ref)
        ref_m = np.array([prof_ref[k][0] for k in ks])
        ref_s = np.array([prof_ref[k][1] for k in ks])
        ax.fill_between(ks, ref_m - 2 * ref_s, ref_m + 2 * ref_s,
                        color=GOLD, alpha=0.25,
                        label="ergodic (Haar in block) $\\pm2\\sigma$")
        ax.plot(ks, ref_m, "-", color=GOLD)
        for t, c in zip(times, PALETTE):
            m = [prof_t[t][k][0] for k in ks]
            s = [prof_t[t][k][1] for k in ks]
            ax.errorbar(ks, m, yerr=s, fmt="o-", ms=4, color=c,
                        label=rf"$t={t:g}$", capsize=2)
        ax.plot(ks, [prof_avg[k] for k in ks], "s--", color=GREY,
                label="late-time average")
        ax.set_xlabel("subsystem size $k$ (modes)")
        ax.set_ylabel(r"$\overline{S_{\rm ent}}(k)$ (nats)")
        ax.set_title(rf"mode-bipartition entanglement profile, "
                     rf"$\tilde g={G_REP:g}$, $d={d}$, $L_z=0$")
        ax.legend(fontsize=7)
        save_fig(fig, "fig_C1_page_time.pdf")
        save_npz("data_C1_page.npz", ks=np.array(ks), ref_m=ref_m, ref_s=ref_s,
                 prof_avg=np.array([prof_avg[k] for k in ks]))

    # ---------------- C2: profile vs coupling ----------------------------
    with Timer("C2: S_ent(k) vs coupling"):
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        gts = (6.0, 12.0) if QUICK else (3.0, 6.0, 12.0, 15.0)
        for gt, c in zip(gts, PALETTE):
            Hg = fb.H(md.cubic_vertex_table(MODES, g_eff=gt * G_UNIT))
            psi = md.BlockEvolver(Hg).psi(psi0, 20.0)
            pr = md.page_profile(psi, fb)
            ks = sorted(pr)
            ax.plot(ks, [pr[k][0] for k in ks], "o-", ms=4, color=c,
                    label=rf"$\tilde g={gt:g}$")
        ax.plot(ks, ref_m, "--", color=GOLD, label="ergodic ref")
        ax.set_xlabel("subsystem size $k$")
        ax.set_ylabel(r"$\overline{S_{\rm ent}}(k)$ (nats)")
        ax.set_title("profile vs coupling ($t=20$)")
        ax.legend(fontsize=7)
        save_fig(fig, "fig_C2_page_g.pdf")

    # ---------------- C3: correlated growth ------------------------------
    with Timer("C3: correlated growth of eta, S_mult, S_ent"):
        ts = np.linspace(0, 40, 21 if QUICK else 81)
        eta_t, smult_t, sent_t = [], [], []
        for t in ts:
            psi = ev.psi(psi0, t)
            pN = md.multiplicity_distribution(psi, fb)
            eta_t.append(1 - pN.get(1, 0.0))
            smult_t.append(md.shannon(list(pN.values())))
            sent_t.append(md.entanglement_entropy(psi, fb, [2]))  # (2,0)|rest
        fig, ax = plt.subplots(figsize=(4.4, 3.0))
        ax.plot(ts, eta_t, "-", color=RUST, label=r"$\eta$")
        ax.plot(ts, smult_t, "-", color=INK, label=r"$S_{\rm mult}$")
        ax.plot(ts, sent_t, "-", color=TEAL,
                label=r"$S_{\rm ent}\,[(2,0)\,|\,{\rm rest}]$")
        ax.set_xlabel("$t$")
        ax.set_ylabel("nats / probability")
        ax.set_title("correlated onset at the gravitationally fixed structure")
        ax.legend()
        save_fig(fig, "fig_C3_sent_smult.pdf")
        save_npz("data_C3_growth.npz", ts=ts, eta=np.array(eta_t),
                 smult=np.array(smult_t), sent=np.array(sent_t))
        summary["C3_final"] = dict(eta=float(eta_t[-1]),
                                   smult=float(smult_t[-1]),
                                   sent=float(sent_t[-1]))

    # ---------------- C4: multi-l profile ---------------------------------
    with Timer("C4: multi-l profile, SU(2)-preserving truncation"):
        modes_ml = [(2, m) for m in range(-2, 3)] + \
                   [(3, m) for m in range(-3, 4)]
        occ0_ml = tuple(1 if mo == (2, 0) else 0 for mo in modes_ml)
        fig, ax = plt.subplots(figsize=(4.6, 3.1))
        Nmax_list = (3,) if QUICK else (3, 4, 5)
        for Nmax, c in zip(Nmax_list, PALETTE):
            fbm = md.FockBasis(modes_ml, Nmax=Nmax, lz=0)
            tab = md.cubic_vertex_table(modes_ml, g_eff=G_REP * G_UNIT)
            Hm = fbm.H(tab)
            psi0m = np.zeros(fbm.dim, complex)
            psi0m[fbm.index[occ0_ml]] = 1.0
            psi = md.BlockEvolver(Hm).psi(psi0m, 20.0)
            pr = md.page_profile(psi, fbm,
                                 max_subsets=30 if QUICK else 120)
            ks = sorted(pr)
            ax.errorbar(ks, [pr[k][0] for k in ks],
                        yerr=[pr[k][1] for k in ks], fmt="o-", ms=3,
                        color=c, capsize=2, label=rf"$N_{{\max}}={Nmax}$ "
                        rf"(dim {fbm.dim})")
            summary[f"C4_profile_Nmax{Nmax}"] = {k: pr[k][0] for k in ks}
        ax.set_xlabel("subsystem size $k$ (of 12 modes)")
        ax.set_ylabel(r"$\overline{S_{\rm ent}}(k)$ (nats)")
        ax.set_title(r"multi-$\ell$ profile ($\ell=2\oplus3$, $t=20$, "
                     r"$L_z=0$)")
        ax.legend(fontsize=7)
        save_fig(fig, "fig_C4_page_multil.pdf")

    save_json("summary_page.json", summary)
    print("\nrun03 complete.")


if __name__ == "__main__":
    main()
