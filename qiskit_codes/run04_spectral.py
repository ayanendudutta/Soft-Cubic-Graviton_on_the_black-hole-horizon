#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
 run04_spectral.py -- spectral statistics, done honestly
================================================================================
Implements the two-sided spectral analysis of the paper, whose new
methodological content is the symmetry-preserving total-N truncation:

 (1) Per-mode cutoff d (the object placed on qubits): Lz-sector- and parity-
     resolved <r> and spacing histograms, d = 4, 5, 6 refinement.  Finding
     (unchanged): GOE-grade repulsion <r> ~ 0.52 -- but fed by the cutoff-
     induced SU(2) breaking; NOT attributable to the physical dynamics.

 (2) Total-N cutoff (SU(2)-exact): H commutes with L^2 and Lz to machine
     precision, so the spectrum resolves into fixed-(L^2, Lz) blocks -- the
     PHYSICAL symmetry sectors -- within which <r> is computed, with Nmax
     convergence.  Statistics here are attributable to the (truncated-in-N)
     model without the boundary artefact.  Single l = 2 multiplet AND the
     multi-l (l = 2 (+) 3) register.

 (3) Spectral form factor of the pooled physical sectors (dip-ramp-plateau
     diagnostic), reported descriptively.

NO chaos claim for the physical near-horizon dynamics is made; the two-sided
statement is printed verbatim and recorded in the JSON summary.

Outputs: fig_D1_permode.pdf, fig_D2_su2blocks.pdf, fig_D3_sff.pdf,
summary_spectral.json.  Runtime: full ~20-60 min (d = 6 diagonalisations and
Nmax = 5 multi-l blocks); QUICK ~2 min.
================================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

from nhq import model as md
from nhq.util import Timer, save_json, save_npz, save_fig, QUICK, \
    INK, RUST, TEAL, GOLD, GREY, PALETTE

MODES = [(2, m) for m in range(-2, 3)]
G_UNIT = 1.0
G_REP = 12.0


def main():
    summary = {"two_sided_statement": [
        "(i) the truncated simulation Hamiltonian (per-mode Fock cutoff, the "
        "object actually placed on qubits) exhibits refinement-stable "
        "GOE-grade level repulsion in its exact-symmetry-resolved sectors;",
        "(ii) this repulsion is fed by the cutoff-induced breaking of the "
        "model's SU(2), and cannot be attributed to the untruncated "
        "near-horizon dynamics.  The SU(2)-preserving total-N truncation "
        "resolves the spectrum into physical fixed-(L^2, Lz) sectors; the "
        "statistics found there are reported as measured, with Nmax "
        "convergence and per-sector level counts, and are NOT extrapolated. "
        "No chaos claim is made for the physical dynamics; no MSS "
        "saturation is claimed."]}

    tab = md.cubic_vertex_table(MODES, g_eff=G_REP * G_UNIT)

    # ---------------- D1: per-mode cutoff, sector-resolved ----------------
    with Timer("D1: per-mode cutoff, Lz/parity-resolved <r>"):
        d_list = (4, 5) if QUICK else (4, 5, 6)
        pooled = {}
        for d in d_list:
            spacings, rs, ns = [], [], []
            lz_range = (0, 1, 2) if QUICK else (0, 1, 2, 3, 4)
            for lz in lz_range:
                fb = md.FockBasis(MODES, d=d, lz=lz)
                if fb.dim < 20:
                    continue
                H = fb.H(tab)
                if lz == 0:
                    P = fb.parity_perm()
                    blocks = md.simultaneous_blocks(H, [P])
                else:
                    blocks = [((lz,), H)]
                for labels, Hb in blocks:
                    E = np.linalg.eigvalsh(Hb)
                    r, n = md.r_ratio(E)
                    if not np.isnan(r):
                        rs.append(r)
                        ns.append(n)
                        spacings.append(md.unfolded_spacings(E))
            r_pool = float(np.average(rs, weights=ns))
            pooled[d] = (r_pool, int(np.sum(ns)),
                         np.concatenate(spacings))
            summary[f"D1_r_pooled_d{d}"] = r_pool
            summary[f"D1_levels_d{d}"] = int(np.sum(ns))
            print(f"  d={d}: pooled <r> = {r_pool:.4f} over {np.sum(ns)} "
                  f"spacings (GOE {md.R_GOE:.3f}, Poisson {md.R_POISSON:.3f})")

        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        d_show = d_list[-1]
        s = pooled[d_show][2]
        x = np.linspace(0, 4, 200)
        ax.hist(s, bins=30, density=True, color=GOLD, alpha=0.6,
                label=rf"$d={d_show}$, $\langle r\rangle="
                      rf"{pooled[d_show][0]:.3f}$")
        ax.plot(x, np.exp(-x), "--", color=GREY, label="Poisson")
        ax.plot(x, (np.pi / 2) * x * np.exp(-np.pi * x ** 2 / 4), "-",
                color=RUST, label="Wigner (GOE)")
        ax.set_xlabel("normalised spacing $s$")
        ax.set_ylabel("$P(s)$")
        ax.set_title("per-mode cutoff: $(L_z, P)$-resolved spacings\n"
                     "(truncation-fed repulsion; see two-sided statement)")
        ax.legend()
        save_fig(fig, "fig_D1_permode.pdf")

    # ---------------- D2: SU(2)-preserving truncation ---------------------
    def pooled_r(blocks, min_spacings=11):
        """Pool the INDIVIDUAL gap ratios r_i over all fixed-(L^2,Lz) sectors
        large enough to support statistics, and return (mean, standard error,
        N_r).  The per-sector means alone give no uncertainty; the pooled
        sample does, and it is the uncertainty quoted in the paper."""
        allr = []
        for _, Hb in blocks:
            E = np.sort(np.linalg.eigvalsh(Hb))
            n = len(E)
            E = E[int(0.1 * n):int(0.9 * n)]
            s = np.diff(E)
            s = s[s > 1e-10]
            if len(s) < min_spacings:
                continue
            allr.append(np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:]))
        if not allr:
            return None, None, 0
        r = np.concatenate(allr)
        return (float(r.mean()),
                float(r.std(ddof=1) / np.sqrt(len(r))), int(len(r)))

    with Timer("D2: total-N truncation, fixed-(L^2, Lz) blocks"):
        results_su2 = {}
        # single multiplet
        Nmax_list = (4, 5) if QUICK else (4, 5, 6, 7)
        for Nmax in Nmax_list:
            fbN = md.FockBasis(MODES, Nmax=Nmax, lz=0)
            HN = fbN.H(tab)
            L2 = fbN.L2()
            dev = np.abs(HN @ L2 - L2 @ HN).max()
            blocks = md.simultaneous_blocks(HN, [L2], tol=1e-6)
            info = []
            for labels, Hb in blocks:
                L2val = labels[0]
                L = 0.5 * (-1 + np.sqrt(1 + 4 * max(L2val, 0)))
                E = np.linalg.eigvalsh(Hb)
                r, n = md.r_ratio(E)
                info.append(dict(L=float(np.round(L, 3)), dim=Hb.shape[0],
                                 r=None if np.isnan(r) else float(r),
                                 n=int(n)))
            rp, rse, nr = pooled_r(blocks)
            results_su2[f"l2_Nmax{Nmax}"] = dict(commHL2=float(dev),
                                                 blocks=info,
                                                 dim=fbN.dim,
                                                 r_pooled=rp, r_SE=rse,
                                                 n_r=nr)
            if rp is None:
                print(f"  l=2 multiplet Nmax={Nmax}: dim={fbN.dim}, "
                      f"||[H,L2]||={dev:.1e}, {len(info)} sectors -- all "
                      "physical sectors too small for spacing statistics")
            big = [b for b in info if b["r"] is not None and b["n"] >= 10]
            if big:
                r_mean = float(np.average([b["r"] for b in big],
                                          weights=[b["n"] for b in big]))
                print(f"  l=2 multiplet Nmax={Nmax}: dim={fbN.dim}, "
                      f"||[H,L2]||={dev:.1e}, "
                      f"weighted <r> over (L^2,Lz=0) blocks = {r_mean:.3f}")
                results_su2[f"l2_Nmax{Nmax}"]["r_weighted"] = r_mean

        # multi-l register
        modes_ml = [(2, m) for m in range(-2, 3)] + \
                   [(3, m) for m in range(-3, 4)]
        tab_ml = md.cubic_vertex_table(modes_ml, g_eff=G_REP * G_UNIT)
        Nmax_ml = (3,) if QUICK else (3, 4, 5)
        for Nmax in Nmax_ml:
            fbm = md.FockBasis(modes_ml, Nmax=Nmax, lz=0)
            Hm = fbm.H(tab_ml)
            L2 = fbm.L2()
            dev = np.abs(Hm @ L2 - L2 @ Hm).max()
            blocks = md.simultaneous_blocks(Hm, [L2], tol=1e-6)
            info = []
            spac = []
            for labels, Hb in blocks:
                L2val = labels[0]
                L = 0.5 * (-1 + np.sqrt(1 + 4 * max(L2val, 0)))
                E = np.linalg.eigvalsh(Hb)
                r, n = md.r_ratio(E)
                info.append(dict(L=float(np.round(L, 3)), dim=Hb.shape[0],
                                 r=None if np.isnan(r) else float(r),
                                 n=int(n)))
                if n >= 10:
                    spac.append(md.unfolded_spacings(E))
            rp, rse, nr = pooled_r(blocks)
            results_su2[f"multil_Nmax{Nmax}"] = dict(commHL2=float(dev),
                                                     blocks=info,
                                                     dim=fbm.dim,
                                                     r_pooled=rp, r_SE=rse,
                                                     n_r=nr)
            if rp is not None:
                print(f"  multi-l Nmax={Nmax}: pooled <r> = {rp:.4f} "
                      f"+- {rse:.4f}  (N_r = {nr}); "
                      f"{(rp-md.R_POISSON)/rse:.1f} sigma above Poisson, "
                      f"{(md.R_GOE-rp)/rse:.1f} sigma below GOE")
            big = [b for b in info if b["r"] is not None and b["n"] >= 10]
            if big:
                r_mean = float(np.average([b["r"] for b in big],
                                          weights=[b["n"] for b in big]))
                results_su2[f"multil_Nmax{Nmax}"]["r_weighted"] = r_mean
                print(f"  multi-l Nmax={Nmax}: dim={fbm.dim}, "
                      f"||[H,L2]||={dev:.1e}, weighted <r> = {r_mean:.3f}")
            if Nmax == Nmax_ml[-1] and spac:
                s_all = np.concatenate(spac)
                fig, ax = plt.subplots(figsize=(4.2, 3.0))
                x = np.linspace(0, 4, 200)
                ax.hist(s_all, bins=30, density=True, color=TEAL, alpha=0.6,
                        label=rf"fixed-$(L^2,L_z)$ blocks, "
                              rf"$N_{{\max}}={Nmax}$")
                ax.plot(x, np.exp(-x), "--", color=GREY, label="Poisson")
                ax.plot(x, (np.pi / 2) * x * np.exp(-np.pi * x ** 2 / 4),
                        "-", color=RUST, label="Wigner (GOE)")
                ax.set_xlabel("normalised spacing $s$")
                ax.set_ylabel("$P(s)$")
                ttl = (r"SU(2)-exact truncation: physical $(L^2,L_z)$ "
                       "sectors, " + rf"$\ell=2\oplus3$, $N_{{\max}}={Nmax}$")
                if rp is not None:
                    ttl += "\n" + rf"pooled $\langle r\rangle={rp:.3f}"\
                           rf"({int(round(rse*1000))})$, $N_r={nr}$"
                ax.set_title(ttl, fontsize=9)
                ax.legend(fontsize=7)
                save_fig(fig, "fig_D2_su2blocks.pdf")
        summary["D2"] = results_su2
        save_json("summary_spectral_D2corrected.json", results_su2)

    # ---------------- D3: spectral form factor ----------------------------
    with Timer("D3: spectral form factor (descriptive)"):
        d = 4 if QUICK else 6
        fb = md.FockBasis(MODES, d=d, lz=1)
        E = np.linalg.eigvalsh(fb.H(tab))
        ts = np.logspace(-1, 3, 200)
        sff = md.spectral_form_factor(E, ts)
        fig, ax = plt.subplots(figsize=(4.0, 2.9))
        ax.loglog(ts, sff, color=INK)
        ax.set_xlabel("$t$")
        ax.set_ylabel(r"${\rm SFF}(t)/{\rm SFF}(0)$")
        ax.set_title(rf"SFF, per-mode cutoff $d={d}$, $L_z=1$ block "
                     "(descriptive)")
        save_fig(fig, "fig_D3_sff.pdf")
        save_npz("data_D3_sff.npz", ts=ts, sff=sff)

    save_json("summary_spectral.json", summary)
    print("\nrun04 complete.  The two-sided statement is recorded verbatim "
          "in summary_spectral.json.")


if __name__ == "__main__":
    main()
