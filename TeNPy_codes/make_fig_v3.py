#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Revised MPS figures for the paper (two-tier framing + d-cutoff scaling).
  figH2  main summary   : (a) eta vs M, (b) S_max & N vs M,
                          (c) CUTOFF convergence eta vs d, (d) truncation vs reg.
  figH3  appendix merge : (a) S_max(t) dynamics, (b) multiplicity tower [H3+H4].
Reads out_mps/climb_l{L}_d{D}.{json,npz} (mode climb at d=4 plus the d-scaling)."""
import os, glob, json, re
import numpy as np
from nhq_tenpy.figstyle import plt, PALETTE

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_mps")


def load():
    runs = {}
    for f in sorted(glob.glob(os.path.join(OUT, "climb_l*_d*.json"))):
        m = re.search(r"l(\d+)_d(\d+)", f); L, D = int(m.group(1)), int(m.group(2))
        s = json.load(open(f)); s["_npz"] = f.replace(".json", ".npz")
        runs[(L, D)] = s
    return runs


def qpm(d):  # qubits/mode, Gray encoding
    return int(np.ceil(np.log2(d)))


def fig_summary(runs):
    d4 = sorted([k for k in runs if k[1] == 4])
    M = np.array([runs[k]["M"] for k in d4], float)
    eta = np.array([runs[k]["eta_late_mean"] for k in d4])
    Smx = np.array([runs[k]["S_max_late_mean"] for k in d4])
    Nt = np.array([runs[k]["Ntot_late_mean"] for k in d4])
    tre = np.array([runs[k]["trunc_err_final"] for k in d4])
    Ls = [k[0] for k in d4]

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4))

    a = axs[0, 0]
    a.plot(M, eta, "o-", color=PALETTE[1], label=r"$d=4$ (mode climb)")
    # overlay d=8 points to show cutoff-robustness of the trend
    d8 = sorted([k for k in runs if k[1] == 8])
    a.plot([runs[k]["M"] for k in d8], [runs[k]["eta_late_mean"] for k in d8],
           "D", ms=5, mfc="none", color=PALETTE[4], label=r"$d=8$ (check)")
    a.set_xlabel(r"number of modes $M$")
    a.set_ylabel(r"late-time inelasticity $\overline{\eta}$")
    a.set_title(r"(a) inelasticity saturates ($\overline{\eta}_\infty\!\approx\!0.16$)")
    a.legend(fontsize=7.5)

    a = axs[0, 1]
    a.plot(M, Smx, "s-", color=PALETTE[2], label=r"$\overline{S}_{\max}$")
    a.set_xlabel(r"number of modes $M$")
    a.set_ylabel(r"peak entanglement (nats)", color=PALETTE[2])
    a.tick_params(axis="y", labelcolor=PALETTE[2])
    a.set_title("(b) entanglement saturates (area-law)")
    a2 = a.twinx()
    a2.plot(M, Nt, "^--", color=PALETTE[3])
    a2.set_ylabel(r"mean number $\overline{N}$", color=PALETTE[3])
    a2.tick_params(axis="y", labelcolor=PALETTE[3]); a2.grid(False)

    # (c) CUTOFF convergence: eta vs d for each l that has multiple d
    a = axs[1, 0]
    for i, L in enumerate((2, 3, 4, 5)):
        ds = sorted([k[1] for k in runs if k[0] == L])
        if len(ds) < 2:
            continue
        etad = [runs[(L, dd)]["eta_late_mean"] for dd in ds]
        a.plot(ds, etad, "o-", color=PALETTE[i % len(PALETTE)],
               label=rf"$\ell_{{\max}}={L}$")
    a.set_xscale("log", base=2)
    a.set_xticks([4, 8, 16, 32]); a.set_xticklabels([4, 8, 16, 32])
    a.set_xlabel(r"per-mode occupation cutoff $d$")
    a.set_ylabel(r"late-time inelasticity $\overline{\eta}$")
    a.set_title(r"(c) cutoff convergence: $\overline{\eta}(d)$ flat")
    a.legend(fontsize=7.5, ncol=2)

    a = axs[1, 1]
    a.semilogy(M, np.clip(tre, 1e-16, None), "o-", color=PALETTE[5],
               label=r"$d=4$")
    a.semilogy([runs[k]["M"] for k in d8],
               [runs[k]["trunc_err_final"] for k in d8], "D", ms=5, mfc="none",
               color=PALETTE[4], label=r"$d=8$")
    a.axhline(1e-3, color=PALETTE[1], lw=1.0, ls=(0, (4, 3)))
    a.set_xlabel(r"number of modes $M$")
    a.set_ylabel(r"final truncation error")
    a.set_title(r"(d) truncation error $<10^{-3}$")
    a.legend(fontsize=7.5)

    fig.suptitle(r"MPS climb: rigidity persists to $M=60$ (120 qubits) and is "
                 r"cutoff-converged", y=1.005, fontsize=10.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_H2_summary.{ext}"),
                    dpi=(150 if ext == "png" else None))
    plt.close(fig)


def fig_appendix(runs):
    d4 = sorted([k for k in runs if k[1] == 4])
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for i, k in enumerate(d4):
        z = np.load(runs[k]["_npz"], allow_pickle=True)
        L, M = k[0], runs[k]["M"]
        axs[0].plot(z["times"], z["S_max"], color=PALETTE[i % len(PALETTE)],
                    lw=1.4, label=rf"$\ell_{{\max}}={L}$ ($M={M}$)")
        axs[1].bar(z["pbar_N"] + (i - len(d4) / 2) * 0.12, z["pbar_val"],
                   width=0.12, color=PALETTE[i % len(PALETTE)], alpha=0.9,
                   edgecolor="white", linewidth=0.3)
    axs[0].set_xlabel(r"time $t$  (units $R_S/c$)")
    axs[0].set_ylabel(r"peak entanglement $S_{\max}(t)$ (nats)")
    axs[0].set_title("(a) peak-entanglement dynamics")
    axs[0].legend(fontsize=6.5, ncol=2)
    axs[1].set_xlabel(r"total graviton number $N$")
    axs[1].set_ylabel(r"late-time-averaged $\overline{P}(N)$")
    axs[1].set_yscale("log"); axs[1].set_xticks(range(0, 9)); axs[1].set_xlim(-0.5, 8.5)
    axs[1].set_title("(b) multiplicity tower")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_H3_tower.{ext}"),
                    dpi=(150 if ext == "png" else None))
    plt.close(fig)


if __name__ == "__main__":
    runs = load()
    fig_summary(runs); fig_appendix(runs)
    print("wrote fig_H2_summary and fig_H3_tower (pdf+png) to", OUT)
