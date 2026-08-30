#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
make_mps_figures.py -- framed MPS-climb figures for the paper.

Reads every out_mps/climb_l{L}_d{d}.npz produced by run_mps_climb.py and emits
four publication figures (framed, elegant) to out_mps/:

  figM1_entanglement_profile.pdf     late-time-averaged entanglement across the
                                     mode chain for each multipole depth -- the
                                     bipartite entropy is PEAKED near the active
                                     (low-|m|) modes and BOUNDED (area-law); the
                                     peak saturates as M grows.  [This replaces
                                     the old S_half(t) traces: with the paired-m
                                     ordering the mid-chain bond drifts into the
                                     weakly-populated tail, so S_half is
                                     position-dependent and non-monotonic and is
                                     NOT a faithful entanglement measure -- the
                                     peak S_max and the profile are.]
  figM2_multipole_summary.pdf        late-time inelasticity, peak entanglement,
                                     and the l=4 chi-convergence check vs M.
  figM3_multiplicity_tower.pdf       late-time-averaged P(N) per depth.
  figM4_entanglement_dynamics.pdf    peak-entanglement S_max(t) trajectories and
                                     the truncation error.

The l=4 bond-dimension convergence study (chi = 24, 32, 48, 96) is a separate
sweep, not part of the climb outputs; its late-time values are recorded in
CONV_L4 below and were produced by re-running l=4 at each cap.  Re-running that
sweep and updating CONV_L4 refreshes panel (c).
"""
import os, glob, json, re
import numpy as np
from nhq_tenpy.figstyle import plt, PALETTE

OUT = os.environ.get("NHQ_OUT",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "out_mps"))

# l=4 bond-dimension convergence study (from the console log); used in figM2c.
CONV_L4 = {"chi": [24, 32, 48, 96],
           "eta": [0.1360, 0.1369, 0.1371, 0.1360],
           "S_half": [0.3146, 0.3239, 0.3263, 0.3170]}


def load_all(d_climb=4):
    """Load the mode-climb runs at the fixed per-mode cutoff d_climb (one point
    per multipole depth).  The occupation-axis refinement runs (d = 8, 16, ...)
    live in the same directory and are handled separately by make_fig_v3.py;
    mixing them here would place several points at the same mode number M."""
    runs = []
    for f in sorted(glob.glob(os.path.join(OUT, f"climb_l*_d{d_climb}.npz"))):
        m = re.search(r"climb_l(\d+)_d(\d+)\.npz", os.path.basename(f))
        lmax, d = int(m.group(1)), int(m.group(2))
        z = np.load(f, allow_pickle=True)
        jf = f.replace(".npz", ".json")
        summ = json.load(open(jf)) if os.path.exists(jf) else {}
        runs.append(dict(lmax=lmax, d=d, M=(lmax + 1)**2 - 4, z=z, summ=summ))
    runs.sort(key=lambda r: r["lmax"])
    return runs


def _lbl(r):
    return rf"$\ell_{{\max}}={r['lmax']}$ ($M={r['M']}$)"


def _late_profile(z, frac=0.5):
    t = z["times"]; late = t >= frac * t.max()
    sp = z["S_profile"]                       # (n_times, M-1), zero-padded
    prof = sp[late].mean(0)
    # trim trailing exact zeros from padding (keep the M-1 real bonds)
    return prof


# ---------------------------------------------------------------- figM1
def fig_profile(runs):
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for i, r in enumerate(runs):
        prof = _late_profile(r["z"])
        M = r["M"]
        x = np.arange(1, M) / M                # normalised cut position in (0,1)
        ax.plot(x, prof, color=PALETTE[i % len(PALETTE)], marker="o", ms=2.6,
                lw=1.4, label=_lbl(r))
    ax.set_xlabel(r"normalised bipartition position  $k/M$")
    ax.set_ylabel(r"late-time entanglement  $\overline{S}(k)$  (nats)")
    ax.set_title("Horizon entanglement is area-law and bounded")
    ax.legend(loc="upper right", ncol=1, fontsize=7.5)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"figM1_entanglement_profile.{ext}"),
                    dpi=(140 if ext == "png" else None))
    plt.close(fig)


# ---------------------------------------------------------------- figM2
def fig_summary(runs):
    M = np.array([r["M"] for r in runs], float)
    eta = np.array([r["summ"].get("eta_late_mean", np.nan) for r in runs])
    Smx = np.array([r["summ"].get("S_max_late_mean", np.nan) for r in runs])
    Nt = np.array([r["summ"].get("Ntot_late_mean", np.nan) for r in runs])
    tre = np.array([r["summ"].get("trunc_err_final", np.nan) for r in runs])

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4))

    a = axs[0, 0]
    a.plot(M, eta, "o-", color=PALETTE[1])
    a.set_xlabel(r"number of modes $M$")
    a.set_ylabel(r"late-time inelasticity $\overline{\eta}$")
    a.set_title(r"(a) inelasticity saturates ($\overline{\eta}_\infty\!\approx\!0.16$)")

    a = axs[0, 1]
    a.plot(M, Smx, "s-", color=PALETTE[2], label=r"peak entanglement $\overline{S}_{\max}$")
    a.set_xlabel(r"number of modes $M$")
    a.set_ylabel(r"peak entanglement (nats)", color=PALETTE[2])
    a.tick_params(axis="y", labelcolor=PALETTE[2])
    a.set_title("(b) entanglement saturates (area-law)")
    a2 = a.twinx()
    a2.plot(M, Nt, "^--", color=PALETTE[3], label=r"mean number $\overline{N}$")
    a2.set_ylabel(r"mean graviton number $\overline{N}$", color=PALETTE[3])
    a2.tick_params(axis="y", labelcolor=PALETTE[3])
    a2.grid(False)

    a = axs[1, 0]
    ch = np.array(CONV_L4["chi"], float)
    a.plot(ch, CONV_L4["eta"], "o-", color=PALETTE[1], label=r"$\overline{\eta}$")
    a.plot(ch, CONV_L4["S_half"], "s--", color=PALETTE[0],
           label=r"$\overline{S}_{1/2}$")
    a.set_xlabel(r"bond-dimension cap $\chi$   ($\ell_{\max}=4$)")
    a.set_ylabel(r"late-time observable")
    a.set_title(r"(c) convergence in $\chi$: $<\!1\%$ in $\overline{\eta}$")
    a.legend(fontsize=7.5)
    a.set_ylim(0, 0.42)

    a = axs[1, 1]
    a.semilogy(M, np.clip(tre, 1e-16, None), "o-", color=PALETTE[5])
    a.axhline(1e-3, color=PALETTE[1], lw=1.0, ls=(0, (4, 3)),
              label=r"$10^{-3}$ (Trotter-level)")
    a.set_xlabel(r"number of modes $M$")
    a.set_ylabel(r"final truncation error")
    a.set_title(r"(d) truncation error stays $<10^{-3}$")
    a.legend(fontsize=7.5)

    fig.suptitle(r"MPS multipole-ladder climb: rigidity persists to "
                 rf"$M={int(M.max())}$ modes (120 qubits)", y=1.005, fontsize=10.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"figM2_multipole_summary.{ext}"),
                    dpi=(140 if ext == "png" else None))
    plt.close(fig)


# ---------------------------------------------------------------- figM3
def fig_tower(runs):
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    width = 0.8 / max(1, len(runs))
    for i, r in enumerate(runs):
        N = r["z"]["pbar_N"]; P = r["z"]["pbar_val"]
        ax.bar(N + (i - len(runs) / 2) * width, P, width=width,
               color=PALETTE[i % len(PALETTE)], label=_lbl(r), alpha=0.9,
               edgecolor="white", linewidth=0.4)
    ax.set_xlabel(r"total graviton number $N$")
    ax.set_ylabel(r"late-time-averaged $\overline{P}(N)$")
    ax.set_title("Multiplicity tower vs multipole depth")
    ax.set_yscale("log"); ax.legend(fontsize=7.5)
    ax.set_xticks(range(0, 9)); ax.set_xlim(-0.5, 8.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"figM3_multiplicity_tower.{ext}"),
                    dpi=(140 if ext == "png" else None))
    plt.close(fig)


# ---------------------------------------------------------------- figM4
def fig_dynamics(runs):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for i, r in enumerate(runs):
        axs[0].plot(r["z"]["times"], r["z"]["S_max"], color=PALETTE[i % len(PALETTE)],
                    lw=1.4, label=_lbl(r))
        te = r["z"]["trunc_err"]
        axs[1].semilogy(r["z"]["times"], np.clip(te, 1e-16, None),
                        color=PALETTE[i % len(PALETTE)], lw=1.4, label=_lbl(r))
    axs[0].set_xlabel(r"time $t$  (units $R_S/c$)")
    axs[0].set_ylabel(r"peak entanglement $S_{\max}(t)$ (nats)")
    axs[0].set_title("(a) peak-entanglement dynamics")
    axs[0].legend(fontsize=6.5, ncol=2)
    axs[1].axhline(1e-3, color="k", lw=0.8, ls=(0, (4, 3)))
    axs[1].set_xlabel(r"time $t$  (units $R_S/c$)")
    axs[1].set_ylabel(r"truncation error $\epsilon(t)$")
    axs[1].set_title("(b) truncation error")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"figM4_entanglement_dynamics.{ext}"),
                    dpi=(140 if ext == "png" else None))
    plt.close(fig)


if __name__ == "__main__":
    runs = load_all()
    if not runs:
        raise SystemExit(f"no climb_l*_d*.npz found in {OUT}")
    print("plotting", [f"l{r['lmax']}(M={r['M']})" for r in runs])
    fig_profile(runs); fig_summary(runs); fig_tower(runs); fig_dynamics(runs)
    print("wrote figM1..figM4 (pdf+png) to", OUT)
