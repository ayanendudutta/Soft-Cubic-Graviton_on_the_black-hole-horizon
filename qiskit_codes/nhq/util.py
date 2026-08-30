#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities for the run scripts: output paths, JSON/NPZ saving,
timestamped progress, figure style, and the QUICK smoke-test mode
(the environment variable NHQ_QUICK=1 selects reduced sizes, at which every
script completes in minutes; full runs use the parameters of the paper)."""
import json
import os
import time
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

QUICK = os.environ.get("NHQ_QUICK", "0") == "1"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "out")
os.makedirs(OUT, exist_ok=True)

INK, RUST, TEAL, GOLD, GREY = "#13294B", "#A23B2E", "#1B7A7A", "#C7902C", "#6B6B6B"
PALETTE = [INK, RUST, TEAL, GOLD, "#5B4B8A", "#2E7D32"]

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03, "savefig.facecolor": "white",
    "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
    "legend.fontsize": 8,
    # --- framed, elegant look: full box on every axes ---------------------
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.spines.left": True, "axes.spines.bottom": True,
    "axes.linewidth": 0.8, "axes.edgecolor": "#2B2B2B",
    "axes.axisbelow": True,
    # ticks inward on all four sides, with minor ticks
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.size": 4.0, "ytick.major.size": 4.0,
    "xtick.minor.size": 2.2, "ytick.minor.size": 2.2,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
    # a very light grid reads as tidy without cluttering
    "axes.grid": True, "grid.color": "#B8B8B8", "grid.alpha": 0.35,
    "grid.linewidth": 0.5, "grid.linestyle": (0, (3, 3)),
    # legend styling: framed, subtle
    "legend.frameon": True, "legend.framealpha": 0.9,
    "legend.edgecolor": "#CCCCCC", "legend.fancybox": False,
    "legend.borderpad": 0.4, "legend.handlelength": 1.6,
    "lines.linewidth": 1.6, "lines.markersize": 4.5,
    "axes.prop_cycle": mpl.cycler(color=PALETTE),
    "mathtext.fontset": "cm",
})


def frame_axes(ax):
    """Force the elegant full-frame look on a single Axes (belt-and-braces for
    scripts that toggle spines locally)."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_edgecolor("#2B2B2B")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.minorticks_on()
    ax.grid(True, color="#B8B8B8", alpha=0.35, linewidth=0.5,
            linestyle=(0, (3, 3)))
    ax.set_axisbelow(True)
    return ax


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {self.label} ...", flush=True)
        return self

    def __exit__(self, *a):
        print(f"[{time.strftime('%H:%M:%S')}] {self.label} done "
              f"({time.time() - self.t0:.1f}s)", flush=True)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def save_json(name, obj):
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(_jsonable(obj), f, indent=1)
    print(f"  wrote {path}")


def save_npz(name, **arrays):
    path = os.path.join(OUT, name)
    np.savez_compressed(path, **arrays)
    print(f"  wrote {path}")


def save_fig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def pdict_to_arrays(pN, Nmax=None):
    """dict N -> value  ->  (Ns, values) arrays on 0..Nmax."""
    if Nmax is None:
        Nmax = max(pN.keys())
    Ns = np.arange(Nmax + 1)
    return Ns, np.array([pN.get(int(N), 0.0) for N in Ns])
