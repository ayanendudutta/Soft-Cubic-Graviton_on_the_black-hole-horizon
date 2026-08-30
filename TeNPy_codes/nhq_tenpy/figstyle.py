#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-contained framed, elegant matplotlib style for the MPS figures.
Matches the style of the paper's nhq.util but carries no Qiskit dependency."""
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

INK, RUST, TEAL, GOLD = "#13294B", "#A23B2E", "#1B7A7A", "#C7902C"
VIOLET, GREEN, GREY = "#5B4B8A", "#2E7D32", "#6B6B6B"
PALETTE = [INK, RUST, TEAL, GOLD, VIOLET, GREEN]

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03, "savefig.facecolor": "white",
    "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
    "legend.fontsize": 8,
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.spines.left": True, "axes.spines.bottom": True,
    "axes.linewidth": 0.8, "axes.edgecolor": "#2B2B2B", "axes.axisbelow": True,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.size": 4.0, "ytick.major.size": 4.0,
    "xtick.minor.size": 2.2, "ytick.minor.size": 2.2,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
    "axes.grid": True, "grid.color": "#B8B8B8", "grid.alpha": 0.35,
    "grid.linewidth": 0.5, "grid.linestyle": (0, (3, 3)),
    "legend.frameon": True, "legend.framealpha": 0.9,
    "legend.edgecolor": "#CCCCCC", "legend.fancybox": False,
    "legend.borderpad": 0.4, "legend.handlelength": 1.6,
    "lines.linewidth": 1.6, "lines.markersize": 4.5,
    "axes.prop_cycle": mpl.cycler(color=PALETTE),
    "mathtext.fontset": "cm",
})
