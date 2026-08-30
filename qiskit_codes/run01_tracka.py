#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
 run01_tracka.py -- the elastic backbone: calibration on KNOWN physics
================================================================================
Produces the elastic-benchmark calibration figures of the paper:

  fig_A1_trotter.pdf   IHO split-operator Trotter infidelity vs r  (r^-4)
  fig_A2_otoc.pdf      grid OTOC vs the analytic cosh^2(Omega t)   (Remark 1:
                       a c-number commutator, NOT an MSS statement)
  fig_A3_shift.pdf     't Hooft shift circuit exactness vs coupling c_l
  fig_A4_spectrum.pdf  discretised IHO spectrum + level-spacing (integrable;
                       <r> far above GOE because near picket-fence -- reported
                       as the documented non-chaotic elastic limit)

Runtime: ~2-5 min (the nq = 7 spectrum is the slow part); NHQ_QUICK=1
reduces nq.
================================================================================
"""
import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector, Operator

from nhq import tracka as ta
from nhq import model as md
from nhq.util import Timer, save_json, save_npz, save_fig, QUICK, \
    INK, RUST, TEAL, GOLD, GREY


def main():
    summary = {}
    nq = 5 if QUICK else 6
    Omega = 1.0

    # ---------------- A1: Trotter convergence ---------------------------
    with Timer("A1: IHO Strang-Trotter convergence"):
        H = ta.iho_hamiltonian_dense(nq, Omega)
        psi0 = ta.gaussian_packet(nq)
        t = 1.0
        psi_ex = sla.expm(-1j * H * t) @ psi0
        rs = np.array([1, 2, 4, 8, 16, 32])
        infid = []
        for r in rs:
            psi_c = Statevector(psi0).evolve(
                ta.iho_trotter_circuit(nq, Omega, t, int(r))).data
            infid.append(max(1 - abs(np.vdot(psi_ex, psi_c)) ** 2, 1e-16))
        infid = np.array(infid)
        summary["A1_infidelity"] = dict(zip(rs.tolist(), infid.tolist()))

        fig, ax = plt.subplots(figsize=(3.4, 2.7))
        ax.loglog(rs, infid, "o-", color=INK, label="measured")
        ax.loglog(rs, infid[1] * (rs / rs[1]) ** -4.0, "--", color=GREY,
                  label=r"$\propto r^{-4}$ (2nd order)")
        ax.set_xlabel("Trotter steps $r$")
        ax.set_ylabel(r"infidelity $1-|\langle\psi_{\rm ex}|\psi_r\rangle|^2$")
        ax.set_title(f"IHO split-operator Trotter ($n_q={nq}$, $t=1$)")
        ax.legend()
        save_fig(fig, "fig_A1_trotter.pdf")

    # ---------------- A2: OTOC vs cosh^2 --------------------------------
    with Timer("A2: grid OTOC vs analytic cosh^2"):
        N, du, u, p = ta.grid(nq)
        F = np.exp(-1j * np.outer(p, u)) / np.sqrt(N)
        uop = np.diag(u).astype(complex)
        pop = F.conj().T @ np.diag(p).astype(complex) @ F
        psi0 = ta.gaussian_packet(nq, sigma=1 / np.sqrt(2))
        ts = np.linspace(0, 2.0, 41)
        C = md.otoc_squared_commutator(H, uop, pop, ts, state=psi0)
        summary["A2_max_reldev_early"] = float(np.max(np.abs(
            C[ts < 1.0] / np.cosh(Omega * ts[ts < 1.0]) ** 2 - 1)))

        fig, ax = plt.subplots(figsize=(3.4, 2.7))
        ax.plot(ts, C, "o", ms=3, color=INK, label="grid $C(t)$")
        ax.plot(ts, np.cosh(Omega * ts) ** 2, "-", color=RUST,
                label=r"$\cosh^2(\Omega t)$")
        ax.set_xlabel("$t$")
        ax.set_ylabel(r"$C(t)=\langle|[\hat u(t),\hat p]|^2\rangle$")
        ax.set_title("bare IHO OTOC (c-number commutator)")
        ax.legend()
        save_fig(fig, "fig_A2_otoc.pdf")
        save_npz("data_A2_otoc.npz", ts=ts, C=C)

    # ---------------- A3: exact shift circuit ---------------------------
    with Timer("A3: 't Hooft shift circuit exactness"):
        nq_s = 3 if QUICK else 4
        GN = 0.01
        ls = np.arange(2, 9)
        cls = 8 * np.pi * GN / (ls ** 2 + ls + 1)
        devs = []
        for c in cls:
            dev = np.abs(Operator(ta.shift_circuit(nq_s, c)).data
                         - ta.shift_dense(nq_s, c)).max()
            devs.append(dev)
        summary["A3_max_dev"] = float(np.max(devs))

        fig, ax = plt.subplots(figsize=(3.4, 2.7))
        ax.semilogy(cls, devs, "s-", color=TEAL)
        ax.set_xlabel(r"shift coupling $c_\ell = 8\pi G_N/(\ell^2+\ell+1)$")
        ax.set_ylabel("circuit vs dense max deviation")
        ax.set_title(f"'t Hooft shift: exact circuit ($n_q={nq_s}$/register)")
        save_fig(fig, "fig_A3_shift.pdf")

    # ---------------- A4: IHO spectrum + spacing ------------------------
    with Timer("A4: IHO spectrum and level spacing"):
        nq_sp = 6 if QUICK else 7
        Hs = ta.iho_hamiltonian_dense(nq_sp, Omega)
        E = np.sort(np.linalg.eigvalsh(Hs).real)
        r_mean, nlev = md.r_ratio(E)
        s = md.unfolded_spacings(E)
        summary["A4_r_mean"] = r_mean
        summary["A4_n_levels"] = nlev

        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))
        axes[0].plot(E, ".", ms=2, color=INK)
        axes[0].set_xlabel("level index")
        axes[0].set_ylabel("$E$")
        axes[0].set_title(rf"discretised IHO spectrum ($n_q={nq_sp}$)")
        x = np.linspace(0, 4, 200)
        axes[1].hist(s, bins=30, density=True, color=GOLD, alpha=0.6,
                     label=rf"grid IHO, $\langle r\rangle={r_mean:.3f}$")
        axes[1].plot(x, np.exp(-x), "--", color=GREY, label="Poisson")
        axes[1].plot(x, (np.pi / 2) * x * np.exp(-np.pi * x ** 2 / 4), "-",
                     color=RUST, label="Wigner (GOE)")
        axes[1].set_xlabel("normalised spacing $s$")
        axes[1].set_ylabel("$P(s)$")
        axes[1].set_title("integrable elastic limit (documented)")
        axes[1].legend()
        save_fig(fig, "fig_A4_spectrum.pdf")

    save_json("summary_tracka.json", summary)
    print("\nrun01 complete.")


if __name__ == "__main__":
    main()
