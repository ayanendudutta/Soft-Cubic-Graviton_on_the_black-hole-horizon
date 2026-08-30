#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
run_mps_climb.py -- multipole-ladder climb (checkpointed, fast).

Evolves the single-(2,0)-graviton quench under the derived cubic Hamiltonian
H = H2^grav + H3^(hhh) on an MPS, from the l = 2 multiplet up to l_max, and
records the diagnostics the exact-diagonalization registers were too small to
settle (bounded chi / area-law entanglement <=> perturbative rigidity persists).

ENGINE (fast hybrid).  H3 is a long-range 3-body vertex whose MPO bond dimension
chi_MPO grows with l (41, 114, ~200, ... for l = 3, 4, 5, ...).  Two integrators
were compared:
  * ExpMPO (Zaletel W^II) costs ~ (chi * chi_MPO)^2  per step  -> ~32 s/step at
    l = 4;  but it CAN seed entanglement from a product state.
  * two-site TDVP costs ~ chi^2 * chi_MPO  per step (linear in chi_MPO) -> much
    cheaper at large l;  but it CANNOT grow entanglement from a product state
    (the long-range TDVP cold-start).
So we use a SHORT, tightly-capped ExpMPO warm-up (a couple of steps at bond
`warmup_chi`) purely to nudge the state off the product manifold, then two-site
TDVP for the bulk.  Net: ~10 s/step at l = 4 instead of ~32 s/step, and the gain
grows with l.  Validated against exact block evolution at l = 2 (eta, S_half to
~1e-3) and against the l = 3 reference; dt = 0.1 is within tolerance.

CHECKPOINTING.  The MPS state + running record are pickled every snapshot to
out_mps/climb_l{L}_d{d}.ckpt; re-running the SAME command resumes.  --walltime S
stops and checkpoints after S seconds (set 1e9 to run straight through).

Recommended (fast, converged) settings are the defaults below:
    python3 run_mps_climb.py --lmax 4 --walltime 1e9
Rough single-core guide with these defaults: l4 ~20 min, l5 ~40 min,
l6 ~1.5 h, l7 ~2.5-4 h.  Multi-threaded BLAS is faster; peak memory < 1 GB.
"""
import warnings; warnings.filterwarnings("ignore")
import argparse, os, sys, time, json, pickle, resource
import numpy as np
import logging
logging.getLogger("tenpy").setLevel(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhq_tenpy.graviton_mps import GravitonCubicModel
from nhq_tenpy.evolve import mps_snapshot
from tenpy.algorithms.mpo_evolution import ExpMPOEvolution
from tenpy.algorithms.tdvp import TwoSiteTDVPEngine

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_mps")
os.makedirs(OUT, exist_ok=True)
_RING = 100000       # disable TeNPy's 2-D "sites per ring" guard (we are 1-D)


def peak_gb():
    # ru_maxrss is in KB on Linux but BYTES on macOS/BSD; normalise to GB.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (rss / (1024**3)) if sys.platform == "darwin" else (rss / (1024**2))


def climb(lmax, d=4, g=12.0, t_final=12.0, dt=0.1, chi_max=24, dsnap=1.0,
          init=(2, 0), svd_min=1e-10, walltime=240.0, n_warmup=2, warmup_chi=8):
    t_wall0 = time.time()
    tag = f"l{lmax}_d{d}"
    ckpt = os.path.join(OUT, f"climb_{tag}.ckpt")

    mdl = GravitonCubicModel(l_max=lmax, d=d, g_eff=g)
    steps = int(round(t_final / dt))
    n_per_snap = max(1, int(round(dsnap / dt)))
    snap_steps = sorted(set(list(range(0, steps + 1, n_per_snap)) + [steps]))

    # ---- resume or initialise -------------------------------------------
    if os.path.exists(ckpt):
        with open(ckpt, "rb") as f:
            st = pickle.load(f)
        psi, rec, cur = st["psi"], st["rec"], st["cur"]
        print(f"[l={lmax}] RESUME step {cur}/{steps} (t={cur*dt:.2f}), "
              f"chi={max(psi.chi)}", flush=True)
    else:
        print(f"[l={lmax}] START  M={mdl.M} modes, {mdl.n_vertices} vertices, "
              f"H-MPO chi={mdl.H_bond_dim}", flush=True)
        psi = mdl.fock_state({init: 1})
        rec = {"times": [], "chi": [], "Ntot": [], "eta": [], "S_half": [],
               "S_max": [], "S_profile": [], "pN": [], "occ": [], "trunc_err": []}
        cur = 0

    def record(step, eng):
        s = mps_snapshot(psi, d, N0=1)
        rec["times"].append(step * dt)
        for k in ("chi", "Ntot", "eta", "S_half", "S_max"):
            rec[k].append(s[k])
        rec["S_profile"].append(s["S_profile"]); rec["pN"].append(s["pN"])
        rec["occ"].append(s["occ"])
        te = getattr(eng, "trunc_err", None)
        rec["trunc_err"].append(float(te.eps) if te is not None else 0.0)
        print(f"    t={step*dt:6.2f}  chi={s['chi']:4d}  Ntot={s['Ntot']:.4f}  "
              f"eta={s['eta']:.4f}  S_half={s['S_half']:.4f}  "
              f"[{time.time()-t_wall0:6.1f}s, {peak_gb():.2f} GB]", flush=True)

    def save_ckpt():
        with open(ckpt, "wb") as f:
            pickle.dump({"psi": psi, "rec": rec, "cur": cur}, f)

    if cur == 0:
        record(0, None)

    # ---- helper: build the right engine for the current phase -----------
    def make_engine(start_step):
        if start_step < n_warmup:      # ExpMPO warm-up (tight cap)
            return ExpMPOEvolution(psi, mdl, {
                "trunc_params": {"chi_max": warmup_chi, "svd_min": svd_min},
                "dt": dt, "N_steps": 1, "order": 2, "approximation": "II",
                "compression_method": "SVD", "start_time": start_step * dt,
                "max_N_sites_per_ring": _RING}), "warmup"
        return TwoSiteTDVPEngine(psi, mdl, {                # bulk: two-site TDVP
            "trunc_params": {"chi_max": chi_max, "svd_min": svd_min},
            "dt": dt, "N_steps": 1, "max_dt": 100.0,
            "start_time": start_step * dt,
            "max_N_sites_per_ring": _RING}), "tdvp"

    # ---- advance snapshot by snapshot, honouring the wall-clock budget ---
    for b in [s for s in snap_steps if s > cur]:
        # the warm-up boundary may fall inside this interval: split at it
        sub_bounds = [b]
        if cur < n_warmup < b:
            sub_bounds = [n_warmup, b]
        for sb in sub_bounds:
            eng, phase = make_engine(cur)
            eng.options['N_steps'] = sb - cur
            eng.run()
            cur = sb
        record(cur, eng)
        save_ckpt()
        if time.time() - t_wall0 > walltime and cur < steps:
            print(f"[l={lmax}] wall budget hit at step {cur}/{steps}; "
                  f"checkpoint saved -- re-run same command to continue.",
                  flush=True)
            return None

    # ---- finished: write summary + arrays, drop checkpoint --------------
    times = np.array(rec["times"]); late = times >= 0.5 * t_final
    lateavg = lambda k: float(np.mean(np.array(rec[k])[late]))
    pbar = {}
    for pN in np.array(rec["pN"], dtype=object)[late]:
        for N, v in pN.items():
            pbar[int(N)] = pbar.get(int(N), 0.0) + v
    nz = int(np.sum(late)); pbar = {N: v / nz for N, v in sorted(pbar.items())}
    summary = dict(
        lmax=lmax, M=mdl.M, d=d, g=g, t_final=t_final, dt=dt, chi_max_cap=chi_max,
        svd_min=svd_min, n_vertices=mdl.n_vertices, H_MPO_chi=mdl.H_bond_dim,
        chi_reached=int(max(rec["chi"])),
        hit_chi_cap=bool(max(rec["chi"]) >= chi_max),
        eta_late_mean=lateavg("eta"), S_half_late_mean=lateavg("S_half"),
        S_max_late_mean=lateavg("S_max"), Ntot_late_mean=lateavg("Ntot"),
        trunc_err_final=float(rec["trunc_err"][-1]),
        pbar_late={str(k): v for k, v in pbar.items()}, peak_GB=peak_gb())
    with open(os.path.join(OUT, f"climb_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    np.savez_compressed(
        os.path.join(OUT, f"climb_{tag}.npz"),
        times=times, chi=np.array(rec["chi"]), eta=np.array(rec["eta"]),
        S_half=np.array(rec["S_half"]), S_max=np.array(rec["S_max"]),
        Ntot=np.array(rec["Ntot"]),
        S_profile=np.array([np.pad(sp, (0, mdl.M - 1 - len(sp)))
                            for sp in rec["S_profile"]]),
        occ=np.array(rec["occ"]),
        pbar_N=np.array(sorted(int(k) for k in pbar)),
        pbar_val=np.array([pbar[k] for k in sorted(pbar)]),
        trunc_err=np.array(rec["trunc_err"]), modes=np.array(mdl.modes))
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print(f"[l={lmax}] TRAJECTORY COMPLETE  chi_reached={summary['chi_reached']}"
          f"  eta_late={summary['eta_late_mean']:.4f}"
          f"  S_half_late={summary['S_half_late_mean']:.4f}"
          f"  (peak {summary['peak_GB']:.2f} GB)", flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmax", type=int, required=True)
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--g", type=float, default=12.0)
    ap.add_argument("--tfinal", type=float, default=12.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--chimax", type=int, default=24)
    ap.add_argument("--dsnap", type=float, default=1.0)
    ap.add_argument("--svdmin", type=float, default=1e-10)
    ap.add_argument("--walltime", type=float, default=240.0)
    ap.add_argument("--nwarmup", type=int, default=2)
    ap.add_argument("--warmupchi", type=int, default=8)
    a = ap.parse_args()
    climb(a.lmax, d=a.d, g=a.g, t_final=a.tfinal, dt=a.dt, chi_max=a.chimax,
          dsnap=a.dsnap, svd_min=a.svdmin, walltime=a.walltime,
          n_warmup=a.nwarmup, warmup_chi=a.warmupchi)
