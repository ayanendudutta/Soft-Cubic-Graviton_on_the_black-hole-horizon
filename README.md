# The leading-soft cubic graviton self-interaction on the black-hole horizon

Derivation and quantum-simulation code accompanying the paper.

The paper expands the Einstein–Hilbert action to cubic order about the
Schwarzschild horizon, in the even Regge–Wheeler gauge of the
Gaddam–Groenenboom–'t Hooft (GGV) near-horizon framework, and derives the cubic
graviton self-interaction. Its central analytic result is a vanishing theorem —
at leading soft order the self-coupling of the purely traceless longitudinal
polarizations is identically zero, because with the trace/transverse-scalar
sector switched off the fluctuation reduces to a two-dimensional block whose
`sqrt(-g) R` is a total (Euler) derivative at every order in `kappa`. The
surviving interaction lives in the trace sector, with on-shell equal-multipole
weight

```
W(lam, lam, lam) = -3 lam (2 lam^2 + lam + 3) / (lam + 1)^2 ,   lam = l^2 + l + 1
W(7, 7, 7)       = -567/16 = -35.4375
```

The second-quantized Hamiltonian built from this vertex is then evolved in real
time on Qiskit/Aer, with every circuit result overlaid on an exact-diagonalization
counterpart.

This repository contains three parts: the symbolic derivation and its
independent numerical verifications; the exact-diagonalization and Qiskit/Aer
simulation suite that produces every figure and every quoted number of the main
text; and a tensor-network (matrix-product-state) extension that lifts the
same Hamiltonian off the small exact-diagonalization registers and climbs the
multipole ladder to `l_max = 7` (60 modes, the 120-qubit target of Appendix H),
testing whether the perturbative rigidity persists as the register grows.

---

## Repository layout

```
.
├── README.md
├── .gitignore
│
├── derivation_codes/            symbolic derivation and its verifications
│   ├── derive_L3.py               cubic expansion of sqrt(-g) R  ->  L23.pkl
│   ├── cubic.py                   momentum-space vertex; vanishing theorem  ->  V3.pkl
│   ├── onshell.py                 on-shell weight W(lam1, lam2, lam3)  ->  W.pkl
│   ├── organize.py                exact covariant decomposition  ->  fit.pkl
│   ├── dilaton_W.py               independent (warped-product) re-derivation of W
│   ├── theorem_explicit.py        term-by-term collapse of the traceless sector
│   ├── verify_gr.py               exact-nonlinear-GR check of L2 and L3
│   ├── run_verify2.py             multi-seed driver for the above
│   ├── verify_euler_identities.py Euler-homogeneity check of -1/4 and kappa^3/6
│   └── rerun_derived_kernel.py    the derived weight propagated into the model
│
├── qiskit_codes/                simulation suite
│   ├── nhq/                       the simulation package
│   │   ├── encoding.py              standard-binary / Gray / unary bosonic encodings
│   │   ├── model.py                 vertex tables, Fock engines, observables, references
│   │   ├── circuits.py              SparsePauliOp Hamiltonians, Trotter circuits, Aer runs
│   │   ├── tracka.py                elastic backbone: grid IHO and the Dray–'t Hooft shift
│   │   └── util.py                  output paths, smoke-test mode, figure style
│   ├── run00_validate.py          multi-engine validation table
│   ├── run01_tracka.py            elastic-benchmark calibration
│   ├── run02_multiplicity.py      multiplicity dynamics, onset, ablation, multi-l
│   ├── run03_page_profile.py      mode-bipartition entanglement profiles
│   ├── run04_spectral.py          two-sided spectral statistics
│   ├── run05_otoc.py              squared commutator, dense and circuit
│   ├── run06_cascade.py           resonance audit, cascade, power laws
│   ├── run07_noise_resources.py   transpiled resources and depolarizing-noise budget
│   ├── run08_circuit_figures.py   the three circuit diagrams
│   └── out/                       figures, raw arrays, and JSON summaries
│
└── TeNPy_codes/                 tensor-network (MPS) extension: climbing the ladder
    ├── requirements.txt           dependencies (physics-tenpy; no Qiskit for the climb)
    ├── run_all.sh                 climb l = 2..7, then build the figures
    ├── run_mps_climb.py           the checkpointed, resumable climb driver
    ├── make_mps_figures.py        the framed climb figures figM1..figM4
    ├── make_fig_v3.py             the two-tier (d = 4 climb + d = 8 check) figures
    ├── validate_tenpy_vs_ed.py    two-engine spectrum + dynamics check vs nhq
    ├── validate_dynamics.py       hybrid-evolver dynamics check vs nhq
    ├── nhq_tenpy/                 the tensor-network package
    │   ├── vertex.py                cubic-vertex table, transcribed from nhq.model
    │   ├── graviton_mps.py          the Lz-graded MPO model
    │   ├── evolve.py                hybrid ExpMPO/TDVP evolver + MPS observables
    │   └── figstyle.py              self-contained framed matplotlib style
    └── out_mps/                    climb summaries, arrays, and figures
```

`out/` is committed: it holds the exact figures and numbers appearing in the
paper, so that results can be inspected without re-running anything.

---

## Environment

Every result in the paper was produced on this stack:

| component | version |
|---|---|
| Python | 3.12 |
| Qiskit | 2.4.2 |
| Qiskit-Aer | 0.17.2 |
| NumPy | 2.4 |
| SciPy | 1.17 |
| SymPy | 1.14 |
| Matplotlib | 3.10 |
| TeNPy (`physics-tenpy`) | 1.1.0 |

```bash
python3 -m venv venv && source venv/bin/activate
pip install "qiskit==2.4.2" "qiskit-aer==0.17.2" \
            "numpy>=2.4" "scipy>=1.17" "sympy>=1.14" matplotlib pylatexenc
```

`pylatexenc` is required only by `run08_circuit_figures.py`, which renders
circuit diagrams with LaTeX-style gate labels. `physics-tenpy==1.1.0` is
required only by the tensor-network extension; the derivation and the Qiskit
suite do not need it. Its dependencies are pinned in
`TeNPy_codes/requirements.txt`:

```bash
pip install -r TeNPy_codes/requirements.txt
```

No GPU, no hardware credentials, and no network access are required. The
simulation suite runs on a laptop.

### The run engine

All quantum results in the paper are **classical simulations of quantum
circuits**; no run was executed on physical hardware, and the paper defers
hardware execution behind the quantified noise budget of `run07`. Three
distinct engines are used, and the discipline of the suite is that they are
cross-checked against one another rather than trusted individually:

| engine | what it is | where |
|---|---|---|
| **E1** | dense Kronecker-product matrices on the full `d^M` product space | `nhq.model.H_dense_kron` |
| **E2** | direct occupation-basis (Fock) matrices, per-mode and/or total-`N` truncated, optionally restricted to an `Lz` block | `nhq.model.FockBasis` |
| **E3** | Qiskit `SparsePauliOp` on an encoded qubit register, evolved by decomposed Suzuki–Trotter circuits | `nhq.circuits` |

Concretely:

- **Exact dynamics** (`run02`–`run06`) are computed in E2 by one
  eigendecomposition per block (`nhq.model.BlockEvolver`), or, where only a
  few snapshot times on many Hamiltonians are needed, by SciPy's Krylov
  `expm_multiply` (`nhq.model.evolve_krylov`).
- **Circuit dynamics** use `AerSimulator(method="statevector")`. Long
  evolutions use the sequential-segment protocol
  (`nhq.circuits.trotter_snapshot_states`): one transpiled segment
  `U_T(dt)` is applied repeatedly, each saved statevector fed back as the next
  initial state, so memory is bounded by a single segment and the Trotter
  error accumulates linearly, as it would on hardware.
- **Shot-based measurement** (multiplicity readout, the OTOC Hadamard test)
  samples the same backend at finite shots with binomial errors.
- **Noise** (`run07`) uses `AerSimulator(method="density_matrix")` with an
  `AerSimulator` depolarizing model, giving the exact noisy density matrix
  with no shot noise.
- The **derivation** is pure SymPy plus NumPy/FFT lattice computations; it
  imports no Qiskit at all, except in `rerun_derived_kernel.py`, which reuses
  the `nhq` model engine.

One Qiskit pitfall is enforced everywhere and is worth restating: a
`PauliEvolutionGate` applied **un-decomposed** is silently replaced by the exact
matrix exponential in statevector simulation, which masks all Trotter error.
Every circuit here therefore calls `.decompose(reps=...)` before simulation, so
the Trotter error is genuine and set solely by the Suzuki repetitions; both the
decomposition-independence and the resulting `r^-4` scaling are asserted in
`run00`.

---

## Quick start

```bash
cd qiskit_codes
python3 run00_validate.py        # ~40 s: all engines agree, charges as documented
```

`run00` gates everything else. It should end with `run00 complete: all engines
agree; charges as documented.`

A smoke-test mode reduces every register and sweep so that a whole script
finishes in seconds to a couple of minutes:

```bash
NHQ_QUICK=1 python3 run02_multiplicity.py
```

Smoke-test results are **not** the results of the paper — the mode exists to
verify that a script runs end to end. Reproducing the paper requires the full
mode. Note that either mode overwrites `out/`; to keep the committed
production outputs, copy the directory aside first or run from a scratch clone.

---

## Reproducing the paper

Scripts are mutually independent and may be run in any order after `run00`.
Run them from inside `qiskit_codes/`; all outputs go to `out/`.

| script | produces | figures in the paper | dominant cost |
|---|---|---|---|
| `run00_validate.py` | `validation_summary.json` | validation table | engine cross-checks and the Trotter convergence scan |
| `run01_tracka.py` | `summary_tracka.json`, `data_A2_otoc.npz` | 16, 17, 24 | dense diagonalization of the `nq = 7` inverted-oscillator grid |
| `run02_multiplicity.py` | `summary_multiplicity.json`, `data_B1_*`, `data_B2_*` | 4, 5, 6, 7, 8 | the `d = 6` coupling sweep and a twenty-segment ten-qubit Aer Trotter evolution |
| `run03_page_profile.py` | `summary_page.json`, `data_C1_*`, `data_C3_*` | 9, 18, 19, 20 | entanglement averaging over mode bipartitions, including the Haar reference |
| `run04_spectral.py` | `summary_spectral.json`, `data_D3_sff.npz` | 12, 13, 23 | the `d = 6` and multi-multipole `Nmax = 5` block diagonalizations |
| `run05_otoc.py` | `summary_otoc.json`, `data_E1_*`, `data_E2_*` | 14, 21 | Hadamard-test interferometers at ten times on an eleven-qubit register |
| `run06_cascade.py` | `summary_cascade.json`, `data_F1_*`, `data_F2_*` | 10, 11, 22 | Krylov evolution across the coupling sweep at `d = 3` and `d = 4` |
| `run07_noise_resources.py` | `summary_noise.json`, `resource_table.txt` | 15 | exact density-matrix evolution of a 29520-cx circuit at eight noise strengths |
| `run08_circuit_figures.py` | `summary_circuit_figures.json` | 1, 2, 3 | circuit construction and rendering |

On a single modern core with no parallelism, most of these complete within
about a minute. `run02` and `run05` are substantially heavier, and are also the
memory-hungry ones: both drive Aer with deep transpiled circuits on ten- and
eleven-qubit registers, so a few gigabytes of free RAM are worth having before
starting them. `NHQ_QUICK=1` reduces every register and sweep and is the right
way to confirm that a script runs end to end before committing to a full run.

`run01` also emits `fig_A3_shift.pdf`, a scan of the 't Hooft shift-circuit
exactness against the coupling. The paper quotes that check numerically rather
than as a figure, so the file has no figure number.

### The derivation chain

`derivation_codes/` splits into a **sequential chain** and a set of
**independent verifications**.

The chain must be run in order, since each stage consumes the pickle written by
the previous one (these intermediates are `.gitignore`d and regenerate in well
under a minute):

```bash
cd derivation_codes
python3 derive_L3.py    # ~10 s  -> L23.pkl   cubic expansion, 76 monomials
python3 cubic.py        # ~5 s   -> V3.pkl    375 multilinear terms; traceless restriction = 0
python3 onshell.py      # ~15 s  -> W.pkl     closed-form W(lam1, lam2, lam3)
python3 organize.py     # ~1 s   -> fit.pkl   exact covariant decomposition
```

The independent verifications may be run in any order. Three of them need no
input at all; `verify_gr.py` and `run_verify2.py` require `L23.pkl`, and
`rerun_derived_kernel.py` requires the `nhq` package:

```bash
python3 dilaton_W.py               # ~20 s  independent re-derivation of W
python3 theorem_explicit.py        # ~25 s  term-by-term collapse (needs nothing)
python3 verify_gr.py               # ~1 s   exact-nonlinear-GR vs symbolic L2, L3
python3 run_verify2.py             # ~1 s   multi-seed driver for the same check
python3 verify_euler_identities.py # ~25 s  Euler homogeneity on a 16^4 lattice
python3 rerun_derived_kernel.py    # ~1 s   derived weight through the model
```

`verify_euler_identities.py` honours `NHQ_QUICK=1`, which selects an `8^4`
lattice. `rerun_derived_kernel.py` locates the simulation suite as the sibling
directory containing the `nhq` package and writes
`summary_derived_kernel.json` into that suite's `out/`, so it runs correctly
from any working directory.

---

## The tensor-network extension

`TeNPy_codes/` lifts the same Hamiltonian `H = H2^grav + H3` off the small
exact-diagonalization registers and represents it as a matrix-product operator,
so real-time evolution can reach mode numbers far beyond exact diagonalization.
The single-`(2,0)`-graviton quench is evolved from the `l = 2` multiplet
(5 modes) up to `l_max = 7` (60 modes, the 120-qubit target of Appendix H),
and the diagnostics the exact-diagonalization registers were too small to
settle are read off: whether the inelasticity, the sub-Page entanglement, and
the bounded bond dimension **persist** as the tower grows.

### Method

- **Conserved charge.** `H3` changes total boson number by `+-1`, so total `N`
  is not conserved and cannot block-diagonalize the problem. What is conserved
  exactly is the azimuthal charge `Lz = sum_i m_i n_i`. Each bosonic mode `i` is
  therefore given a U(1) charge `m_i * n_i`, so the global conserved number is
  `Lz`; a cubic term `b_i^dag b_j^dag b_k` carries net charge
  `m_i + m_j - m_k` and TeNPy admits it into the MPO only when that is zero —
  i.e. only when `m1 + m2 = m3` — so the magnetic selection rule is enforced by
  the symmetry structure itself. Working in the fixed `Lz = 0` sector is what
  makes the climb to 45 and 60 modes tractable.
- **Evolution (hybrid).** `H3` is a genuinely long-range three-body vertex, so
  two-site TDVP cannot grow entanglement from the product initial state (the
  long-range cold-start). Forming `exp(-iH dt)` as an MPO (Zaletel W^II) does
  seed the entanglement, but its cost `~(chi * chi_MPO)^2` per step grows with
  the MPO bond dimension `chi_MPO` (41, 114, 228, 429 at `l = 3, 4, 5, 6`). The
  evolver therefore runs a short, tightly-capped ExpMPO warm-up purely to move
  the state off the product manifold, then two-site TDVP — whose cost
  `~chi^2 * chi_MPO` is linear in `chi_MPO` — for the bulk of the trajectory.
- **Site ordering.** As orbital ordering matters in quantum-chemistry DMRG, the
  default `paired_m` order places `+m` and `-m` partners adjacent, localizing
  the dominant `(m, -m)` splitting channel and minimizing the bond dimension.
- **Observables without the statevector.** `P(N)` is obtained from the
  characteristic function `<exp(i theta N_tot)>` — a bond-dimension-one product
  operator contracted as a transfer network — by inverse DFT; entanglement is
  read from the MPS Schmidt spectra. The `d^M` statevector is never formed.

### Two-engine validation

The same engine-against-engine discipline as the rest of the repository carries
over: at the `l = 2` register (small enough to diagonalize densely) the MPO and
the reference occupation-basis engine `nhq.model` must agree to machine
precision, and the real-time dynamics of the quench must agree between exact
block evolution and near-exact TDVP.

```bash
cd TeNPy_codes
python3 validate_tenpy_vs_ed.py     # full Lz=0 spectrum + dynamics, l=2, d=4 and d=6
python3 validate_dynamics.py        # hybrid evolver vs exact block evolution, l=2
```

Both scripts import the reference engine from the sibling `qiskit_codes/`
directory automatically; set `NHQ_QISKIT_DIR` only to point elsewhere. The
spectrum check agrees to `~1e-13` over the full `Lz = 0` spectrum
(`5.3e-14` at `d = 4`, `2.2e-13` at `d = 6`), and the dynamics agree to
`~1e-3` (integrator-limited, the same tolerance as the Qiskit Trotter engine),
improving as `dt` is reduced.

### Running the climb

```bash
cd TeNPy_codes
bash run_all.sh                              # l = 2..7 to completion, then figures
```

or one register at a time:

```bash
python3 run_mps_climb.py --lmax 2 --walltime 1e9     # 5 modes
python3 run_mps_climb.py --lmax 3 --walltime 1e9     # 12 modes
python3 run_mps_climb.py --lmax 4 --walltime 1e9     # 21 modes
python3 run_mps_climb.py --lmax 5 --walltime 1e9     # 32 modes
python3 run_mps_climb.py --lmax 6 --walltime 1e9     # 45 modes
python3 run_mps_climb.py --lmax 7 --walltime 1e9     # 60 modes
```

The run is **resumable**: the full MPS state and running record are checkpointed
to `out_mps/climb_l{L}_d{d}.ckpt` after every snapshot, so re-running the same
command resumes from the last snapshot. `--walltime S` makes a run stop and
checkpoint after `S` seconds (default 240); set it large (`1e9`) to run
straight through. On completion a trajectory prints `TRAJECTORY COMPLETE`,
writes its `.json` and `.npz`, and deletes the checkpoint.

The defaults are the validated, converged settings: per-mode cutoff `d = 4`,
reduced coupling `g = 12`, evolution time `t = 12` in units `R_S/c`, integrator
step `dt = 0.1`, bond-dimension cap `chi = 24`, with a two-step ExpMPO warm-up
at a tight cap. The climb cost grows steeply with depth; the low multipoles are
quick, while `l = 6, 7` are heavy. Peak memory stays well under a gigabyte
throughout. `out_mps/` ships with the completed reference runs for `l = 2..7`,
so the figures can be rebuilt without re-running the climb.

| flag | default | meaning |
|---|---|---|
| `--lmax` | (required) | highest multipole; modes `M = (l_max+1)^2 - 4` |
| `--d` | 4 | per-mode boson cutoff (local dimension) |
| `--g` | 12.0 | reduced coupling `g-tilde` |
| `--tfinal` | 12 | evolution time in units `R_S/c` |
| `--dt` | 0.1 | integrator step |
| `--chimax` | 24 | bond-dimension cap |
| `--svdmin` | 1e-10 | Schmidt-value floor |
| `--walltime` | 240 | per-call wall-clock budget (s); set 1e9 to finish |
| `--nwarmup` | 2 | ExpMPO warm-up steps |
| `--warmupchi` | 8 | tight bond cap during warm-up |

### The figures

```bash
python3 make_mps_figures.py      # figM1..figM4  (the mode-climb figures, d = 4)
python3 make_fig_v3.py           # fig_H2_summary, fig_H3_tower  (two-tier d = 4 + d = 8)
```

`make_mps_figures.py` reads every `out_mps/climb_l*_d4.npz` and writes: the
late-time entanglement profile across the mode chain (`figM1`), the four-panel
summary of late-time inelasticity, peak entanglement, mean number, the `l = 4`
bond-dimension convergence, and the truncation error versus mode number
(`figM2`), the late-time multiplicity tower `P(N)` per depth (`figM3`), and the
peak-entanglement and truncation-error dynamics (`figM4`). `make_fig_v3.py`
reads both the `d = 4` mode-climb and the `d = 8` occupation-refinement runs and
writes the two two-tier figures used in Appendix H, showing that the saturated
observables are robust to both truncation axes.

The mode-climb figures use the fixed `d = 4` runs so that each mode number
appears once; the `d = 8, 16, 32` runs are the occupation-axis refinement and
enter only through `make_fig_v3.py`. The half-cut entropy `S_half` is not a
faithful size-independent entanglement measure under the paired-`m` ordering
(the central bond drifts into the weakly-populated high-`|m|` tail), so the
figures report the peak `S_max` and the full profile instead — the correct
observables for this site ordering.

### What the climb shows

The perturbative rigidity and the sub-Page entanglement persist to `M = 60`
modes: the inelasticity converges to a finite `eta-bar ~ 0.156`, the peak
entanglement saturates at an area-law `S_max ~ 0.45` nats, the mean graviton
number saturates near `N-bar ~ 1.20`, and the final truncation error stays
below `10^-3` at fixed bond dimension — the sharp, quantitative statement that
the dynamics is not scrambling, since a scrambling state could not be held by a
small-bond MPS. The mode-axis climb (`d = 4`) and the occupation-axis
refinement (`d = 8`) agree across the tower, so the saturated observables are
robust to both truncations.

**Scope.** The tensor-network method extends the *dynamical and entanglement*
observables (inelasticity, `P(N,t)`, entanglement, the multiplicity tower) to
large `M`. It does not extend the interior level-spacing statistics `<r>` —
those need many highly-entangled interior eigenstates, outside standard
DMRG/TDVP — so that result stays at its exact-diagonalization register, as the
paper states.



**The vanishing theorem** is checked four independent ways, and the checks do
not share a substrate:

- `cubic.py` reduces the 375-term momentum-space vertex on the conservation
  surface `p3 = -p1 - p2` and returns an exact symbolic zero once the
  transverse scalar is set to `K = 0`, with the longitudinal trace left
  arbitrary — the sharpened form of the theorem.
- `theorem_explicit.py` exhibits the collapse term by term: the
  thirteen-monomial cubic density of the traceless two-dimensional block is an
  exact total derivative with explicit divergence potentials, and both
  Euler–Lagrange derivatives vanish identically.
- `dilaton_W.py` re-derives the surviving vertex from the exact warped-product
  identity without ever forming the four-dimensional vertex (nineteen
  monomials against seventy-six), reproduces the closed-form weight with ratio
  exactly 1 on the whole equal-`lambda` diagonal, and independently returns
  zero at `K = 0`.
- `verify_gr.py` / `run_verify2.py` evaluate the exact nonlinear `sqrt(-g) R`
  on random traceless configurations and find the cubic Taylor coefficient at
  machine zero.

**The normalization coefficients** — the `-1/4` of the quadratic action and the
`kappa^3/6` of the cubic self-sourcing identity — are fixed by
`verify_euler_identities.py`, by exact-nonlinear GR on a periodic `16^4`
lattice, independently of any perturbative expansion. It reports four
diagnostics: pipeline consistency (`dF/dkappa + J` at machine zero, confirming
that the independently coded nonlinear `R` and `G_{mu nu}` agree),
`c2 = -0.250000`, the Euler identities `2 s2/j1 = 3 s3/j2 = -1.000000`, and the
discriminating negative control, in which the `eta`-raised, measure-free
contraction gives a seed-dependent ratio instead of `-1`. The last diagnostic
is the point: the exact identity holds only for the densitized,
full-metric-raised current.

**The off-diagonal weights are prescription-dependent, and the code says so.**
`dilaton_W.py` evaluates `W` under four defensible off-shell prescriptions and
prints the spread. The equal-`lambda` value is prescription-independent and
reproduces `-35.4375`; the off-diagonal ratios span a factor of about three to
four. The multi-multipole shift reported in the paper is therefore one
representative prescription carrying an explicitly quantified scheme
systematic, not a sharp number. `rerun_derived_kernel.py` propagates one
prescription through the model and measures the consequence
(`TV = 0.093` on the late-time distribution, `-8.6%` on the inelasticity),
after asserting that on a single-multipole register the derived weight is an
exact overall constant.

**The simulation's non-tautology test** is the structure ablation in `run02`:
the gravitationally fixed vertex table is re-run against a uniform-magnitude
table and five magnitude-reshuffled tables with the selection rules left
intact. The inelasticity varies ninefold across variants while every variant
remains equally non-ergodic — which is what licenses the paper's decomposition
of the signal into a generic part and a structure-specific part.

---

## Conventions

Units are `hbar = 1`, `kappa = 1`, `R = R_S = 1`. Frequencies are
`omega_l = sqrt(l^2 + l + 1) / R`.

**Couplings.** All couplings are quoted as a reduced `g̃`, which multiplies the
unit-normalized, gravitationally fixed vertex table (Gaunt coefficient ×
normalization × combinatorial weight). `g̃` replaces the Planck-suppressed
physical magnitude `gamma |W| K` and sweeps the overall scale only; the
relative structure is never dialed. The working window is `g̃ ≲ 12–15`, which is
Fock-converged at the few-percent level (`d = 4` against `d = 5, 6`). Results
are therefore mechanism structure, not absolute rates.

**Operator-level magnetic rule.** The Gaunt selection rule applies to the
harmonics of the field modes. Once the real field is expanded, each creation
operator enters with a conjugated harmonic, and the field-label rule translates
into the physical conservation law `m1 + m2 = m3` for the quanta in
`b1† b2† b3`, with the amplitude carrying a `(-1)^{m3}` phase and a
combinatorial factor 2 for distinct creation pairs. Transcribing the
field-label rule directly onto operator labels silently violates `Lz`
conservation; `[H, Lz] = 0` is asserted at run time in every simulation for
exactly this reason.

**Two truncations.** The per-mode cutoff `d` is what a qubit register
implements, but it breaks the model's `su(2)` at the occupation boundary
(`‖[H, L²]‖ ≈ 22` at `d = 4` on the `l = 2` multiplet). The total-occupation
cutoff `sum_i n_i ≤ Nmax` commutes with the ladder generators exactly
(`‖[H, L²]‖ ~ 1e-16`), so it is the truncation under which fixed-`(L², Lz)`
spectral statistics are attributable to the model. Both are implemented, and
the paper reports the spectral statistics two-sidedly for this reason.

**Mode ordering and endianness.** Qiskit is little-endian: qubit 0 is the
rightmost character of a Pauli label. Mode `i` occupies global qubits
`[i·nb, (i+1)·nb)`, so mode 0 is the least-significant block, matching a dense
Kronecker embedding in which mode 0 is the least-significant factor. The two
orderings are easy to transpose silently, which is why the encoded and dense
engines are compared bit-for-bit in `run00`.

---

## Scope of what the code computes

These constraints are properties of the model as implemented, and they bound
what any number produced here can mean.

- `N` counts occupation of the GGV longitudinal (non-radiative) sector. The map
  to radiative Regge–Wheeler–Zerilli multiplicity is open.
- The radial kernel is the leading-soft `K = 1`. The derived weight supplies
  the multi-multipole structure and carries the prescription systematic
  described above.
- The background is static Schwarzschild. Entanglement results are statements
  about entanglement structure, not about evaporation; "Page-like profile"
  means the subsystem-size dependence of the entropy compared with a
  random-state reference, and nothing more.
- Registers use `l ≥ 2`; the monopole and dipole are gauge-special and are
  excluded. The odd-parity and radiative sectors lie outside the truncation.
- No chaos claim is made for the physical dynamics: no Lyapunov exponent is
  extracted for the derived Hamiltonian, and no saturation of the
  Maldacena–Shenker–Stanford bound is claimed. A Lyapunov fit is performed only
  on the exactly solvable inverted-oscillator benchmark, where the exponential
  window exists by construction, and serves to validate the fitting pipeline.
- The spectral statistics of the physical sectors are reported as an indicative
  trend on Hilbert spaces too small to establish a universality class, and are
  not extrapolated. The tensor-network climb extends the dynamical and
  entanglement observables to 60 modes but does **not** extend these
  level-spacing statistics, which require many highly-entangled interior
  eigenstates and stay at the exact-diagonalization register.
- All quantum-circuit results are Aer statevector or density-matrix
  simulations. Hardware execution is deferred on the budget computed in
  `run07`: at ten qubits, a ten-percent bias on the inelasticity requires a
  two-qubit error rate `p2 ≲ 5e-5`. Depolarizing noise drives the state toward
  the maximally mixed state, whose spurious inelasticity (`0.953` on that
  register) dwarfs the ideal signal — noise manufactures apparent particle
  production, so any hardware measurement of the inelasticity must be reported
  jointly with the fidelity.
- The tensor-network climb is a classical MPS simulation. Its convergence
  controls are the per-mode cutoff `d` and the bond-dimension cap `chi`; both
  are checked (the `d = 4` climb against a `d = 8` refinement, and `chi` against
  a `chi = 24, 32, 48, 96` sweep at `l = 4`), and the truncation error is
  reported alongside every trajectory. The area-law reading is meaningful only
  while that error stays small — which it does across the whole tower, and whose
  eventual growth would itself signal the onset of scrambling.

---

## Citation

Please cite the paper when using this code. A. Dutta, *The leading-soft cubic
graviton self-interaction on the black-hole horizon*.
