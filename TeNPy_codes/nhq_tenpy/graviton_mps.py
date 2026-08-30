#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
 nhq_tenpy.graviton_mps -- TeNPy MPS/MPO build of the derived near-horizon
                           cubic graviton Hamiltonian H = H2^grav + H3^(hhh)
================================================================================
This is the tensor-network engine for the multipole-ladder extension of
the paper.  It reproduces, mode for mode and coefficient for coefficient, the
occupation-basis Hamiltonian of `nhq.model` (the Qiskit-suite reference), but
represents it as a matrix-product operator so that real-time evolution (TDVP)
and ground/excited-state search (DMRG) can reach mode numbers far beyond exact
diagonalization.

Physics (units hbar = 1, kappa = 1, R = R_S = 1), identical to nhq.model:

    H2^grav = sum_i  omega_{l_i} b_i^dag b_i ,     omega_l = sqrt(l^2+l+1)/R,
    H3      = sum_{(i<=j),k} V_{ijk} ( b_i^dag b_j^dag b_k + h.c. ),
    V_{ijk} = sym * g_eff * G_op[123] * K * / sqrt(8 w_i w_j w_k),   K = 1,
    G_op    = (-1)^{m3} Gaunt(l1 m1, l2 m2, l3, -m3),   nonzero iff
              m1 + m2 = m3, triangle(l1,l2,l3), l1+l2+l3 even,
    sym     = 2 (i<j) or 1 (i=j)     [unrestricted-sum fold].

Key design choice -- the conserved charge.  H3 changes total boson number by
+-1, so total N is NOT conserved and cannot be used to block-diagonalize.
What IS conserved exactly is the azimuthal charge

    Lz = sum_i m_i n_i        ( [H, Lz] = 0 exact ).

We therefore give each bosonic mode i a U(1) charge equal to m_i * n_i, so the
global conserved quantum number is Lz.  A cubic term b_i^dag b_j^dag b_k then
carries net charge m_i + m_j - m_k, which TeNPy admits into the MPO iff it is
zero -- i.e. iff m1 + m2 = m3 -- so the selection rule is enforced by the
symmetry structure itself.  Working in a fixed-Lz sector (Lz = 0 for the
single-(2,0) quench) shrinks the MPS bonds and is what makes the climb to
l_max = 6, 7 (45, 60 modes) tractable.

The vertex COEFFICIENTS are imported verbatim from nhq.model.cubic_vertex_table
(single source of truth); this module only re-expresses them as an MPO and
validates the re-expression against the reference occupation-basis engine.
================================================================================
"""
import os
import sys
import numpy as np

from tenpy.networks.site import BosonSite
from tenpy.linalg.charges import ChargeInfo, LegCharge
from tenpy.models.lattice import Lattice
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.networks.mps import MPS

# --- vertex table (self-contained; byte-for-byte the paper's nhq.model) ---
from nhq_tenpy.vertex import cubic_vertex_table, omega_l


# ============================================================================
# modes
# ============================================================================
def build_modes(l_max, l_min=2):
    """Partial-wave modes (l, m), l = l_min..l_max, m = -l..l.
    Ordering: by l, then by m ascending.  M = (l_max+1)^2 - l_min^2 modes."""
    modes = []
    for l in range(l_min, l_max + 1):
        for m in range(-l, l + 1):
            modes.append((l, m))
    return modes


def mode_ordering(modes, scheme="paired_m"):
    """Return a permutation (list of original indices) giving the 1-D site
    order along the MPS chain.

    The choice matters exactly as orbital ordering matters in quantum-chemistry
    DMRG: a poor order spreads the physical correlations across every bond and
    inflates the bond dimension for a state that is really weakly entangled.
    Because the vertex conserves Lz (m1 + m2 = m3) and the (2,0) quench feeds
    dominantly the (m, -m) pair channel, keeping +m and -m PARTNERS adjacent
    localizes that correlation and is the low-entanglement order.

    scheme = "by_l"          : (l, m) order  -- l blocks, m within.
    scheme = "interleave_m"  : sort by (m, l) -- equal-m modes contiguous
                               (partners land at opposite ends: BAD for the
                               paired quench).
    scheme = "paired_m"      : sort by (|m|, l, sign) with +m before -m, m = 0
                               first -- each +m sits next to its -m partner.
                               Default.
    """
    idx = list(range(len(modes)))
    if scheme == "by_l":
        key = lambda i: (modes[i][0], modes[i][1])
    elif scheme == "interleave_m":
        key = lambda i: (modes[i][1], modes[i][0])
    elif scheme == "paired_m":
        key = lambda i: (abs(modes[i][1]), modes[i][0], modes[i][1] < 0)
    else:
        raise ValueError(scheme)
    return sorted(idx, key=key)


# ============================================================================
# sites: bosonic mode with U(1) charge Lz = m * n
# ============================================================================
def boson_site_lz(d, m):
    """BosonSite with local cutoff d (Nmax = d-1) whose conserved U(1) charge
    is m * n (so that the total charge over the register is Lz)."""
    chinfo = ChargeInfo([1], ['Lz'])
    site = BosonSite(Nmax=d - 1, conserve='N')
    qflat = (int(m) * np.arange(d)).reshape(-1, 1)
    leg = LegCharge.from_qflat(chinfo, qflat)
    site.change_charge(leg)
    return site


# ============================================================================
# the model
# ============================================================================
class GravitonCubicModel(MPOModel):
    """MPO model for H2^grav + H3^(hhh) on the ordered mode chain.

    Parameters
    ----------
    l_max : int              highest multipole (l_min = 2 fixed).
    d     : int              per-mode occupation cutoff (local dim).
    g_eff : float            reduced coupling g-tilde.
    order_scheme : str       site-ordering heuristic (see mode_ordering).
    R : float                horizon radius (=1).
    """

    def __init__(self, l_max, d=4, g_eff=12.0, order_scheme="paired_m",
                 l_min=2, R=1.0, vertex_reltol=0.0):
        modes = build_modes(l_max, l_min=l_min)
        perm = mode_ordering(modes, order_scheme)
        modes_ord = [modes[i] for i in perm]
        M = len(modes_ord)

        # remember the data
        self.modes_original = modes
        self.perm = perm                       # site s <- original mode perm[s]
        self.inv_perm = {orig: s for s, orig in enumerate(perm)}
        self.modes = modes_ord                 # in SITE order
        self.M = M
        self.d = d
        self.g_eff = g_eff
        self.R = R
        self.omega = np.array([omega_l(l, R) for (l, _) in modes_ord])
        self.ms = np.array([m for (_, m) in modes_ord])
        self.ls = np.array([l for (l, _) in modes_ord])

        # sites (in SITE order): ONE unit cell containing all M modes, so the
        # finite MPS is a length-M open chain in exactly this site order.
        sites = [boson_site_lz(d, m) for (_, m) in modes_ord]
        lat = Lattice([1], sites, bc="open", bc_MPS="finite")
        self.lat = lat

        cm = CouplingModel(lat)
        self._sites = sites
        self._combined_op_cache = set()

        # --- H2: onsite omega * n ------------------------------------------
        for s in range(M):
            cm.add_onsite_term(self.omega[s], s, 'N')

        # --- H3: build the reference vertex table in SITE order ------------
        # cubic_vertex_table works on whatever mode list it is given, so pass
        # the SITE-ordered modes; the returned indices are already site indices.
        table = cubic_vertex_table(modes_ord, g_eff=g_eff, R=R)
        # optional: drop couplings smaller than vertex_reltol * max|V| to shrink
        # the MPO bond dimension (a controlled, convergence-checkable
        # approximation for pushing to high l_max; reltol = 0 keeps it exact).
        self.vertex_reltol = vertex_reltol
        self.n_vertices_full = len(table)
        if vertex_reltol > 0.0 and table:
            vmax = max(abs(v) for v in table.values())
            cut = vertex_reltol * vmax
            table = {k: v for k, v in table.items() if abs(v) >= cut}
        self.table = table
        n_terms = 0
        for (i, j, k), V in table.items():
            # b_i^dag b_j^dag b_k  (operator-product order: leftmost applied last)
            self._add_term(cm, V, [('Bd', i), ('Bd', j), ('B', k)])
            # h.c. :  b_k^dag b_j b_i
            self._add_term(cm, np.conj(V), [('Bd', k), ('B', j), ('B', i)])
            n_terms += 1
        self.n_vertices = n_terms

        H_MPO = cm.calc_H_MPO()
        self.H_bond_dim = max(H_MPO.chi)
        MPOModel.__init__(self, lat, H_MPO)

    # ---- collision-safe multi-site term adder -----------------------------
    def _add_term(self, cm, strength, op_site_pairs):
        """Add strength * (product of on-site operators) to the CouplingModel,
        where op_site_pairs is a list of (opname, site) in OPERATOR-PRODUCT
        order (leftmost operator applied last).  Same-site operators are
        multiplied into a single combined operator (registered on the site);
        distinct sites are then passed strictly increasing.  Valid because
        bosonic operators on different sites commute (no JW string)."""
        if abs(strength) < 1e-16:
            return
        # group ops by site, preserving operator-product order within each site
        by_site = {}
        for op, s in op_site_pairs:
            by_site.setdefault(s, []).append(op)
        sites_sorted = sorted(by_site.keys())
        ops_for_call = []
        for s in sites_sorted:
            names = by_site[s]
            if len(names) == 1:
                ops_for_call.append(names[0])
            else:
                cname = "prod_" + "_".join(names)     # e.g. "prod_Bd_Bd_B"
                if (s, cname) not in self._combined_op_cache:
                    site = self._sites[s]
                    mat = site.get_op(names[0]).to_ndarray()
                    for nm in names[1:]:
                        mat = mat @ site.get_op(nm).to_ndarray()
                    site.add_op(cname, mat, need_JW=False)
                    self._combined_op_cache.add((s, cname))
                ops_for_call.append(cname)
        if len(sites_sorted) == 1:
            cm.add_onsite_term(strength, sites_sorted[0], ops_for_call[0])
        else:
            cm.add_multi_coupling_term(strength, sites_sorted, ops_for_call,
                                       'Id')

    # ---- convenience: build an MPS Fock (product) initial state -----------
    def fock_state(self, occ_by_original_mode):
        """Product MPS with given occupations, keyed by ORIGINAL (l,m) mode.
        occ_by_original_mode : dict {(l, m): n}.  Missing modes -> 0."""
        p_state = []
        for s in range(self.M):
            lm = self.modes[s]
            n = int(occ_by_original_mode.get(lm, 0))
            if n > self.d - 1:
                raise ValueError(f"occupation {n} exceeds cutoff d-1={self.d-1}")
            p_state.append(n)
        return MPS.from_product_state(self.lat.mps_sites(), p_state,
                                      bc=self.lat.bc_MPS)

    def single_graviton_state(self, l=2, m=0):
        return self.fock_state({(l, m): 1})
