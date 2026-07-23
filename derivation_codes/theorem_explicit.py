#!/usr/bin/env python3
# theorem_explicit.py
# ---------------------------------------------------------------------------
# Explicit term-by-term demonstration of the vanishing theorem (Appendix E).
#
# Traceless longitudinal 2D block, eta = diag(-1,+1):
#   tracelessness eta^{ab} h_ab = h_zz - h_tt = 0  =>  h_tt = h_zz = u, h_tz = v
#   g2 = [[-1+ku, kv],[kv, 1+ku]],   det g2 = -[1 - k^2(u^2 - v^2)].
#
# Shows, for u=u(t,z), v=v(t,z):
#   (1) sqrt(-g2) R2, expanded in k, is at EVERY order a total 2D derivative;
#       prints the explicit divergence potentials at O(k^1,2,3).
#   (2) the cubic vertex (Euler-Lagrange derivative of the cubic action)
#       vanishes identically in both polarizations.
#
# Requires sympy >= 1.12.  Runtime: under a minute.
# ---------------------------------------------------------------------------
import sympy as sp

t, z, k = sp.symbols('t z kappa')
u = sp.Function('u')(t, z)
v = sp.Function('v')(t, z)
X = [t, z]
g2 = sp.Matrix([[-1 + k*u, k*v], [k*v, 1 + k*u]])


def ricci_scalar_2d(g, X):
    gi = g.inv(); n = 2
    G = [[[0]*n for _ in range(n)] for _ in range(n)]
    for l in range(n):
        for a in range(n):
            for b in range(n):
                s = 0
                for m in range(n):
                    s += gi[l, m]*(sp.diff(g[m, a], X[b]) + sp.diff(g[m, b], X[a]) - sp.diff(g[a, b], X[m]))
                G[l][a][b] = sp.together(s/2)
    R = 0
    for a in range(n):
        for b in range(n):
            e = 0
            for l in range(n):
                e += sp.diff(G[l][a][b], X[l]) - sp.diff(G[l][a][l], X[b])
                for m in range(n):
                    e += G[l][l][m]*G[m][a][b] - G[l][b][m]*G[m][a][l]
            R += gi[a, b]*e
    return R


def order(expr, n):
    return sp.expand(sp.diff(expr, k, n).subs(k, 0)/sp.factorial(n))


def euler_lagrange(Lag, f):
    """EL derivative dL/df - d_i(dL/df_i) + d_i d_j(dL/df_ij) for a 2D field."""
    ft, fz = sp.diff(f, t), sp.diff(f, z)
    ftt, ftz, fzz = sp.diff(f, t, t), sp.diff(f, t, z), sp.diff(f, z, z)
    F, Ft, Fz, Ftt, Ftz, Fzz = sp.symbols('F Ft Fz Ftt Ftz Fzz')
    fwd = {ftt: Ftt, ftz: Ftz, fzz: Fzz, ft: Ft, fz: Fz, f: F}
    bwd = {F: f, Ft: ft, Fz: fz, Ftt: ftt, Ftz: ftz, Fzz: fzz}
    Ls = Lag.subs(fwd, simultaneous=True)
    dF, dFt, dFz = sp.diff(Ls, F), sp.diff(Ls, Ft), sp.diff(Ls, Fz)
    dFtt, dFtz, dFzz = sp.diff(Ls, Ftt), sp.diff(Ls, Ftz), sp.diff(Ls, Fzz)
    el = (dF.subs(bwd)
          - sp.diff(dFt.subs(bwd), t) - sp.diff(dFz.subs(bwd), z)
          + sp.diff(dFtt.subs(bwd), t, t) + sp.diff(dFtz.subs(bwd), t, z)
          + sp.diff(dFzz.subs(bwd), z, z))
    return sp.simplify(sp.expand(el))


R2 = ricci_scalar_2d(g2, X)
L = sp.together(sp.sqrt(-g2.det())*R2)
L1, L2, L3 = order(L, 1), order(L, 2), order(L, 3)

print("Order-by-order density  [sqrt(-g2) R2]_n  (u=h_tt=h_zz, v=h_tz):\n")
print("  L1 =", sp.simplify(L1))
print("     = d_t(u_t - v_z) + d_z(u_z - v_t)          [total derivative]\n")
print("  L2 =", sp.simplify(L2))
print("     = -[d_t(u v_z) - d_z(u v_t)]  (Jacobian)   [total derivative]\n")
print("  L3  (13 monomials):")
for term in sp.Add.make_args(sp.expand(L3)):
    print("      ", term)

# explicit cubic divergence potential
P3 = sp.Rational(1, 2)*(u**2 - v**2)*(sp.diff(u, t) - sp.diff(v, z))
Q3 = sp.Rational(1, 2)*(u**2 - v**2)*(sp.diff(u, z) - sp.diff(v, t))
res3 = sp.simplify(sp.expand(L3 - (sp.diff(P3, t) + sp.diff(Q3, z))))
print("\n  L3 - [ d_t P3 + d_z Q3 ]  with")
print("      P3 = 1/2 (u^2 - v^2)(u_t - v_z),  Q3 = 1/2 (u^2 - v^2)(u_z - v_t)")
print("      =", res3, "   [cubic term is an exact total derivative]\n")

# cubic vertex = EL derivative -> must vanish
print("Cubic vertex (Euler-Lagrange derivative of the cubic action):")
print("  delta/delta u :", euler_lagrange(L3, u))
print("  delta/delta v :", euler_lagrange(L3, v))
print("  => the traceless longitudinal self-vertex is identically zero.")
