#!/usr/bin/env python3
"""Cubic momentum-space vertex from L3, symmetrized over the three legs,
reduced on the conservation surface p3 = -p1-p2."""
import sympy as sp, pickle

t, z = sp.symbols('t z', real=True)
a = sp.Function('a')(t, z); b = sp.Function('b')(t, z)
c = sp.Function('c')(t, z); k = sp.Function('k')(t, z)

with open('L23.pkl','rb') as f:
    L3 = pickle.load(f)['L3']

w = sp.symbols('w1:4'); q = sp.symbols('q1:4')
A = sp.symbols('A1:4'); B = sp.symbols('B1:4')
C = sp.symbols('C1:4'); Kk = sp.symbols('K1:4')
E = [sp.exp(sp.I*(w[j]*t + q[j]*z)) for j in range(3)]

sub = {a: sum(A[j]*E[j] for j in range(3)),
       b: sum(B[j]*E[j] for j in range(3)),
       c: sum(C[j]*E[j] for j in range(3)),
       k: sum(Kk[j]*E[j] for j in range(3))}
expr = L3
for f, s in sub.items():
    expr = expr.replace(f, s)
expr = sp.expand(expr.doit())

allamps = [x for j in range(3) for x in (A[j],B[j],C[j],Kk[j])]
e = sp.expand(expr / (E[0]*E[1]*E[2])).subs({t:0, z:0})
p = sp.Poly(e, *allamps)
V3 = sp.Integer(0)
for mono, coeff in zip(p.monoms(), p.coeffs()):
    degs = [sum(mono[4*i:4*i+4]) for i in range(3)]
    if all(dg == 1 for dg in degs):
        term = coeff
        for x, m in zip(allamps, mono):
            term *= x**m
        V3 += term
V3 = sp.expand(V3)
print("V3 multilinear terms:", len(V3.args))

# conservation surface: p3 = -p1 - p2
V3c = sp.expand(V3.subs({w[2]:-w[0]-w[1], q[2]:-q[0]-q[1]}))

with open('V3.pkl','wb') as f:
    pickle.dump({'V3': V3, 'V3c': V3c}, f)

# ---------------- THEOREM CHECK ----------------
# traceless longitudinal sector only: trace = -a+c = 0 -> c=a ; K=0
# (in (t,z) coords a traceless symmetric 2-tensor has T_tt=T_zz=P, T_tz=B)
P1,P2,P3 = sp.symbols('P1:4')
subT = {C[0]:A[0], C[1]:A[1], C[2]:A[2], Kk[0]:0, Kk[1]:0, Kk[2]:0}
V3_traceless = sp.simplify(sp.expand(V3c.subs(subT)))
print("\n=== V3 restricted to traceless longitudinal polarizations (K=0) ===")
print(V3_traceless)
