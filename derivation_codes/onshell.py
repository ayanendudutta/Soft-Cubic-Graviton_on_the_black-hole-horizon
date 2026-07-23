#!/usr/bin/env python3
"""On-shell three-point weight W(l1,l2,l3):
   - vertex function = IBP-unambiguous V3c (conservation surface)
   - kinematics: rest-frame legs, p_i.p_j = -w_i w_j, w_i = sqrt(lam_i) (mu=1)
   - polarizations: GGV residue mode  eps = eta + ptilde(k) at k^2=-lam:
       T: P_i = lam_i/(lam_i+1), B_i = 0 ;  K_i = -1 ;  Hbar_i = -2/(lam_i+1)
   Also: Hbar=0 variant (leading 1/lambda) to quantify the trace-mode correction."""
import sympy as sp, pickle

w = sp.symbols('w1:4'); q = sp.symbols('q1:4')
A = sp.symbols('A1:4'); B = sp.symbols('B1:4')
C = sp.symbols('C1:4'); Kk = sp.symbols('K1:4')

V3c = pickle.load(open('V3.pkl','rb'))['V3c']

lam = sp.symbols('lam1:4', positive=True)
om  = [sp.sqrt(lam[i]) for i in range(3)]

def W_of(lams=None, exact_Hbar=True):
    subs = {}
    # kinematics: legs 1,2 at rest, creation (w=+omega); q's zero
    subs[q[0]] = 0; subs[q[1]] = 0
    subs[w[0]] = om[0]; subs[w[1]] = om[1]
    for i in range(3):
        P_i  = lam[i]/(lam[i]+1)
        Hb_i = -sp.Integer(2)/(lam[i]+1) if exact_Hbar else sp.Integer(0)
        subs[A[i]] = P_i - Hb_i/2
        subs[C[i]] = P_i + Hb_i/2
        subs[B[i]] = 0
        subs[Kk[i]] = -1
    e = V3c.subs(subs, simultaneous=True)
    e = sp.simplify(sp.expand(e))
    if lams is not None:
        e = e.subs({lam[i]: lams[i] for i in range(3)})
        e = sp.simplify(e)
    return e

print("=== W(lam1,lam2,lam3), exact residue polarization ===")
Wgen = W_of()
Wgen = sp.simplify(Wgen)
print(Wgen)

print("\n=== equal-lambda W(lam,lam,lam) ===")
lam0 = sp.symbols('lam', positive=True)
Weq = sp.simplify(Wgen.subs({lam[0]:lam0, lam[1]:lam0, lam[2]:lam0}))
print(sp.factor(sp.simplify(Weq)))
print("large-lam expansion:", sp.series(Weq, lam0, sp.oo, 3))

print("\n=== numeric table (mu=1) ===")
import itertools
lam_of = lambda l: l*l + l + 1
print(f"{'(l1,l2,l3)':>12} {'W exact':>14} {'W(Hbar=0)':>14}")
for (l1,l2,l3) in [(2,2,2),(2,2,4),(2,4,4),(4,4,4),(2,4,6),(2,2,6),(6,6,6),(2,6,8)]:
    Wv  = float(W_of([lam_of(l1),lam_of(l2),lam_of(l3)], True))
    Wv0 = float(W_of([lam_of(l1),lam_of(l2),lam_of(l3)], False))
    print(f"{str((l1,l2,l3)):>12} {Wv:>14.6f} {Wv0:>14.6f}")
with open('W.pkl','wb') as f:
    pickle.dump({'Wgen': Wgen, 'Weq': Weq}, f)
