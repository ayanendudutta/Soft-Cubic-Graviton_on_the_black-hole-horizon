#!/usr/bin/env python3
"""Two-engine verification: exact nonlinear GR on a periodic 2D grid (embedded
in 4D) vs the derived L3 vertex.  Fields: random low-wavevector plane waves on
zero-sum-closed triads, which keeps the triple products below the Nyquist
frequency and avoids aliasing (the discipline of App. D of the paper).

Engine A: exact sqrt(-g)R with finite-kappa Taylor extraction (Richardson).
Engine B: symbolic L2/L3 evaluated on the same configuration.
Also: the THEOREM check (traceless config -> cubic coefficient = 0)."""
import numpy as np, sympy as sp, pickle, itertools

rng = np.random.default_rng(7)
N = 16
L = 2*np.pi
tg, zg = np.meshgrid(np.arange(N)*L/N, np.arange(N)*L/N, indexing='ij')

# low-k plane waves closed under zero-sum triads; the conjugate wavevectors
# are supplied automatically by the cosines
kvecs = [(1,0),(0,1),(1,1),(-1,1),(2,1),(1,2)]

def rand_field():
    f = np.zeros((N,N))
    for (kw,kq) in kvecs:
        amp = rng.normal(scale=0.5); ph = rng.uniform(0,2*np.pi)
        f += amp*np.cos(kw*tg + kq*zg + ph)
    return f

def make_cfg(traceless=False):
    a = rand_field(); b = rand_field(); c = rand_field(); k = rand_field()
    if traceless:
        c = a.copy(); k = np.zeros_like(k)
    return a,b,c,k

def spectral_d(f, axis):
    kf = np.fft.fftfreq(N, d=1.0/N)  # integer wavenumbers
    F = np.fft.fftn(f)
    shape = [1,1]; shape[axis] = N
    F = F * (1j*kf.reshape(shape))
    return np.real(np.fft.ifftn(F))

def exact_action_density(a,b,c,k,kap):
    """exact sqrt(-g)R, box-averaged, for the 4D block metric"""
    # metric components (functions on grid)
    gtt = -1 + kap*a; gtz = kap*b; gzz = 1 + kap*c; gAA = 1 + kap*k
    # 4x4 metric field: indices 0..3 = t,z,y1,y2
    g = np.zeros((4,4,N,N))
    g[0,0]=gtt; g[0,1]=gtz; g[1,0]=gtz; g[1,1]=gzz; g[2,2]=gAA; g[3,3]=gAA
    # inverse: block 2x2 for (t,z), diagonal transverse
    det2 = gtt*gzz - gtz**2
    gi = np.zeros_like(g)
    gi[0,0] = gzz/det2; gi[1,1] = gtt/det2; gi[0,1] = -gtz/det2; gi[1,0] = -gtz/det2
    gi[2,2] = 1/gAA; gi[3,3] = 1/gAA
    detg = det2*gAA*gAA  # (negative)
    sqg = np.sqrt(-detg)
    # derivatives: only w.r.t. t (axis0), z (axis1)
    dg = np.zeros((2,4,4,N,N))
    for mu in range(4):
        for nu in range(4):
            if np.any(g[mu,nu]):
                dg[0,mu,nu] = spectral_d(g[mu,nu],0)
                dg[1,mu,nu] = spectral_d(g[mu,nu],1)
    # Christoffels
    Gam = np.zeros((4,4,4,N,N))
    for rho in range(4):
        for mu in range(4):
            for nu in range(mu,4):
                s = np.zeros((N,N))
                for sig in range(4):
                    t1 = dg[nu,sig,mu] if nu<2 else 0.0
                    t2 = dg[mu,sig,nu] if mu<2 else 0.0
                    t3 = dg[0,mu,nu] if sig==0 else (dg[1,mu,nu] if sig==1 else 0.0)
                    s += gi[rho,sig]*( (t1 if isinstance(t1,np.ndarray) else 0) \
                                     + (t2 if isinstance(t2,np.ndarray) else 0) \
                                     - (t3 if isinstance(t3,np.ndarray) else 0) )
                Gam[rho,mu,nu] = s/2; Gam[rho,nu,mu] = s/2
    # Ricci
    Ric = np.zeros((4,4,N,N))
    for mu in range(4):
        for nu in range(mu,4):
            s = np.zeros((N,N))
            for rho in range(4):
                if rho<2:
                    s += spectral_d(Gam[rho,mu,nu],rho)
                if nu<2:
                    s -= spectral_d(Gam[rho,mu,rho],nu)
                for lam in range(4):
                    s += Gam[rho,rho,lam]*Gam[lam,mu,nu] - Gam[rho,nu,lam]*Gam[lam,mu,rho]
            Ric[mu,nu]=s; Ric[nu,mu]=s
    R = np.zeros((N,N))
    for mu in range(4):
        for nu in range(4):
            R += gi[mu,nu]*Ric[mu,nu]
    return np.mean(sqg*R)

def taylor_coeffs(a,b,c,k):
    """even/odd split + Richardson on F(kap)=<sqrt(-g)R> -> s2, s3

    F(kap) = s2 kap^2 + s3 kap^3 + s4 kap^4 + s5 kap^5 + ...
    The even/odd split separates the two parities exactly, so each Richardson
    elimination acts on a single parity:
      even(kap) = s2 kap^2 + s4 kap^4  ->  16 even(h) -   even(2h) = 12 s2 h^2
      odd (kap) = s3 kap^3 + s5 kap^5  ->  32 odd (h) -   odd (2h) = 24 s3 h^3
    The odd combination must carry the weight 32, not 8: with the weight 8 the
    s3 contributions cancel and the expression returns -4 s5 h^2 instead."""
    h = 0.02
    kaps = [h, 2*h]
    F = {}
    for s in (+1,-1):
        for kk in kaps:
            F[(s,kk)] = exact_action_density(a,b,c,k, s*kk)
    even = lambda kk: (F[(1,kk)]+F[(-1,kk)])/2   # s2 k^2 + s4 k^4
    odd  = lambda kk: (F[(1,kk)]-F[(-1,kk)])/2   # s3 k^3 + s5 k^5
    s2 = (16*even(h) - even(2*h))/(12*h**2)
    s3 = (32*odd(h) - odd(2*h))/(24*h**3)
    return s2, s3

# ---------- Engine B: evaluate symbolic L2, L3 on grid ----------
with open('L23.pkl','rb') as f:
    D = pickle.load(f)
t, z = sp.symbols('t z', real=True)
af = sp.Function('a')(t,z); bf = sp.Function('b')(t,z)
cf = sp.Function('c')(t,z); kf = sp.Function('k')(t,z)

def eval_L(Lexpr, a,b,c,k):
    """evaluate differential polynomial on grid via spectral derivatives"""
    fields = {'a':a,'b':b,'c':c,'k':k}
    cache = {}
    def getd(name, nt, nz):
        key=(name,nt,nz)
        if key in cache: return cache[key]
        f = fields[name]
        for _ in range(nt): f = spectral_d(f,0)
        for _ in range(nz): f = spectral_d(f,1)
        cache[key]=f; return f
    total = np.zeros((N,N))
    for term in sp.Add.make_args(sp.expand(Lexpr)):
        coeff = 1.0; arr = np.ones((N,N))
        for fac in sp.Mul.make_args(term):
            if fac.is_Number:
                coeff *= float(fac); continue
            base, expo = (fac.base, int(fac.exp)) if fac.is_Pow else (fac, 1)
            if isinstance(base, sp.Derivative):
                name = base.expr.func.__name__
                nt = sum(cnt for (v,cnt) in base.variable_count if v==t)
                nz = sum(cnt for (v,cnt) in base.variable_count if v==z)
                val = getd(name, nt, nz)
            else:
                name = base.func.__name__
                val = getd(name, 0, 0)
            for _ in range(expo): arr = arr*val
        total += coeff*arr
    return np.mean(total)

# --- BEGIN REPORT ----------------------------------------------------------
# Everything above this marker is definition-only and is re-used verbatim by
# run_verify2.py, which splits this file on the marker and executes the prefix.

print(f"{'seed':>4} {'s2_exact':>13} {'s2_L2':>13} {'s3_exact':>13} {'s3_L3':>13} {'rel err s3':>11}")
for seed in range(4):

    rng = np.random.default_rng(seed)
    a,b,c,k = make_cfg()
    s2, s3 = taylor_coeffs(a,b,c,k)
    L2v = eval_L(D['L2'], a,b,c,k)
    L3v = eval_L(D['L3'], a,b,c,k)
    print(f"{seed:>4} {s2:>13.8f} {L2v:>13.8f} {s3:>13.8f} {L3v:>13.8f} {abs(s3-L3v)/max(1e-30,abs(s3)):>11.2e}")

# Traceless configurations: BOTH the quadratic and the cubic coefficient
# vanish, because in the two-dimensional traceless block the quadratic density
# is a null Lagrangian and the cubic density an exact total derivative
# (the order-by-order statement of the vanishing theorem).  The two columns are
# therefore reported as absolute machine zeros; their ratio is meaningless.
print("\n=== THEOREM: traceless configs (c=a, K=0) -> s2 and s3 must vanish ===")
print(f"{'seed':>4} {'s2_exact':>14} {'s3_exact':>14}")
for seed in range(4):
    rng = np.random.default_rng(100+seed)
    a,b,c,k = make_cfg(traceless=True)
    s2, s3 = taylor_coeffs(a,b,c,k)
    print(f"{seed:>4} {s2:>14.6e} {s3:>14.6e}")
