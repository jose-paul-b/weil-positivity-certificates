"""(A') Rigorous truncated Weil matrix on a POLYNOMIAL basis (orthonormal Legendre, one parity), and rigorous
upper bounds on the semilocal threshold.  Same normalisation as weil_mp.py / semilocal_hodge.py (real-space form):
  Q_P(f) = 2 F(i/2) F(-i/2) - (gamma + log pi) g(0) - g(0) log(1 - e^{-2L}) + int_0^{2L} [g(0) e^{-t} - e^{-t/4} g(t/2)]/(1 - e^{-t}) dt
           - 2 sum_{p <= P, k log p < L} (log p) p^{-k/2} g(k log p),       L = 2a = autocorrelation support.
Basis phi_m(x) = sqrt((2m+1)/(2a)) P_m(x/a) on [-a, a] (orthonormal), m = parity, parity+2, ..., <= D.
  g_mn(u) = int phi_m(x) phi_n(x-u) dx = s_mn h_mn(u/a),  s_mn = sqrt((2m+1)(2n+1))/2,
  h_mn(w) = int_{w-1}^{1} P_m(s) P_n(s-w) ds  (0 <= w <= 2), an EXACT rational polynomial (Taylor in w):
          = sum_j (-w)^j/j! [A_j(1) - A_j(w-1)],  A_j' = P_m P_n^{(j)}.     Quadratic form uses (h_mn + h_nm)/2.
  pole: F_m(+-i/2) = 2a sqrt((2m+1)/(2a)) i_m(-+a/2)   (modified spherical Bessel, i_m(-x) = (-1)^m i_m(x)).
  arch integral with H = sum eta_k w^k:  eta_0 K_0 - sum_{k>=1} eta_k K_k,
     K_0 = -(3/4) int_0^{2L} e^{-t/8} shc(3t/8)/shc(t/2) dt,  K_k = (2a)^{-k} int_0^{2L} e^{t/4} t^{k-1}/shc(t/2) dt,
     shc(x) = sinh(x)/x = sinc(ix): integrands meromorphic (poles at 2 pi i Z \\ 0 only), acb.integral is rigorous.
Every matrix entry is an arb ball; for any exact vector v, v^T M v is a ball containing Q_P(sum v_m phi_m).
Usage: python poly_upper.py P L D parity [--scan lo hi n]    (L rational, = 2a)"""
import sys, os, json, time, math, pickle
from fractions import Fraction
import mpmath as mp
from sympy import primerange
from flint import arb, acb, arb_mat, fmpq_poly, fmpq, ctx
here = os.path.dirname(os.path.abspath(__file__))
PREC = 1600

def h_polys(D, parity):
    fn = os.path.join(here, f"hpoly_D{D}_p{parity}.pkl")
    if os.path.exists(fn):
        return pickle.load(open(fn, "rb"))
    ms = list(range(parity, D + 1, 2)); w = fmpq_poly([0, 1]); wm1 = fmpq_poly([-1, 1])
    Pl = {m: fmpq_poly.legendre_p(m) for m in ms}
    H = {}
    for m in ms:
        for n in ms:
            tot = fmpq_poly([0]); dn = Pl[n]; fact = 1
            for j in range(n + 1):
                A = (Pl[m]*dn).integral()
                term = A(fmpq(1)) - A(wm1)
                mono = fmpq_poly([0]*j + [fmpq((-1)**j, fact)])
                tot += mono*term
                dn = dn.derivative(); fact *= (j + 1)
            H[(m, n)] = tot
    S = {}
    for i, m in enumerate(ms):
        for n in ms[i:]:
            S[(m, n)] = [(c.p, c.q) for c in ((H[(m, n)] + H[(n, m)])/2).coeffs()]
    pickle.dump((ms, S), open(fn, "wb"))
    return ms, S

def qarb(c): return arb(int(c[0]))/int(c[1])
def shc(x): return (acb(0, 1)*x).sinc()    # sinh(x)/x = sinc(ix), entire

def moments(Lfr, kmax):
    ctx.prec = PREC
    b = arb(Lfr.numerator)/Lfr.denominator; a = b/2; twoL = 2*b
    tol = arb(2)**(-(PREC - 80))
    K0 = acb.integral(lambda t, _: -acb(3)/4*(-t/8).exp()*shc(3*t/8)/shc(t/2), 0, acb(twoL), rel_tol=tol, abs_tol=tol)
    K = [K0.real]
    for k in range(1, kmax + 1):
        Ik = acb.integral(lambda t, _: (t/4).exp()*t**(k - 1)/shc(t/2), 0, acb(twoL), rel_tol=tol, abs_tol=tol)
        K.append(Ik.real/(2*a)**k)
    return K

def i_sph(m, x):
    return (arb.pi()/(2*x)).sqrt()*x.bessel_i(arb(m) + arb(1)/2)

def build_M(P, Lfr, D, parity):
    ms, S = h_polys(D, parity)
    ctx.prec = PREC
    b = arb(Lfr.numerator)/Lfr.denominator; a = b/2
    kmax = max(len(c) for c in S.values()) - 1
    K = moments(Lfr, kmax)
    us = []
    for q in primerange(2, P + 1):
        kk = 1
        while True:
            u = kk*arb(int(q)).log()
            if u < b: us.append((u/a, arb(int(q)).log()/arb(int(q))**(arb(kk)/2))); kk += 1
            elif u > b: break
            else: raise ValueError("window edge too close to k log p")
    c0 = arb.const_euler() + arb.pi().log() + (1 - (-2*b).exp()).log()
    Fp = {m: 2*a*((2*m + 1)/(2*a)).sqrt()*i_sph(m, a/2) for m in ms}      # F_m(-i/2); F_m(i/2) = (-1)^m Fp
    N = len(ms); M = arb_mat(N, N)
    for i, m in enumerate(ms):
        for k2 in range(i, N):
            n = ms[k2]; eta = [qarb(c) for c in S[(m, n)]]
            s = (arb((2*m + 1)*(2*n + 1))).sqrt()/2
            arch = -c0*eta[0] + eta[0]*K[0] - sum((eta[k]*K[k] for k in range(1, len(eta))), arb(0))
            prime = arb(0)
            for w_, wt in us:
                hv = arb(0)
                for e in reversed(eta): hv = hv*w_ + e
                prime += wt*hv
            pole = 2*(-1)**m*Fp[m]*Fp[n]
            M[i, k2] = pole + s*(arch - 2*prime)
            M[k2, i] = M[i, k2]
    return ms, M, us

def to_mp(x):
    m_, e_ = x.mid().man_exp(); return mp.ldexp(mp.mpf(int(m_)), int(e_))
def from_mp(v):
    sg, man, ex, _ = mp.mpf(v)._mpf_
    if man == 0: return arb(0)
    return (-1)**sg*arb(int(man))*arb(2)**int(ex)

def lowest(M, dps=100):
    mp.mp.dps = dps; N = M.nrows()
    Am = mp.matrix(N, N)
    for i in range(N):
        for k in range(N): Am[i, k] = to_mp(M[i, k])
    E, V = mp.eigsy(Am)
    i0 = min(range(N), key=lambda i: E[i])
    return E[i0], [V[i, i0] for i in range(N)]

def quad_form(M, v):
    ctx.prec = PREC
    vb = [from_mp(mp.mpf(mp.nstr(x, 60))) for x in v]        # exact dyadic vector (rounded to ~200 bits)
    N = len(vb); tot = arb(0); nrm = arb(0)
    for i in range(N):
        nrm += vb[i]**2
        for k in range(N): tot += vb[i]*vb[k]*M[i, k]
    return tot, nrm, vb

def check_float(P, Lfr, ms):
    """independent float64 construction of the same matrix: phi_m pointwise (numpy Legendre), g_mn(u) by
    Gauss-Legendre in x, pole by Gauss-Legendre, archimedean part by scipy.quad of weil_mp.row's formula."""
    import numpy as np
    from numpy.polynomial import legendre as npl
    from scipy.integrate import quad
    a = Lfr.numerator/Lfr.denominator/2; L = 2*a
    xg, wg = npl.leggauss(160)
    def phi(m, x):
        c = np.zeros(m + 1); c[m] = 1
        return np.where(np.abs(x) <= a*(1 + 1e-15), np.sqrt((2*m + 1)/(2*a))*npl.legval(np.clip(x/a, -1, 1), c), 0.0)
    def gmn(m, n, u):                            # int phi_m(x) phi_n(x-u) dx, 0 <= u <= L
        lo, hi = u - a, a
        if hi <= lo: return 0.0
        x = (hi - lo)/2*xg + (hi + lo)/2
        return (hi - lo)/2*np.sum(wg*phi(m, x)*phi(n, x - u))
    gs = lambda m, n, u: (gmn(m, n, u) + gmn(n, m, u))/2
    us = []
    for q in primerange(2, P + 1):
        kk = 1
        while kk*math.log(q) < L: us.append((kk*math.log(q), math.log(q)/q**(kk/2))); kk += 1
    N = len(ms); Mf = np.zeros((N, N)); x = a*xg
    for i, m in enumerate(ms):
        for k, n in enumerate(ms):
            if k < i: Mf[i, k] = Mf[k, i]; continue
            G0 = gs(m, n, 0.0)
            Fpm = a*np.sum(wg*phi(m, x)*np.exp(-x/2)); Fmm = a*np.sum(wg*phi(m, x)*np.exp(x/2))
            Fpn = a*np.sum(wg*phi(n, x)*np.exp(-x/2)); Fmn = a*np.sum(wg*phi(n, x)*np.exp(x/2))
            integ = lambda t: (G0*math.exp(-t) - math.exp(-t/4)*gs(m, n, t/2))/(-math.expm1(-t)) if t > 0 else 0.0
            A = -0.5772156649015329*G0 + quad(integ, 0, 2*L, limit=400, epsabs=1e-14, epsrel=1e-13)[0] - G0*math.log(-math.expm1(-2*L))
            Mf[i, k] = Fpm*Fmn + Fmm*Fpn - math.log(math.pi)*G0 + A - 2*sum(w*gs(m, n, u) for u, w in us)
    return Mf

if __name__ == "__main__":
    P = int(sys.argv[1]); Lfr = Fraction(sys.argv[2]); D = int(sys.argv[3]); parity = int(sys.argv[4])
    t0 = time.time()
    if "--scan" in sys.argv:          # numerical lambda_min (mid-point matrix) over a grid of windows
        i = sys.argv.index("--scan"); lo, hi, n = Fraction(sys.argv[i+1]), Fraction(sys.argv[i+2]), int(sys.argv[i+3])
        for r in range(n + 1):
            Lr = lo + (hi - lo)*r/n
            ms, M, _ = build_M(P, Lr, D, parity); lam, _ = lowest(M, 90)
            print(f"P={P} D={D} parity={parity} L={float(Lr):.6f}: lambda_min = {mp.nstr(lam, 6)}", flush=True)
        sys.exit()
    ms, M, us = build_M(P, Lfr, D, parity)
    lam, v = lowest(M)
    Qb, nrm, vb = quad_form(M, v)
    maxrad = max((arb(M[i, k].rad()) for i in range(M.nrows()) for k in range(M.nrows())), key=lambda r: float(r.log()) if r > 0 else -1e9)
    maxrad = maxrad.str(3)
    ok = bool(Qb < 0)
    lines = [f"P = {P}, L_up = {Lfr} = {float(Lfr):.6f}, Legendre basis degrees {ms[0]}..{ms[-1]} ({'even' if parity == 0 else 'odd'} f), {len(ms)} functions",
             f"comb (u_n/a, w_n): {[(u.str(8), w.str(8)) for u, w in us]}",
             f"max entry radius of rigorous matrix: {maxrad}",
             f"lambda_min(mid matrix) = {mp.nstr(lam, 12)}",
             f"ball-arithmetic Q_P(f_v) = {Qb.str(15)}   (||f_v||^2 = {nrm.str(12)})"]
    if "--check" in sys.argv:
        import numpy as np
        Mf = check_float(P, Lfr, ms); Nn = len(ms)
        Mmid = np.array([[float(M[i, k].mid()) for k in range(Nn)] for i in range(Nn)])
        vf = np.array([float(x) for x in v])
        lines.append(f"independent float64 matrix: max |M_rig - M_float| = {np.max(np.abs(Mmid - Mf)):.2e};  "
                     f"v^T M_float v = {vf @ Mf @ vf:.6e}, lambda_min(M_float) = {np.linalg.eigvalsh(Mf)[0]:.4e}")
    lines.append(f"RESULT: {'CERTIFIED NEGATIVE: Q_P not positive for supp f in [-L/2, L/2]' if ok else 'NOT certified'}   ({time.time()-t0:.0f}s)")
    for s in lines: print(s, flush=True)
    json.dump(dict(P=P, L_up=str(Lfr), L_up_float=float(Lfr), D=D, parity=parity, degrees=ms, lam_mid=float(lam),
                   Q_ball=Qb.str(30, radius=True), Q_upper=Qb.upper().str(20), certified_negative=ok,
                   coeffs=[mp.nstr(x, 60) for x in v], log=lines),
              open(os.path.join(here, "results", f"polyupper_P{P}_L{Lfr.numerator}_{Lfr.denominator}_D{D}_p{parity}.json"), "w"), indent=1)
