"""Rigorous (interval-arithmetic) certificate of Weil positivity on a compact window, following Zhu's
one-stroke reduction (arXiv 2608.24827, Thm 1.1), with all constants re-derived here.

Setting.  f real, supp f ⊆ [−L, L] (L = half-width; f*f~ has support 2L), F(t) = ∫ f(x) e^{itx} dx.
  Q(f) = 2·F(i/2)·F(−i/2) + (1/2π) ∫_R |F(t)|² Ψ(t) dt,
  Ψ(t) = Re ψ(1/4 + it/2) − log π − Σ_{log n < 2L} (2Λ(n)/√n) cos(t log n)          (ψ = digamma)
  (same normalisation as weil_mp.py, checked against Σ_γ 2|F(γ)|² in spectral_identity.py)

ENVELOPE (Binet's second formula, Re z > 0):  ψ(z) = log z − 1/(2z) − 2∫_0^∞ s ds/((s² + z²)(e^{2πs} − 1)).
  z = 1/4 + iy, y = t/2 > 0:  |s² + z²| ≥ |Im(s² + z²)| = y/2, and ∫_0^∞ s/(e^{2πs} − 1) ds = 1/24, so
  |integral term| ≤ 2·(2/y)·(1/24) = 1/(6y) = 1/(3t);   Re 1/(2z) = (1/8)/|z|² ≤ 1/(2t²);   Re log z = log|z| ≥ log(t/2).
  ⇒ Re ψ(1/4 + it/2) ≥ log(t/2) − 1/(2t²) − 1/(3t) ≥ log(t/2) − 1/t   for t ≥ 3/4.
  Comb ≥ −A_L, A_L = Σ_{log n < 2L} 2Λ(n)/√n.  So Ψ(t) ≥ log(t/2π) − 1/t − A_L (increasing in t), hence
  Ψ ≥ β on [T, ∞) with β := log(T/2π) − 1/T − A_L > 0.
REDUCTION (f even or odd ⇒ |F|² even; Plancherel (1/π)∫_0^∞|F|² = ‖f‖²):
  Q(f) ≥ R(f) := s·2·F(i/2)² + (1/π)∫_0^T (Ψ − β)|F|² dt + β‖f‖²,     s = +1 even f, −1 odd f.
BASIS: orthonormal Legendre T_m(x) = sqrt((2m+1)/2) P_m(x/L)/sqrt(L), m of one parity;
  F_m(t) = 2 i^m sqrt(L(2m+1)/2) j_m(Lt)  ->  real profile g_m(t) = 2 (−1)^{⌊m/2⌋} sqrt(L(2m+1)/2) j_m(Lt)
  (for odd m, F_m = i·g_m; the common factor i drops out of |Σ c_m F_m|²);
  pole: |F_m(i/2)| = 2 sqrt(L(2m+1)/2) i_m(L/2), and F(i/2)² = (Σ c_m p_m)² with p_m = ±that (signs consistent).
  Matrix of R:  β I + s·2 p pᵀ + C,   C_mn = (1/π)∫_0^T (Ψ − β) g_m g_n dt.
LEADING BLOCK A (first N modes): composite Gauss–Legendre in ball arithmetic (nodes/weights are arb enclosures), plus
  the Bernstein-ellipse error bound (Trefethen, ATAP Thm 19.3: |I − I_n| ≤ (64/15) M ρ^{−2n}/(ρ² − 1) on [−1,1])
  added to each entry as a radius. Ellipse semi-minor axis bz < 1/2 keeps Re(1/4 ± iz/2) > 0 (ψ analytic);
  on it |g_m| ≤ e^{L bz}·‖T_m‖_1 ≤ sqrt(2L) e^{L bz}, |cos(z log n)| ≤ cosh(2L bz), |ψ| bounded by covering boxes.
TAIL (modes ≥ N):  |j_n(x)| ≤ x^n/(2n+1)!! (x ≥ 0),  i_n(x) ≤ x^n e^{x²/2}/(2n+1)!!,  Σ_all g_m(t)² ≤ 2L (Bessel ineq.)
  |C_mn| ≤ (M/π) a_m a_n, a_m² = ∫_0^T g_m², M = sup_[0,T]|Ψ − β| (Re ψ(1/4+it/2) increases in t);
  ‖C_AD‖ ≤ (M/π) sqrt(2LT) sqrt(T S_D),  ‖C_DD‖ ≤ (M/π) T S_D,  S_D = Σ_{n∈D} 2L(2n+1)((LT)^n/(2n+1)!!)²;
  ‖p_A‖² ≤ ∫_{−L}^{L} e^{−x} dx = 2 sinh L,  ‖p_D‖² ≤ Σ_{n∈D} 2L(2n+1)((L/2)^n e^{L²/8}/(2n+1)!!)².
  Block lemma: [[A, B],[Bᵀ, D]] ⪰ [[A − ‖B‖, 0],[0, D − ‖B‖]] ⇒
  λ_min(R) ≥ min(λ_min(A), β − ‖C_DD‖ − [s<0]·2‖p_D‖²) − (‖C_AD‖ + 2‖p_A‖‖p_D‖).
λ_min(A): verified Cholesky residual  A − μI = R Rᵀ + E (E in ball arithmetic)  ⇒  λ_min(A) ≥ μ − ‖E‖_F.
CONCLUSION: Q(f) ≥ λ_cert‖f‖² for all real f of that parity with supp f ⊆ [−L, L]."""
import sys, os, time, json, math
import mpmath as mp
from multiprocessing import Pool
from fractions import Fraction
from flint import arb, acb, arb_mat, acb_mat, ctx
here = os.path.dirname(os.path.abspath(__file__))
PREC, BPREC = 256, 1100

def Q(fr): return arb(fr[0])/fr[1]                  # exact rational (num, den) -> enclosing ball
def ser(x):                                          # arb -> picklable exact (mid, rad)
    m, e = x.mid().man_exp(); r, f = x.rad().man_exp()
    return (int(m), int(e), int(r), int(f))
def des(t):
    m, e, r, f = t
    return arb(arb(m)*arb(2)**e, arb(r)*arb(2)**f)

def comb_list(L2):
    """prime powers n = p^k with log n < L2 (arb), comparison in ball arithmetic: list of (n, p)"""
    out = []
    for p in range(2, int(math.exp(float(L2.mid()))) + 3):
        if all(p % q for q in range(2, int(p**0.5) + 1)):
            n = p
            while True:
                if arb(n).log() < L2: out.append((n, p)); n *= p
                else:
                    if not (arb(n).log() >= L2): raise ValueError("window edge too close to log n")
                    break
    return out
def comb_arb(cl): return [(arb(n).log(), 2*arb(p).log()/arb(n).sqrt()) for n, p in cl]
def log_dfact(n): return arb(2*n + 2).lgamma() - n*arb(2).log() - arb(n + 1).lgamma()   # log (2n+1)!!

def worker(args):
    L, h, nq, beta_s, parity, N, panels, cl = args
    ctx.prec = BPREC
    Lb, hb, beta = Q(L), Q(h), des(beta_s); logpi = arb.pi().log(); pi = arb.pi()
    comb = comb_arb(cl)
    log2m1 = [arb(2*m + 1).log() for m in range(0, 2*N + 4)]
    gl = [arb.legendre_p_root(nq, k, weight=True) for k in range(nq)]
    ms = [2*i + parity for i in range(N)]; mmax = ms[-1]
    norms = [2*(arb(2*m + 1)*Lb/2).sqrt()*(1 if (m//2) % 2 == 0 else -1) for m in ms]
    C = None
    for pidx in panels:
        ctx.prec = BPREC
        c = arb(pidx)*hb + hb/2
        cols, wts = [], []
        for xk, wk in gl:
            t = c + hb/2*xk; x = Lb*t
            j = [x.sin()/x, x.sin()/x**2 - x.cos()/x]
            lx = x.log(); logb = lx - log2m1[1]; cut = None         # log bound for j_1: x/3
            for m in range(1, mmax + 1):
                logb += lx - log2m1[m + 1]                     # log(x^{m+1}/(2m+3)!!) bound for j_{m+1}
                if cut is None and logb < -220: cut = m
                if cut is None: j.append((2*m + 1)/x*j[m] - j[m - 1])
                else: j.append(arb(0, logb.exp().upper()))
            psi = acb(arb(1)/4, t/2).digamma().real - logpi - beta
            for ln, w in comb: psi -= w*(t*ln).cos()
            cols.append([norms[i]*j[m] for i, m in enumerate(ms)])
            wts.append(hb/2*wk*psi/pi)
        ctx.prec = PREC
        K = len(cols); Fm = arb_mat(N, K); Dm = arb_mat(K, N)
        for k, col in enumerate(cols):
            wk = +wts[k]
            for i in range(N):
                v = +col[i]; Fm[i, k] = v; Dm[k, i] = v*wk
        P = Fm*Dm
        C = P if C is None else C + P
    ctx.prec = PREC
    if C is None: C = arb_mat(N, N)
    return [[ser(C[i, k]) for k in range(N)] for i in range(N)]

def digamma_sup(Tf, hf, bz):
    """rigorous sup |ψ(w)| over Re w ∈ [1/4 − bz/2, 1/4 + bz/2], 0 ≤ Im w ≤ Tf/2 + hf + 1/2 (|ψ(w̄)| = |ψ(w)|)"""
    ctx.prec = 128
    def box_sup(cx, rx, cy, ry, depth=0):           # rigorous sup of |ψ| over the box, subdividing if needed
        w = acb(arb(cx, rx*arb("1.0001")), arb(cy, ry*arb("1.0001")))
        v = abs(w.digamma())
        if v.is_finite() and float(v.upper()) < 1e6: return float(v.upper())*1.000001
        if depth > 12: raise RuntimeError("digamma box not finite")
        return max(box_sup(cx + sx*rx/2, rx/2, cy + sy*ry/2, ry/2, depth + 1) for sx in (-1, 1) for sy in (-1, 1))
    sup = 0.0; k = 0; ymax = Tf/2 + hf + 0.5
    while k*0.25 <= ymax:
        sup = max(sup, box_sup(arb(1)/4, arb(bz)/2, arb(k)/4, arb(1)/8)); k += 1
    ctx.prec = PREC
    return arb(sup)

def certify(L, T, N, parity=0, h=(1, 2), bz=0.4, target_log10=-55, procs=12, dps=85):
    t0 = time.time(); ctx.prec = PREC
    Lb, Tb, hb = Q(L), Q(T), Q(h)
    if (T[0]*h[1]) % (T[1]*h[0]) != 0: raise ValueError("T/h must be an integer")
    npan = (T[0]*h[1])//(T[1]*h[0])
    cl = comb_list(2*Lb); comb = comb_arb(cl); A_L = sum((w for _, w in comb), arb(0))
    beta = (Tb/(2*arb.pi())).log() - 1/Tb - A_L
    log = []
    def P(s): print(s, flush=True); log.append(s)
    Lf = L[0]/L[1]
    P(f"=== supp f ⊆ [−{L[0]}/{L[1]}, {L[0]}/{L[1]}] (= ±{Lf:.6f}; autocorrelation support {2*Lf:.6f}), "
      f"{'EVEN' if parity == 0 else 'ODD'} sector, T = {T[0]}/{T[1]}, N = {N} modes (orders {parity}..{2*N - 2 + parity})")
    P(f"prime powers in comb: {[n for n, _ in cl]};  A_L = {A_L.str(12)};  β = {beta.str(12)};  e·L·T/2 = {math.e*Lf*T[0]/T[1]/2:.1f}")
    if not (beta > 0 and Tb > arb(3)/4): raise ValueError("need beta > 0 and T > 3/4")
    bzb = arb(bz)
    rho = (4*bzb/hb + ((4*bzb/hb)**2 + 4).sqrt())/2           # Bernstein ellipse with semi-minor axis bz
    if not (hb/2*(rho + 1/rho)/2 < hb): raise ValueError("ellipse too wide")
    MD = digamma_sup(T[0]/T[1], h[0]/h[1], bz)
    Mpsi_ell = MD + arb.pi().log() + abs(beta) + A_L*(2*Lb*bzb).cosh()
    Mf = Mpsi_ell*2*Lb*(2*Lb*bzb).exp()/arb.pi()
    def qerr(n): return npan*hb/2*arb(64)/15*Mf/((rho**2 - 1)*rho**(2*n - 2))   # Trefethen Thm 19.3: n nodes ⇒ ρ^-(2n-2)
    nq = 4
    while not (qerr(nq) < arb(10)**target_log10): nq += 1
    eq = qerr(nq).upper()
    P(f"quadrature: {npan} panels × {nq} nodes, ρ = {rho.str(6)}, sup|ψ| on strip ≤ {MD.str(6)}, per-entry error ≤ {eq.str(3)}")
    Tq = time.time()
    chunks = [list(range(r, npan, procs)) for r in range(procs)]
    with Pool(procs) as pool:
        parts = pool.map(worker, [(L, h, nq, ser(beta), parity, N, ch, cl) for ch in chunks])
    ctx.prec = PREC
    C = arb_mat(N, N)
    for part in parts:
        for i in range(N):
            for k in range(N): C[i, k] += des(part[i][k])
    P(f"quadrature done in {time.time() - Tq:.0f}s")
    s = 1 if parity == 0 else -1
    ms = [2*i + parity for i in range(N)]
    ctx.prec = BPREC
    xh = Lb/2
    pv = [2*(arb(2*m + 1)*Lb/2).sqrt()*(arb.pi()/(2*xh)).sqrt()*xh.bessel_i(arb(m) + arb(1)/2) for m in ms]
    ctx.prec = PREC
    pv = [+v for v in pv]
    A = arb_mat(N, N); er = arb(0, eq)
    for i in range(N):
        for k in range(N): A[i, k] = C[i, k] + s*2*pv[i]*pv[k] + er
        A[i, i] += beta
    # approximate λ_min and verified Cholesky residual
    Tc = time.time()
    ev = acb_mat(A.mid()).eig(algorithm="approx")
    lam_est = min(float(e.real.mid()) for e in ev)
    P(f"approximate λ_min(A) ≈ {lam_est:.4e}   (largest {max(float(e.real.mid()) for e in ev):.3f})")
    res = dict(L=list(L), T=list(T), N=N, parity=parity, lam_est=lam_est, log=log, ok=False)
    if lam_est <= 0:
        P("leading block NOT positive definite: no certificate")
    else:
        mp.mp.dps = dps
        Am = mp.matrix(N, N)
        for i in range(N):
            for k in range(N):
                m_, e_ = A[i, k].mid().man_exp(); Am[i, k] = mp.ldexp(mp.mpf(int(m_)), int(e_))
        mu = mp.mpf(lam_est)*mp.mpf("0.999")
        Rm = mp.cholesky(Am - mu*mp.eye(N))
        Ra = arb_mat(N, N)
        for i in range(N):
            for k in range(i + 1):
                v = Rm[i, k]
                if v != 0:
                    sg, man, ex, _ = mp.mpf(v)._mpf_          # exact: (−1)^sg · man · 2^ex  (man_exp drops the sign here)
                    Ra[i, k] = (-1)**sg*arb(int(man))*arb(2)**int(ex)
        sg, man, ex, _ = mu._mpf_
        Mu = (-1)**sg*arb(int(man))*arb(2)**int(ex)
        E = A - Ra*Ra.transpose()
        fro = arb(0)
        for i in range(N):
            E[i, i] -= Mu
            for k in range(N): fro += arb(abs(E[i, k]).upper())**2
        fro = arb(fro.upper()).sqrt().upper()
        lamA = Mu - fro
        P(f"verified Cholesky: μ = {float(mu):.4e}, ‖E‖_F ≤ {float(fro):.3e}  ⇒  λ_min(A) ≥ {lamA.lower().str(5)}   ({time.time()-Tc:.0f}s)")
        # tail
        def tail_sum(x, extra):
            tot = arb(0); n = 2*N + parity; last = None
            for _ in range(20000):
                term = 2*Lb*(2*n + 1)*((n*x.log() + extra - log_dfact(n)).exp())**2
                tot += term
                if last is not None and term < last/4 and term < tot*arb(10)**-30:
                    r = term/last
                    return (tot + term*r/(1 - r)).upper()
                last = term; n += 2
            raise RuntimeError("tail sum did not converge")
        SD = tail_sum(Lb*Tb, arb(0))
        psiT = acb(arb(1)/4, Tb/2).digamma().real
        M = max(abs(acb(arb(1)/4).digamma().real - arb.pi().log() - beta).upper(),
                abs(psiT - arb.pi().log() - beta).upper()) + A_L
        CAD = (M/arb.pi()*(2*Lb*Tb).sqrt()*(Tb*SD).sqrt()).upper()
        CDD = (M/arb.pi()*Tb*SD).upper()
        pD2 = tail_sum(Lb/2, Lb**2/8)
        pA = (2*Lb.sinh()).sqrt()
        coup = (CAD + 2*pA*pD2.sqrt()).upper()
        tailmin = beta - CDD - (2*pD2 if s < 0 else 0)
        lo = lamA.lower() if lamA.lower() < tailmin.lower() else tailmin.lower()   # rigorous lower end of the min
        lam = lo - coup
        P(f"tail: S_D ≤ {SD.str(3)}, ‖C_AD‖ ≤ {CAD.str(3)}, ‖C_DD‖ ≤ {CDD.str(3)}, ‖p_D‖² ≤ {pD2.str(3)}, coupling ≤ {coup.str(3)}")
        ok = bool(lam > 0)
        P(f"RESULT: Q(f) ≥ {lam.lower().str(6)} ‖f‖²  for all real {'even' if s > 0 else 'odd'} f with supp f ⊆ [−{L[0]}/{L[1]}, {L[0]}/{L[1]}]"
          f"   ->  {'CERTIFIED' if ok else 'NOT CERTIFIED'}   (total {time.time()-t0:.0f}s)")
        res.update(ok=ok, lam_lower=lam.lower().str(20), lamA=lamA.lower().str(20), eq=eq.str(5), coupling=coup.str(5), log=log)
    json.dump(res, open(os.path.join(here, "results", f"cert_L{L[0]}_{L[1]}_T{T[0]}_N{N}_p{parity}.json"), "w"), indent=1)
    return res

if __name__ == "__main__":
    Lf, Tf = Fraction(sys.argv[1]), Fraction(sys.argv[2]); N = int(sys.argv[3])
    parity = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    certify((Lf.numerator, Lf.denominator), (Tf.numerator, Tf.denominator), N, parity)
