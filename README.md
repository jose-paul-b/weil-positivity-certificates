# Certified Weil positivity on small windows, and the semilocal slack

These are computer-assisted, interval-arithmetic certificates (python-flint / Arb) for Weil's quadratic form Q on test functions with small support. The repository also brackets where the *truncated* (semilocal) forms Q_P lose positivity.

**Status:** these results have been checked by an independent automated audit (see `docs/AUDIT.md`) and by no human expert. They are **not** a record: Liu, "Certified Weil Positivity Beyond the Unit Window" (2026, unrefereed), certifies support 2.125. This repository is an independent check by a separate implementation, with a much larger certified constant, plus the semilocal brackets.

## The form

For real f supported in [−a, a], let F be its Fourier transform. In Zhu's normalisation (arXiv 2608.24827):

    Q(f) = 2·F(i/2)·F(−i/2) + (1/2π) ∫ |F(t)|² Ψ(t) dt,
    Ψ(t) = Re ψ(1/4 + it/2) − log π − Σ_{log n < 2a} 2Λ(n) n^(−1/2) cos(t log n).

RH is equivalent to Q(f) ≥ 0 for all such f and every a (Weil's criterion).

**Semilocal form Q_P:** the same, with the sum restricted to n = p^k, p ≤ P.

**Window:** "window L" below means the support of the autocorrelation f * f~, i.e. L = 2a.

## Results

### 1. Full Weil form (`weil_certificate/`)

| supp f | parity | certified bound Q(f) ≥ λ‖f‖² | settings |
|---|---|---|---|
| [−4/5, 4/5] | even | λ = 1.02666e-17 | T=200, N=200 |
| [−4/5, 4/5] | odd | λ = 1.13991e-14 | T=200, N=200 |
| [−31/32, 31/32] | even | λ = 1.00736e-27 | T=600, N=460 |
| [−31/32, 31/32] | odd | λ = 2.11938e-24 | T=600, N=460 |

- **No cross terms:** even and odd functions do not interact in Q, so the two parities together cover all real f.
- **Agreement with Zhu:** the 4/5 case reproduces his Theorem 1.1 (he gets λ ≥ 8.9e-18).
- **Corollary from §2:** the P = 5 certificate at a = 0.973 also gives Q ≥ 0 for every f with window L ≤ log 7 = 1.945910. On those windows the only prime powers are 2, 3, 4 and 5, so Q = Q_5.

### 2. Semilocal brackets (`semilocal/`)

L*_P = sup{L : Q_P ≥ 0 on windows of length L}.

For L < log(next prime after P), Q_P = Q. The slack is L*_P − log(next prime).

| P | lower bound: Q_P ≥ 0 (both parities) | upper bound: explicit odd f with Q_P(f) < 0 | slack bracket |
|---|---|---|---|
| 2 | L = 1.1140 (odd λ ≥ 6.44e-7; even λ ≥ 4.77e-8 at L = 1.1180) | L = 1.1147, Q = −3.6076e-8 | [0.015387, 0.016088] |
| 3 | L = 1.6138 (odd λ ≥ 5.34e-16; even λ ≥ 7.26e-18 at L = 1.6150) | L = 1.6146, Q = −4.8501e-16 | [0.004362, 0.005163] |
| 5 | L = 1.9460 (odd λ ≥ 1.31e-24; even λ ≥ 4.97e-28) | L = 1.9490, Q = −1.1213e-23 | [0.0000898, 0.003090] |
| 7 | (out of reach for this method) | L = 2.3998, Q = −2.6465e-32 | ≤ 0.001905 |

**The odd sector fails first (P = 2, 3).** The even block is certified positive on a window where an explicit odd f is certified negative: L = 1.1147 for P = 2, and L = 1.6146 for P = 3.

**Heuristic mechanism (not proved here).** On log p < L < log(next prime),

    Q_P(f) − Q(f) = 2 (log p / √p) · g(log p),   with g = f * f~.

For odd f, g(log p) is typically negative.

**Prior work.** Connes–Consani (arXiv 2106.01715, §2.2–2.4) already observed that truncated forms lose positivity shortly after each new prime. This repository only adds rigorous brackets and the odd-first ordering.

## Method

**Lower bounds** (`cert_weil.py`, and `cert_semilocal.py`, which is the same code with an extra prime cutoff P):
1. Zhu's one-stroke reduction Q ≥ R, with β = log(T/2π) − 1/T − A_L. The constants are re-derived in the file header, together with an envelope bound for Re ψ.
2. Basis F_m(t) = 2 i^m √(a(2m+1)/2) · j_m(a t) (Legendre polynomials on [−a, a]), with N modes of one parity.
3. Matrix entries on [0, T] by panel Gauss–Legendre quadrature. The error is bounded through Bernstein ellipses (Trefethen, ATAP Thm 19.3) and an enclosure of ψ on the ellipse.
4. The tail beyond N modes is controlled by |j_n(x)| ≤ x^n / (2n+1)!! and a block lemma.
5. Smallest eigenvalue: an approximate shift μ is chosen, and a Cholesky factor R of A − μI is computed and converted exactly to rationals. Then λ_min(A) ≥ μ − ‖A − μI − RRᵀ‖_F, with the residual computed in ball arithmetic.

**Upper bounds** (`poly_upper.py`):
- An explicit odd polynomial f, a combination of Legendre polynomials of degree ≤ 61–81.
- Q_P(f) is evaluated entirely in ball arithmetic. The autocorrelations are exact rational polynomials, the pole term is in closed form, and the archimedean integral is done by `acb.integral`. The prime sum is finite.

**What must be trusted:**
- python-flint 0.9.0 (Arb). mpmath only supplies approximate eigenvalues, the Cholesky factor and test vectors; every quantity used in an inequality is a ball.

## Running

```
pip install -r requirements.txt

cd weil_certificate
python cert_weil.py 4/5 200 200 0        # args: a  T  N  parity (0 = even, 1 = odd); about 1 min with 12 processes
python cert_weil.py 31/32 600 460 1      # about 3 min

cd ../semilocal
python cert_semilocal.py 2 557/1000 1200 520 1 12   # args: P  a  T  N  parity  processes; about 6 min
python poly_upper.py 2 11147/10000 61 1             # args: P  L(window)  max degree  parity; about 2 min
python collate.py                                   # rebuilds results/bracket_P*.json and results/summary_table.txt
```

- **Output:** each certificate writes a JSON file to `results/`; the printed log ends with a `RESULT` line.
- **Note on arguments:** `cert_*.py` take the half-width a; `poly_upper.py` takes the window L = 2a.
- **Longest run:** the P = 3 odd certificate (a = 8069/10000, T = 1200, N = 725) takes about 70 min.

## Layout

```
weil_certificate/cert_weil.py      full-form certificate
weil_certificate/results/          JSON certificates and logs
semilocal/cert_semilocal.py        lower bounds for Q_P
semilocal/poly_upper.py            certified negative values of Q_P
semilocal/collate.py               assembles the brackets
semilocal/results/                 certificates, upper bounds, bracket_P*.json, summary_table.txt
docs/AUDIT.md                      independent audit (claims, reruns, independent re-implementation, bug found)
```

## Verification history

**Audit outcome.** An independent audit (`docs/AUDIT.md`) found every claim confirmed. Its checks:
- it reran certificates;
- it compared the two code paths (they agree to 28–37 digits, with no normalisation offset);
- it checked a separate mpmath implementation of the Weil form against 160 zeta zeros (agreement 1e-30), and that implementation reproduced the negative values to 19–25 digits.

**Bug found and fixed.** The audit found an off-by-one in the quadrature error bound: ρ^(−2n) should be ρ^(−(2n−2)) for n Gauss nodes. Its effect on any λ is at most about 2e-52, far below every certified margin. It is fixed in both scripts, and every certificate in `results/` was regenerated with the fixed code.

**Path changes after the audit.** The scripts here differ from the audited versions only in that fix and in writing their output to `results/`.

## Credits and disclosure

This work was carried out with substantial AI assistance (Anthropic's Claude), including code, runs and the audit. Errors are mine; please report them as issues.

References:
- X. Zhu, arXiv 2608.24827
- A. Connes, C. Consani, arXiv 2106.01715
- A. Connes, arXiv 2602.04022
- E. Bombieri, "Remarks on Weil's quadratic functional in the theory of prime numbers" (2000)

Author: Jose Paul B. License: MIT (see LICENSE).
