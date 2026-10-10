# Independent audit of the semilocal Weil-positivity brackets

> This is the audit as originally written. File paths refer to the original working folder. The quadrature off-by-one it reports (§1.5) has since been fixed in both scripts, and all certificates in `results/` were regenerated with the fixed code. `upper.py`, `summary.txt` and the auditor's scratch scripts are not part of this repository.

Date: 2026-10-10. Auditor: Claude (independent agent, skeptical review).
Scope: `summary.txt`, `bracket_P{2,3,5,7}.json`, `cert_semilocal.py` (and the parent `F:\riemann\cert_weil.py`), `poly_upper.py`, `upper.py`, `collate.py`.
All reruns were done on byte-identical copies of the scripts in a scratch folder. SHA-256 of `cert_semilocal.py` is CBF72E07..., and of `poly_upper.py` B22227FB.... **No project file was modified except this one.**

Notation: f is real and supp f is in [-a, a]. The window is L = 2a, the support of g = f*f~. F(z) = int f(x) e^{izx} dx. Q_P is the Weil form with the comb restricted to n = p^k, p <= P.

---

## 0. Verdicts

| Claim | Verdict |
|---|---|
| 1. Lower bounds (P=2 odd 1.1140 / even 1.1180; P=3 odd 1.6138 / even 1.6150; P=5 both 1.9460) | **Confirmed with a caveat.** There is a real bug in the quadrature error bound (section 1.5). It is inherited from `cert_weil.py`. Its effect is about 1e-52 against margins of at least 5e-28, so every certificate survives the fix. Corrected reruns of P=5 are in section 3.4. |
| 2. Upper bounds (P=2 1.1147, P=3 1.6146, P=5 1.9490, P=7 2.3998) | **Confirmed.** All four reproduce bit-for-bit with the current code. My own implementation reproduces all four values to 22-25 significant digits. |
| 3a. Slack brackets | **Confirmed**, with a presentation caveat: the lower ends are rounded *up* (section 4.4). |
| 3b. "Odd fails first" for P=2 and P=3 | **Confirmed.** |
| 3c. Full Weil form Q >= 0 on every window <= ln 7 | **Confirmed** (it rests on the two P=5 certificates, which I reran with the corrected bound). |

---

## 1. (a) `cert_semilocal.py` versus `cert_weil.py`

I diffed the two files with Python `difflib`. The complete list of changes is below.

| # | Change | Rigour |
|---|---|---|
| 1 | `comb_list(L2, Pmax)`: the prime loop is capped at `Pmax`. Prime powers p^k with p <= P and log p^k < 2a are kept. The edge test `log n < 2a` is still done in ball arithmetic, and the code raises an error if it is undecidable. | OK |
| 2 | `A_L`, `beta` and the comb inside Psi (in the worker) are all built from the same truncated list `cl`. | OK, see 1.1 |
| 3 | The worker takes its working precision `BPREC` as an argument, and `calibrate_bprec` picks it. | OK, see 1.3 |
| 4 | The eigenvalue estimate is done at 768 bits instead of 256. | OK. It only chooses the shift mu. |
| 5 | Cholesky retry: if `mp.cholesky(A - mu I)` fails, mu is divided by 8. The code gives up when mu < 1e-6 * estimate. | OK, see 1.2 |
| 6 | Diagnostic prints and JSON file name. The command line takes P and the number of processes. | No effect |

Nothing else changed. In particular the envelope, the reduction, the tail bounds, the digamma box bound, the residual check and the final inequality are untouched.

### 1.1 Truncating the comb in Zhu's beta / A_L reduction
This is legitimate. The certified object is Q_P itself, with Psi_P(t) = Re psi(1/4 + it/2) - log pi - sum_{n in comb_P} (2 Lambda(n)/sqrt n) cos(t log n). Every place the comb enters uses only facts that hold for any finite comb with positive weights:
- comb_P(t) >= -A_L^P, which gives Psi_P >= log(t/2pi) - 1/t - A_L^P >= beta on [T, inf);
- |comb_P(z)| <= A_L^P cosh(2a*bz) on the Bernstein ellipse;
- M = sup_[0,T] |Psi_P - beta| <= max(endpoint values) + A_L^P.

A_L is computed from the truncated list. That is the correct constant for Q_P; the full-comb A_L would be wrong for Q_P, and needlessly large. The logs show the expected combs: P=2 gives [2]; P=3 gives [2, 4, 3]; P=5 gives [2, 4, 3, 5].

### 1.2 The "retry if Cholesky fails" logic
This is safe. mu only sets the shift. The certified bound is always lamA = mu - ||E||_F. Here E = A - R R^T - mu I is evaluated in ball arithmetic, with R and mu converted exactly from mpmath to dyadic balls, so the bound is valid for any R and any mu. A failed factorization raises `ValueError`; I checked the mpmath 1.3.0 source, which tests `s < eps`. A smaller mu gives a weaker but still verified bound. Nothing unverified is ever accepted. None of the claimed certificates actually used the retry: `grep "Cholesky failed"` finds no hits, and every log shows mu = 0.999 * estimate.

### 1.3 Per-run precision changes
These are harmless. `calibrate_bprec` only increases the working precision of the j_m recurrence, and all radii are propagated anyway. The pole vector is still computed at the module constant `BPREC = 1100`, which is fine.

### 1.4 Floats or mpmath values entering an inequality
- `lam_est` and mu (mpmath) only choose the shift. They are converted exactly.
- `digamma_sup` returns `float(upper) * 1.000001`. This is a valid upper bound, because the 1e-6 margin is far larger than the rounding of a double.
- `ok = bool(lam > 0)` is an arb comparison.

Nothing else is trusted without a ball.

### 1.5 BUG (inherited from `cert_weil.py`): off-by-one in the Gauss-Legendre error bound
The code uses, per panel, `qerr(n) = npan*(h/2)*(64/15)*Mf / ((rho^2 - 1) * rho^(2n))` with **n = nq = the number of Gauss nodes**.

Trefethen (ATAP Thm 19.3) states the Gauss bound for the **(n+1)-point** rule: |I - I_n| <= (64/15) M rho^(-2n)/(rho^2 - 1). The re-derivation is short. An m-point rule is exact up to degree 2m-1, so the first error term is T_{2m}. Then |a_k| <= 2M rho^(-k) and |I(T_k) - G_m(T_k)| <= 2 + 2/(k^2 - 1). Together these give
|I - G_m| <= (64/15) M rho^(-2(m-1))/(rho^2 - 1).
So the code's bound is too small by a factor rho^2 = 12.158, with rho = 3.4868.

This is not a theoretical quibble. For f = T_{2m} we have M = (rho^{2m} + rho^{-2m})/2 exactly. The true m-point error then exceeds the code's bound by a factor of 8.18 for m = 58 and 59, the node counts used (script `gauss_bound.py`):

    m=59: actual error 1.56421   code bound 0.191197   ratio 8.181   corrected bound 2.32453

**Effect on the numbers.** Each entry of A can be off by up to (rho^2 - 1) * eq in addition to its ball. So lambda_min drops by at most N*(rho^2 - 1)*eq:

| certificate | N | eq | extra error <= | certified lambda | still positive? |
|---|---|---|---|---|---|
| P=2 odd, L=1.1140 | 520 | 1.02e-56 | 5.9e-53 | 6.44e-7 | yes |
| P=2 even, L=1.1180 | 270 | 6.22e-56 | 1.9e-52 | 4.77e-8 | yes |
| P=3 odd, L=1.6138 | 725 | 1.80e-56 | 1.5e-52 | 5.34e-16 | yes |
| P=3 even, L=1.6150 | 240 | 5.48e-56 | 1.5e-52 | 7.26e-18 | yes |
| P=5 odd, L=1.9460 | 860 | 2.48e-56 | 2.4e-52 | 1.31e-24 | yes |
| P=5 even, L=1.9460 | 460 | 1.24e-56 | 6.4e-53 | 4.97e-28 | yes |

The same bug sits in the older `F:\riemann\cert_L31_32_*` (1.9375) certificates. Their margins are 1.0e-27 and 2.1e-24, against an extra error of about 6e-53, so they also survive.

**Fix.** Change `rho**(2*n)` to `rho**(2*n - 2)` in `qerr`; the node count then rises by one. I made this one-line patch in a scratch copy and reran the P=5 certificates (section 3.4).

### 1.6 Re-checked core steps (not changed by the copy)
I re-derived the following and found them correct:
- the Binet envelope Re psi(1/4 + it/2) >= log(t/2) - 1/t for t >= 3/4;
- the reduction Q >= s*2F(i/2)^2 + (1/pi) int_0^T (Psi - beta)|F|^2 + beta*||f||^2;
- the Legendre and Bessel transforms and pole signs;
- the analytic continuation of Re psi on the ellipse, and the coverage of the digamma boxes (including the Im w < 0 strip, handled by conjugation);
- the tail bounds (|j_n(x)| <= x^n/(2n+1)!!, Bessel's inequality, the geometric tail sum with decreasing ratio);
- the block lemma;
- the residual argument.

---

## 2. (b) `poly_upper.py` and `upper.py`

### 2.1 Every term of Q_P(f) is enclosed (poly_upper)
- **Autocorrelation.** g_mn(u) = s_mn h_mn(u/a), where h_mn is an exact rational polynomial (python-flint `fmpq_poly`, Taylor expansion in w). It is valid on 0 <= w <= 2, which is exactly where it is used. It is symmetrised as (h_mn + h_nm)/2. The cached `hpoly_D*.pkl` files are validated indirectly by the agreement in 3.2.
- **Pole.** 2(-1)^m F_m(-i/2) F_n(-i/2), with F_m(-i/2) = 2a sqrt((2m+1)/(2a)) i_m(a/2) in arb.
- **Archimedean term.** The real-space form is used, with no truncation. ∫_0^inf is split exactly: [0, 2L] is done rigorously with `acb.integral`, and [2L, inf) is done in closed form as -g(0) log(1 - e^{-2L}), because g(t/2) = 0 there. The integrands are exp, sinc(i x) and t^(k-1) with integer exponent, so they are entire, and division by sinh(t/2)/(t/2) gives non-finite balls near the poles 2 pi i k. That makes ignoring the `analytic` flag safe. I checked that integer powers use binary exponentiation (no log branch cut), that the rational endpoint 2L (an inexact ball) is propagated (`[1 +/- 1e-3]` gives a result radius of 1e-3), and that the imaginary part is discarded only because the true value is real. Tolerance is 2^-1520; the error is in the ball.
- **Prime sum.** It is finite, and the k log p < L edge test is done in balls.
- **Norm and quadratic form.** v is rounded to an exact dyadic vector and v^T M v is evaluated in arb at 1600 bits. Only v itself comes from floating point, which is legitimate.

`upper.py` uses the sine basis. It uses [eps, 2L] by `acb.integral`, a bounded remainder on [0, eps] (via |g'| <= ||f|| ||f'||), and the closed-form tail. It is sound but superseded: no claimed bound uses it.

### 2.2 Same normalisation as the lower-bound code: checked concretely
`norm_check.py` evaluates explicit test functions f = y^e (1-y^2)^K, with y = x/a and degree <= 21, so they lie exactly in the basis. Each is evaluated with both project code paths:
- (i) the `cert_semilocal` matrix A = C(T) + beta I + s 2 p p^T, built with the project's own `worker` and pole vector, which gives R_T(f) = Q(f) - (1/pi) ∫_T^inf (Psi - beta)|F|^2 with T = 400;
- (ii) `poly_upper.build_M`.

| window / comb | f | cert path R_T(f) | poly path Q(f) | difference |
|---|---|---|---|---|
| 1.1147 / {2} | y(1-y^2)^10 | 0.00165566214581872492696469204077020 | 0.00165566214581872492696469204078211 | -1.2e-32 |
| 1.1147 / {2} | y^3(1-y^2)^9 | 5.69819890580037480977930222774e-6 | 5.69819890580037480977930396108e-6 | -1.7e-30 |
| 1.1147 / {2} | (1-y^2)^10 (even) | 0.010474354861175136157317484398461 | 0.010474354861175136157317484398473 | -1.3e-32 |
| 1.949 / {2,4,3,5} | y(1-y^2)^10 | 1.58375995411583640791900268519236e-5 | 1.58375995411583640791900268519243e-5 | -6.9e-37 |
| 1.949 / {2,4,3,5} | (1-y^2)^10 | 8.13036641769902495190508253135035e-6 | 8.13036641769902495190508253135109e-6 | -7.4e-37 |
| 2.3998 / {2,4,8,3,9,5,7} | y(1-y^2)^10 | 5.98686035555526540499954096260409e-8 | 5.98686035555526540499954096260589e-8 | -1.8e-38 |

All the rows, plus the bilinear cross terms B(f1, f3) and the y^2, y^5 functions, agree to 28-37 digits. The differences have the sign and size of the omitted tail beyond T. **There is no constant offset and no factor 2: both codes compute the same Q_P.**

---

## 3. (c) Independent spot checks

### 3.1 My own implementation of the Weil form
I wrote `indep_weil.py` in plain mpmath; it uses no project code. The explicit formula, in this normalisation, is

    sum_rho h(gamma) = h(i/2) + h(-i/2) - g(0) log pi + (1/2pi) ∫_R h(t) Re psi(1/4 + it/2) dt - 2 sum_n Lambda(n) n^{-1/2} g(log n)

with h = |F|^2 and g = f*f~, so that g(0) = ||f||^2. Q_P is the right-hand side with n restricted to powers of p <= P. I implemented it in three forms:
- **Fourier form:** Q_P = 2F(i/2)F(-i/2) + (1/pi) ∫_0^inf |F(t)|^2 Psi_P(t) dt, using mpmath `digamma`.
- **Real-space form** (from psi(s) = -gamma + ∫_0^inf (e^{-x} - e^{-sx})/(1 - e^{-x}) dx): Q_P = 2F(i/2)F(-i/2) - (gamma + log pi) g(0) + ∫_0^inf [g(0)e^{-t} - e^{-t/4} g(t/2)]/(1 - e^{-t}) dt - 2 sum (log p) p^{-k/2} g(k log p). Here g is computed by Gauss-Legendre in x from f evaluated pointwise, which is exact for polynomials.
- **Zero sum** (full Q only): 2 sum_{gamma>0} |F(gamma)|^2, using the first 160 zeros from `mpmath.zetazero`.

Validation on smooth functions at window 1.1147:
- the Fourier and real-space forms agree to 1e-27..1e-31 (the difference is the Fourier tail beyond T = 400);
- the real-space form agrees with the **zeta-zero sum** to 1.7e-30, 2.0e-30 and 1.9e-28 for three functions. This pins the normalisation to the actual zeros.
- My real-space value equals the poly_upper value to all 35 printed digits (0.0016556621458187249269646920407821068).

### 3.2 The explicit upper-bound vectors, recomputed independently
`run_vector.py` (real-space form, dps 50-80, two quadrature resolutions for P=2) gives:

| P, window | my Q_P(f) (mpmath) | project ball |
|---|---|---|
| 2, 1.1147 | -3.60762498685477970398068e-8 | [-3.60762498685477970398068025664e-8 +/- 3.61e-39] |
| 3, 1.6146 | -4.850082265395684900170077e-16 | [-4.85008226539568490017007685594e-16 +/- 4.29e-46] |
| 5, 1.9490 | -1.121309898744883596869569e-23 | [-1.12130989874488359686956855358e-23 +/- 4.73e-53] |
| 7, 2.3998 | -2.646472781729821399480846e-32 | [-2.64647278172982139948084623697e-32 +/- 3.79e-62] |

In each case ||f||^2 = 1. The sign and the value are confirmed to 22-25 digits. For P=2 the pieces are: pole -0.023144550328703795853, arch -0.174619331486717351952, prime +0.197763845739171279257.

**A fully Fourier-side check of the P=2 vector.** The real-space and Fourier forms were compared above only on smooth functions, but the test vectors are polynomials with jumps at +-a, where |F|^2 decays only like 1/t^2. So I also evaluated Q_2(v) directly from the digamma form, in `run_vector_fourier3.py` (dps 50):
- (1/pi) ∫_0^T |F|^2 Psi_2 is done by composite Gauss-Legendre in t, with F(t) computed by quadrature in x;
- for t >= T, F(t) = e^{ita} alpha(t) - e^{-ita} beta(t) holds exactly (repeated integration by parts). The endpoint derivatives come from the closed form P_m^(k)(1) = (m+k)!/(2^k k! (m-k)!). Each frequency component (0, 2a, log 2, 2a +- log 2) is integrated separately with `mp.quad` / `mp.quadosc`;
- the two ways of computing F agree to 3e-50 at t = 200 and 4e-44 at t = 100.

Results:

    T = 200:  pole -0.0231445503287037958534,  (1/pi)int_0^T +0.0231392145157307771053,  (1/pi)tail +5.2997367e-6
              Q_2(f) = -3.607624986854779704e-8
    T = 120:  (1/pi)int_0^T +0.0231359678506447122502,  (1/pi)tail +8.5464018e-6
              Q_2(f) = -3.607624986854779704e-8

Both split points give the same value, which agrees with the project ball to 19 digits. So the sign and value of the P=2 upper bound are confirmed by the Fourier/digamma form directly, not only through the real-space rewriting. This also shows that the two forms agree on functions with jumps, which is what makes the lower and upper bounds refer to the same Q_P.

### 3.3 Reruns of project code (current versions)
- **`cert_semilocal.py 2 557/1000 1200 520 1 12`** (P=2 odd, L=1.1140): CERTIFIED, lambda >= 6.4437703333178389111e-7. This is identical to `cert_P2_L557_1000_T1200_N520_p1.json` in every field (413 s).
- **P=2 even (L=1.1180) and P=3 even (L=1.6150).** The logs of these two certificates lack the "BPREC" line, so they were produced by an **earlier version** of `cert_semilocal.py`. I reran both with the current file. The lambdas are identical: 4.7656116681983296742e-8 and 7.2578998770031061527e-18.
- **`poly_upper.py` for P = 2, 3, 5, 7** (current version). All four are CERTIFIED NEGATIVE with balls identical to the stored ones. The stored P=5 result predates the last edit of `poly_upper.py` (07:36 versus 07:37); the rerun gives the same ball and the same coefficients.

### 3.4 The P=5 certificates rerun with the corrected quadrature bound
I applied the one-line fix from section 1.5, `rho**(2*n - 2)` in `qerr`, in the scratch copy `cert_semilocal_fixed.py`. The node count then rises from 59 to 60, which restores a per-entry error bound of about 1e-56, now valid.

| run | result | lambda (corrected code) | lambda (stored certificate) |
|---|---|---|---|
| `5 973/1000 600 460 0` (even, window 1.946) | CERTIFIED (248 s) | 4.9680825054986708096e-28 | 4.9680825054986708096e-28 |
| `5 973/1000 1200 860 1` (odd, window 1.946) | CERTIFIED (1212 s) | 1.3057083279252511545e-24 | 1.3057083279252511545e-24 |

So the two certificates behind "full Q >= 0 on every window <= ln 7" stand with a correct error bound. The lambdas are identical to all 20 digits; the residual norm ||E||_F (about 1e-53) is the only thing the bound affects. One small oddity: the approximate *largest* eigenvalue printed for the odd run differs between the two runs (4.694 versus 4.762). It comes from the unverified `eig(algorithm="approx")`, which only chooses mu, so it has no effect on rigour.

---

## 4. (d) Logic of the consequences

### 4.1 Q = Q_5 on windows <= ln 7
The prime powers n with log n < 1.945910 (n < 7) are 2, 3, 4 and 5; 6 is not a prime power. n = 7 sits exactly at the boundary. For a window L <= ln 7, g(log 7) = 0, because g is continuous and vanishes at its support edge, and every n >= 7 drops out. So the full Q equals Q_5 there.

The P=5 certificates are for half-width 0.973, i.e. window 1.946 > ln 7, in both parities. A function supported in [-L/2, L/2] with L <= ln 7 is also supported in [-0.973, 0.973]. Hence Q(f) = Q_5(f) >= 4.97e-28 * ||f||^2. **Confirmed.** Two remarks:
- on (ln 7, 1.946] the certificate says nothing about Q, because 7 enters the comb there;
- the P=7 row of the bracket table correctly uses ln 7, not 1.946, as its lower end.

### 4.2 Parity decoupling
For real f = f_e + f_o, the cross terms of g = f*f~ cancel: f_e*f_o~ + f_o*f_e~ = -f_e*f_o + f_o*f_e = 0. In the pole term, F(i/2)F(-i/2) = (E + O)(E - O) = E^2 - O^2. Hence Q(f) = Q(f_e) + Q(f_o) and ||f||^2 = ||f_e||^2 + ||f_o||^2. So min(lambda_e, lambda_o) bounds all real f. For complex f, Q(u + iv) = Q(u) + Q(v) because the kernel is real and symmetric. **Confirmed.** This is also visible numerically: the cert path builds one parity at a time.

### 4.3 "Odd fails first"
- P=2: even is certified positive at window 1.1180, hence on every window <= 1.1180, including 1.1147. At 1.1147 an explicit odd f has Q_2 < 0.
- P=3: even is certified positive at 1.6150 >= 1.6146, where odd is certified negative.

So L*_odd <= L_up < L_even-certified <= L*_even. **Confirmed.** No such claim is made for P=5, and none could be made from these data.

### 4.4 Slack brackets
These are the exact values from the certified windows:
- P=2: [1.1140 - ln 3, 1.1147 - ln 3] = [0.015388, 0.016088]
- P=3: [1.6138 - ln 5, 1.6146 - ln 5] = [0.004362, 0.005162]
- P=5: [1.9460 - ln 7, 1.9490 - ln 7] = [0.0000899, 0.0030899]

The quoted lower ends 0.0154, 0.0044 and 0.00009 (also "+0.00009" in `summary.txt`) are **rounded up**, so they slightly overstate the certified lower bounds. Use 0.0153, 0.0043 and 0.00008 (or more digits).

### 4.5 Cosmetic issues
- `summary.txt` and `summary_table.txt` print the upper-bound radii as "+/- 4e-25" and "rad 3.7e-25". That is an artefact of `collate.py` re-parsing the 30-digit ball string at 53-bit precision. The true radii are 3.6e-39 (P=2), 4.3e-46 (P=3), 4.7e-53 (P=5) and 3.8e-62 (P=7).
- The JSON `coeffs` are 60-digit decimal strings. The certified vector is their rounding to about 336 bits. The difference changes Q by less than 1e-90, which is irrelevant.

---

## 5. What was not done
- I did not rerun the P=3 odd certificate (67 min, over the time budget). Its reasoning is identical to the reruns that were done, and its margin (5.3e-16) is enormous compared with the quadrature bug.
- The earlier audit of `cert_weil.py` missed the off-by-one in 1.5. Any other certificate produced with `cert_weil.py` should get the same one-line fix; the margins I saw make it harmless in practice.
- Python-flint (Arb 0.9.0) and mpmath 1.3.0 are trusted as correct implementations.

## 6. Files (auditor scratch, not in the project)
`indep_weil.py`, `run_smooth.py`, `run_zeros.py`, `run_vector.py`, `run_vector_fourier3.py`, `gauss_bound.py`, `norm_check.py`, and `cert_semilocal_fixed.py` (the one-line `qerr` fix), plus their logs. They are in the session scratchpad (`...\scratchpad\mine` and `...\scratchpad\work`).
