"""Collect the certificates in this folder into bracket_P{P}.json and summary.txt.
L_low: largest window L (= autocorrelation support = 2 * half-width) at which BOTH parities are certified positive
       (odd at L, even at some L' >= L; positivity on a window implies it on every smaller window).
L_up : smallest window with a certified-negative explicit test function (poly_upper.py / upper.py)."""
import json, glob, os, math
from fractions import Fraction
from flint import arb
here = os.path.dirname(os.path.abspath(__file__))
nextp = {2: 3, 3: 5, 5: 7, 7: 11}

def certs(P):
    out = []
    for fn in glob.glob(os.path.join(here, "results", f"cert_P{P}_*.json")):
        d = json.load(open(fn))
        if d.get("ok"):
            Lw = 2*Fraction(d["L"][0], d["L"][1])
            out.append(dict(file=os.path.basename(fn), L=Lw, parity=d["parity"], lam=d["lam_lower"], T=d["T"][0], N=d["N"]))
    return out

def uppers(P):
    out = []
    for fn in glob.glob(os.path.join(here, "results", f"polyupper_P{P}_*.json")) + glob.glob(os.path.join(here, "results", f"upper_P{P}_*.json")):
        d = json.load(open(fn))
        if d.get("certified_negative"):
            out.append(dict(file=os.path.basename(fn), L=Fraction(d["L_up"]), Q_ball=d["Q_ball"],
                            basis=(f"Legendre odd, deg<= {d['D']}" if "D" in d else f"sine J={d['J']}")))
    return out

rows = []
for P in (2, 3, 5, 7):
    lnp = math.log(nextp[P])
    C = certs(P); U = uppers(P)
    note = ""
    if P == 7:      # below ln 7 the P=7 form equals the P=5 form (same comb {2,3,4,5}); use the P=5 certificates there
        C5 = [c for c in certs(5) if c["L"] <= Fraction(1945910, 10**6)]
        C5 += [dict(c, note="P=5 cert at L>ln7; Q_7 = Q_5 on every window <= ln 7") for c in certs(5) if c["L"] > Fraction(1945910, 10**6)]
        C = C + C5
        note = "Q_7 = Q_5 on windows <= ln 7, so the P=5 certificate at 1.946 gives positivity of Q_7 on every window <= ln 7 = 1.945910"
    odd = [c for c in C if c["parity"] == 1]; even = [c for c in C if c["parity"] == 0]
    best = None
    for o in sorted(odd, key=lambda c: c["L"], reverse=True):
        e = [c for c in even if c["L"] >= o["L"]]
        if e:
            best = (o, max(e, key=lambda c: c["L"])); break
    up = min(U, key=lambda u: u["L"]) if U else None
    Llow = best[0]["L"] if best else None
    if P == 7 and best and best[0]["L"] > math.log(7): Llow = Fraction(math.log(7))   # sup statement: positive on every window <= ln 7
    r = dict(P=P, ln_next_prime=lnp, L_low=(float(Llow) if Llow else None),
             L_low_odd_cert=(best[0] if best else None), L_low_even_cert=(best[1] if best else None),
             L_up=(float(up["L"]) if up else None), L_up_cert=up,
             slack_bracket=[(float(Llow) - lnp) if Llow else None, (float(up["L"]) - lnp) if up else None], note=note)
    json.dump(r, open(os.path.join(here, "results", f"bracket_P{P}.json"), "w"), indent=1, default=str)
    rows.append(r)
lines = ["Semilocal Weil form Q_P (primes p <= P, all powers), window L = support of f*f~ = 2a.",
         "L_low: Q_P certified positive (both parities) on supp f in [-L/2, L/2]   (cert_semilocal.py, ball arithmetic)",
         "L_up : explicit odd polynomial test function with Q_P < 0 certified in ball arithmetic   (poly_upper.py)", "",
         f"{'P':>2} | {'ln p_next':>9} | {'L_low':>9} | lambda_odd(L_low) / lambda_even | {'L_up':>9} | Q_P(v) ball at L_up | slack bracket"]
for r in rows:
    o, e, u = r["L_low_odd_cert"], r["L_low_even_cert"], r["L_up_cert"]
    fl = lambda s: float(arb(s).mid())
    lam = f"odd {fl(o['lam']):.4e} (T={o['T']},N={o['N']}) / even {fl(e['lam']):.4e} at L={float(e['L']):.4f}" if o else "-"
    if u is None:
        lines.append(f"{r['P']:>2} | {r['ln_next_prime']:.6f} | {r['L_low']:.6f} | {lam} | (pending) |"); continue
    lines.append(f"{r['P']:>2} | {r['ln_next_prime']:.6f} | {r['L_low']:.6f} | {lam} | {r['L_up']:.6f} | {fl(u['Q_ball']):.6e} (rad {float(arb(u['Q_ball']).rad()):.1e}; {u['basis']}) | "
                 f"[{r['slack_bracket'][0]:+.5f}, {r['slack_bracket'][1]:+.5f}]")
    if r["note"]: lines.append("     note: " + r["note"])
open(os.path.join(here, "results", "summary_table.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
