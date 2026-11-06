# -*- coding: utf-8 -*-
"""
measurement_error_correction.py  (v2 - corrected after review)

Connecting a fallible detector to an AGGREGATE input D_{k,t} of the forecasting
model. Rogan-Gladen inversion is an ESTABLISHED method; the contribution is its
careful integration into the measurement architecture, not a new estimator.

CORRECTIONS incorporated (do not regress these):
 1. Truncation to [0,1] gives admissible values but introduces BOUNDARY BIAS;
    only the UNTRUNCATED estimator is unbiased, and only when r,f are known and
    applicable. Estimated r,f add variance and finite-sample bias.
 2. The recovery table is ILLUSTRATIVE EXPECTED VALUES under fixed r,f, not
    empirical recovery from noisy observations.
 3. D_{k,t} needs CATEGORY-SPECIFIC calibration r_k=P(Yhat=k|Y=k),
    f_k=P(Yhat=k|Y!=k) where the negatives INCLUDE other categories, not only
    'none'. Pooled 54/60 and 9/60 do NOT establish r_k,sp_k per category.
 4. Abstention: define Z=1 iff category flagged, 0 otherwise (abstention counts
    as 0). The correction concerns P(flag); calibration must use the same Z.
 5. More corpus messages do NOT remove calibration uncertainty: Var(pi_hat) has
    terms in Var(r_hat),Var(f_hat) that an infinite corpus cannot shrink.
 6. 'Raw flag proportion is upward-biased noise' is wrong: bias q-pi =
    f(1-pi)-(1-r)pi changes sign (here at pi=60%). And 'most labels wrong' is
    wrong: at low prevalence most FLAGS may be wrong, not most labels.
 7. Strongest step: a binomial OBSERVATION MODEL with latent prevalence;
    flag counts alone cannot identify (pi, r, f) jointly - calibration is needed.
"""
import numpy as np

def q_of(r, f, pi):            # population flag probability
    return f + (r - f) * pi
def rg(qhat, r, f):            # untruncated Rogan-Gladen
    return (qhat - f) / (r - f)
def rg_bounded(qhat, r, f):
    return min(1.0, max(0.0, rg(qhat, r, f)))

if __name__ == "__main__":
    r, f = 0.90, 0.15
    print(f"Operating point: r={r}, f={f}  (q = f + (r-f)*pi = {f} + {r-f}*pi)\n")

    print("TABLE 1 - ILLUSTRATIVE EXPECTED VALUES under fixed r,f")
    print("(algebraic recovery from EXPECTED flag rates, not from noisy data)")
    print(f"{'true pi':>9}{'expected q':>13}{'naive reading':>16}{'RG (untrunc.)':>16}")
    for pi in [0.01, 0.05, 0.10, 0.20, 0.40]:
        q = q_of(r, f, pi)
        print(f"{pi*100:>7.0f}% {q*100:>11.1f}% {q*100:>14.1f}% {rg(q,r,f)*100:>14.1f}%")

    print("\nTABLE 2 - BIAS OF THE RAW FLAG PROPORTION as an estimator of pi")
    print("bias(pi) = q - pi = f(1-pi) - (1-r)pi ; sign flips, not universally up")
    pistar = f / (f + (1 - r))
    for pi in [0.05, 0.30, 0.60, 0.80]:
        b = q_of(r, f, pi) - pi
        print(f"   pi={pi*100:>4.0f}%  bias = {b*100:+5.1f} pp")
    print(f"   -> bias crosses zero at pi* = f/(f+(1-r)) = {pistar*100:.0f}%")

    print("\nTABLE 3 - CALIBRATION UNCERTAINTY DOES NOT VANISH WITH CORPUS SIZE")
    print("Var(pi_hat) ~ [Var(qhat) + pi^2 Var(r_hat) + (1-pi)^2 Var(f_hat)]/(r-f)^2")
    print("ASSUMES the three estimates are mutually independent (else add covariance")
    print("terms) and r != f (at r=f the flag rate carries no information on pi).")
    print("m = calibration messages PER CLASS: m+ reference-positives, m- reference-")
    print("negatives (so m=60 means 120 calibration messages). Corpus taken infinite.")
    print("These are illustrative delta-method half-widths, NOT guaranteed coverage.")
    pi = 0.05
    for m in [60, 600, 6000]:
        var_r = r*(1-r)/m
        var_f = f*(1-f)/m
        var_pi = (pi**2*var_r + (1-pi)**2*var_f) / (r-f)**2   # Var(qhat)=0
        hw = 1.96*np.sqrt(var_pi)
        print(f"   m+ = m- = {m:>5}: residual 95% half-width = +/- {hw*100:.2f} pp")
    print("   -> at low prevalence the false-positive-rate uncertainty dominates;")
    print("      a larger corpus cannot compensate a poorly estimated f. The m=60")
    print("      row is especially unsuitable as a symmetric bounded interval.")

    print("\nTABLE 4 - NON-IDENTIFIABILITY: flags alone cannot pin pi AND f,r")
    qobs = 0.1875  # exact expected flag rate at pi=0.05 (not the rounded 18.8%)
    print(f"   one observed flag rate q={qobs} is consistent with many (pi,f) at r={r}:")
    for ftest in [0.150, 0.100, 0.075]:
        pit = (qobs - ftest)/(r - ftest)
        print(f"      if f={ftest:.3f} -> compatible pi = {pit*100:5.2f}%")
    print("   -> Flag counts alone do not identify prevalence and both classification")
    print("      rates. This needs external calibration OR other defensible identifying")
    print("      restrictions; finite calibration constrains pi but does not determine")
    print("      it without uncertainty. The chosen strategy here is a joint model with")
    print("      a calibration likelihood:")
    print("        A+ ~ Binomial(m+, r)      A- ~ Binomial(m-, f)")
    print("        F_t ~ Binomial(n_t, f + (r-f)*pi_t),   D_k,t = pi_t (latent)")
    print("      (see measurement_error_simulation.py for coverage/bias evidence).")
