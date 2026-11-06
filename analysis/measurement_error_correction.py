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
    print("Take an INFINITE corpus (Var(qhat)->0); vary calibration set sizes.")
    pi = 0.05
    for n_cal in [60, 600, 6000]:
        var_r = r*(1-r)/n_cal
        var_f = f*(1-f)/n_cal
        var_pi = (pi**2*var_r + (1-pi)**2*var_f) / (r-f)**2   # Var(qhat)=0
        hw = 1.96*np.sqrt(var_pi)
        print(f"   calibration n={n_cal:>5} (per class): residual 95% half-width = +/- {hw*100:.2f} pp")
    print("   -> at low prevalence the false-positive-rate uncertainty dominates;")
    print("      a larger corpus cannot compensate a poorly estimated f.")

    print("\nTABLE 4 - NON-IDENTIFIABILITY: flags alone cannot pin pi AND f,r")
    qobs = q_of(r, f, 0.05)
    print(f"   one observed flag rate q={qobs*100:.1f}% is consistent with many (pi,f) at r={r}:")
    for ftest in [0.075, 0.10, 0.15, 0.1875]:
        pit = (qobs - ftest)/(r - ftest)
        print(f"      if f={ftest:.4f} -> implied pi = {max(pit,0)*100:5.1f}%")
    print("   -> only independent calibration of r,f identifies pi. Hence the")
    print("      observation model F_k,t ~ Binomial(n_t, f_k+(r_k-f_k)*pi_k,t),")
    print("      with pi_k,t the latent D_k,t and calibration data informing r_k,f_k.")
