# -*- coding: utf-8 -*-
"""
measurement_error_simulation.py
The next evidential step named in review: test interval COVERAGE and estimation
BIAS of the calibrated Rogan-Gladen correction by Monte Carlo, under the joint
observation model
    A+ ~ Binomial(m+, r)          flags among reference-positive calibration msgs
    A- ~ Binomial(m-, f)          flags among reference-negative calibration msgs
    F  ~ Binomial(n , f+(r-f)pi)  flags in the target corpus
with r,f estimated from calibration (not known). Delta-method 95% interval:
    SE = sqrt(Var(qhat)+pihat^2 Var(rhat)+(1-pihat)^2 Var(fhat)) / |rhat-fhat|
Assumes mutual independence of the three estimates and r != f.

This turns the earlier algebra into EMPIRICAL evidence about whether the stated
intervals actually cover, and whether the untruncated estimator is unbiased in
finite samples. No model calls.
"""
import numpy as np
rng = np.random.default_rng(20261106)

def simulate(pi, r, f, m_pos, m_neg, n_corpus, reps=20000):
    q = f + (r - f) * pi
    Ap = rng.binomial(m_pos, r, reps) / m_pos
    An = rng.binomial(m_neg, f, reps) / m_neg
    Fh = rng.binomial(n_corpus, q, reps) / n_corpus
    denom = Ap - An
    ok = np.abs(denom) > 1e-6
    pihat = np.where(ok, (Fh - An) / denom, np.nan)         # untruncated
    var = (Fh*(1-Fh)/n_corpus + pihat**2*Ap*(1-Ap)/m_pos
           + (1-pihat)**2*An*(1-An)/m_neg) / denom**2
    se = np.sqrt(np.clip(var, 0, None))
    lo, hi = pihat - 1.96*se, pihat + 1.96*se
    cover = (lo <= pi) & (pi <= hi)
    bounded = np.clip(pihat, 0, 1)
    oor = (pihat < 0) | (pihat > 1)
    v = ok & np.isfinite(pihat)
    return {
        "mean_untrunc": np.nanmean(pihat[v]),
        "bias_untrunc": np.nanmean(pihat[v]) - pi,
        "mean_bounded": np.nanmean(bounded[v]),
        "bias_bounded": np.nanmean(bounded[v]) - pi,
        "coverage": np.mean(cover[v]),
        "median_hw": np.nanmedian(1.96*se[v]),
        "pct_out_of_range": 100*np.mean(oor[v]),
        "pct_degenerate": 100*np.mean(~ok),
    }

if __name__ == "__main__":
    r, f = 0.90, 0.15
    print(f"Monte Carlo (20,000 reps). True pi=0.05, r={r}, f={f}, corpus n=100,000.")
    print("Nominal interval coverage is 95%.\n")
    hdr = f"{'m+ = m-':>8}{'bias(untr)':>12}{'bias(bnd)':>11}{'coverage':>10}{'med 95% HW':>13}{'% out-range':>13}"
    print(hdr); print("-"*len(hdr))
    for m in [60, 600, 6000]:
        s = simulate(0.05, r, f, m, m, 100000)
        print(f"{m:>8}{s['bias_untrunc']*100:>+11.2f}{s['bias_bounded']*100:>+11.2f}"
              f"{s['coverage']*100:>9.1f}%{s['median_hw']*100:>11.2f}pp{s['pct_out_of_range']:>11.1f}%")
    print("\nSame, but true pi=0.20 (farther from the boundary):")
    print(hdr); print("-"*len(hdr))
    for m in [60, 600, 6000]:
        s = simulate(0.20, r, f, m, m, 100000)
        print(f"{m:>8}{s['bias_untrunc']*100:>+11.2f}{s['bias_bounded']*100:>+11.2f}"
              f"{s['coverage']*100:>9.1f}%{s['median_hw']*100:>11.2f}pp{s['pct_out_of_range']:>11.1f}%")
    print("\nReading: the untruncated estimator is ~unbiased; boundary truncation")
    print("biases upward most at tiny m and low pi; delta-method coverage is poor")
    print("at m=60 near the boundary and approaches nominal only as calibration grows.")
