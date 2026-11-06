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
    cov = np.mean(cover[v])
    nn = int(np.sum(v))
    return {
        "mean_untrunc": np.nanmean(pihat[v]),
        "bias_untrunc": np.nanmean(pihat[v]) - pi,
        "mean_bounded": np.nanmean(bounded[v]),
        "bias_bounded": np.nanmean(bounded[v]) - pi,
        "coverage": cov,
        "coverage_mc_se": np.sqrt(cov*(1-cov)/nn),   # Monte Carlo SE of coverage
        "median_hw": np.nanmedian(1.96*se[v]),
        "pct_out_of_range": 100*np.mean(oor[v]),
        "pct_degenerate": 100*np.mean(~ok),
    }

if __name__ == "__main__":
    r, f = 0.90, 0.15
    REPS = 20000
    print("SIMULATED SCENARIO (results are empirical coverage IN THIS SCENARIO,")
    print("not a general property of the method).")
    print(f"  generating parameters: r={r}, f={f}; corpus n=100,000; reps={REPS:,}")
    print("  estimator: untruncated Rogan-Gladen pihat=(qhat-fhat)/(rhat-fhat)")
    print("  interval : pihat +/- 1.96*SE, SE from the delta-method variance with")
    print("             estimated (rhat,fhat,qhat); coverage = fraction of intervals")
    print("             containing the true pi (interval NOT clipped to [0,1]).")
    print("  Monte Carlo SE of each coverage figure ~ sqrt(p(1-p)/reps) (reported).")
    print("  Nominal coverage = 95%.\n")
    hdr = f"{'pi':>6}{'m+=m-':>7}{'bias(untr)':>12}{'bias(bnd)':>11}{'coverage':>10}{'MC SE':>8}{'med HW':>9}{'%out[0,1]':>11}"
    print(hdr); print("-"*len(hdr))
    for pi in [0.05, 0.20]:
        for m in [60, 600, 6000]:
            s = simulate(pi, r, f, m, m, 100000, reps=REPS)
            print(f"{pi:>6.2f}{m:>7}{s['bias_untrunc']*100:>+11.2f}{s['bias_bounded']*100:>+11.2f}"
                  f"{s['coverage']*100:>9.1f}%{s['coverage_mc_se']*100:>7.2f}{s['median_hw']*100:>8.2f}pp{s['pct_out_of_range']:>10.1f}%")
    print("\nReading: the untruncated estimator is ~unbiased; boundary truncation")
    print("biases upward most at small m and low pi; delta-method coverage is")
    print("below nominal at m=60 near the boundary and approaches the nominal level")
    print("in the evaluated scenarios as calibration grows. These are scenario-")
    print("specific results, not guaranteed coverage for a deployed detector.")
