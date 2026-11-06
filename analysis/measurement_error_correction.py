# -*- coding: utf-8 -*-
"""
measurement_error_correction.py
Recovering true prevalence D_{k,t} from a fallible detector (Rogan-Gladen).

A detector with sensitivity (recall) r and specificity sp flags a fraction of a
corpus. That observed flag rate is NOT the true prevalence pi; it is
    flag_rate = r*pi + (1-sp)*(1-pi).
Reading flag_rate as prevalence overestimates a rare phenomenon, because false
positives dominate at low prevalence. The Rogan-Gladen estimator inverts this:
    pi_hat = (flag_rate - (1-sp)) / (r - (1-sp)),   truncated to [0,1].
It is unbiased when r and sp are known and stable, but its variance inflates as
the denominator (r - (1-sp)) shrinks (poor separation), so it needs many texts
and breaks down when the classifier barely separates the classes.

This makes the honest point for using the detector as a measured input to a
dynamics model: D_{k,t} is defensible as an ERROR-CORRECTED AGGREGATE over many
messages, not as item-level labels (which stay mostly wrong at low prevalence).

No model calls; pure computation. Operating point: r=0.90, sp=0.85 (f=0.15).
"""
import numpy as np

def flag_rate(r, sp, pi): return r*pi + (1-sp)*(1-pi)
def rogan_gladen(fr, r, sp):
    denom = r - (1-sp)
    return np.clip((fr - (1-sp))/denom, 0, 1) if denom > 0 else np.nan
def rg_se(fr, r, sp, n):
    # approximate standard error of the corrected estimate
    denom = r - (1-sp)
    return np.sqrt(fr*(1-fr)/n)/denom if denom > 0 else np.nan

if __name__ == "__main__":
    r, sp = 0.90, 0.85
    print(f"Detector: sensitivity r={r}, specificity sp={sp} (false-alarm rate {1-sp:.2f})")
    print(f"\n{'true pi':>9}{'observed flag rate':>20}{'naive reading':>16}{'Rogan-Gladen':>15}")
    for pi in [0.01, 0.02, 0.05, 0.10, 0.20, 0.40]:
        fr = flag_rate(r, sp, pi)
        print(f"{pi*100:>7.0f}% {fr*100:>18.1f}% {fr*100:>14.1f}% {rogan_gladen(fr,r,sp)*100:>13.1f}%")
    print("\nThe naive flag rate overstates a rare phenomenon (e.g. 18.8% observed when")
    print("true prevalence is 5%); the correction recovers 5%. But precision of the")
    print("correction needs volume:")
    pi = 0.05; fr = flag_rate(r, sp, pi)
    print(f"\n  at true pi=5%, observed flag rate {fr*100:.1f}%, corrected 95% CI half-width:")
    for n in [100, 1000, 10000, 100000]:
        print(f"     n={n:>7}: +/- {1.96*rg_se(fr,r,sp,n)*100:.2f} percentage points")
    print("\nHonest reading: the detector can measure prevalence D_k,t without bias at the")
    print("AGGREGATE level given known, stable r and sp and enough messages; item-level")
    print("labels remain mostly wrong at low prevalence. r and sp must be re-estimated")
    print("per domain, since they drift across corpora.")
