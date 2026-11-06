# -*- coding: utf-8 -*-
"""
prevalence_utility.py
How detector precision and net utility degrade as the phenomenon becomes rare.

Pure computation, no model calls. Given a detector's recall r = P(flag|positive)
and false-alarm rate f = P(flag|negative), the precision at prevalence pi is
    precision(pi) = r*pi / (r*pi + f*(1-pi)).
On a balanced set (pi=0.5) precision looks good; at realistic low prevalence it
collapses. The tool is net-positive only where precision >= C/(B+C), with C the
cost of a false accusation and B the benefit of a correct catch; this gives a
break-even prevalence below which the tool does more harm than good as a verdict.

Operating points are the held-out numbers: recall 54/60 = 0.90; false-alarm rate
16/60 = 0.267 for v2, 9/60 = 0.15 for v2 + negation verifier.
"""
import numpy as np

def precision(r, f, pi):
    return r*pi / (r*pi + f*(1-pi)) if (r*pi + f*(1-pi)) else 0.0

def breakeven_prevalence(r, f, prec_needed):
    a = r*(1-prec_needed); b = prec_needed*f
    return b/(a+b)

R = 54/60
CONFIGS = {"v2": 16/60, "v2+verifier": 9/60}

if __name__ == "__main__":
    print("Precision by prevalence (recall 0.90):")
    print(f"{'prevalence':>12}" + "".join(f"{k:>16}" for k in CONFIGS))
    for pi in [0.50, 0.20, 0.10, 0.05, 0.02, 0.01]:
        print(f"{pi*100:>10.0f}% " + "".join(f"{precision(R,f,pi)*100:>15.1f}%" for f in CONFIGS.values()))
    print("\nBreak-even prevalence (deployed config f=0.15), tool net-positive only above:")
    for name, pn in [("C=B", 0.5), ("C=2B", 2/3), ("C=0.5B", 1/3)]:
        print(f"  {name:8s} needs precision {pn:.2f} -> {breakeven_prevalence(R,9/60,pn)*100:.1f}% prevalence")
