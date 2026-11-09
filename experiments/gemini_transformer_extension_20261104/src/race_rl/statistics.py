"""Paired instance/seed summaries. No results are synthesized when records are absent."""
import math
import numpy as np


def paired_summary(left,right,*,bootstrap_samples=10000,seed=0):
    a,b = np.asarray(left,dtype=float),np.asarray(right,dtype=float)
    if a.ndim != 1 or b.shape != a.shape or len(a) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("at least two complete finite pairs required")
    if bootstrap_samples < 100:
        raise ValueError("insufficient bootstrap replicates")
    delta = a-b
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0,len(delta),size=(bootstrap_samples,len(delta)))].mean(1)
    low,high = np.quantile(means,[0.025,0.975])
    return {"pairs":len(a),"paired_mean_difference":float(delta.mean()),
            "paired_median_difference":float(np.median(delta)),"paired_std":float(delta.std(ddof=1)),
            "paired_bootstrap_ci95":[float(low),float(high)],"bootstrap_samples":bootstrap_samples,
            "bootstrap_seed":seed,"interpretation":"lower normalized PPA_merit is better; negative left-minus-right favors left"}


def holm_adjust(p_values):
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in p_values):
        raise ValueError("invalid p-values")
    result = [0.0]*len(p_values)
    maximum = 0
    for rank,index in enumerate(sorted(range(len(p_values)),key=lambda i:p_values[i])):
        maximum = max(maximum,min(1,(len(p_values)-rank)*p_values[index]))
        result[index] = maximum
    return result
