"""Small statistics helpers: bootstrap confidence intervals over prompts."""
import numpy as np

_RNG = np.random.default_rng(0)

def bootstrap_ci(values, n_boot=10000, alpha=0.05):
    """Mean and (lo, hi) percentile bootstrap CI of a 1-D sample."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), (float("nan"), float("nan"))
    if len(v) == 1:
        return float(v[0]), (float(v[0]), float(v[0]))
    idx = _RNG.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), (float(lo), float(hi))


def median_iqr(values):
    """Median and inter-quartile range (q25, q75)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)), (float(np.percentile(v, 25)), float(np.percentile(v, 75)))
