"""Statistics for paired comparisons between training conditions.

Runs with the same seed share initial weights, data order and augmentations, so a
condition-A run and a condition-B run with the same seed form a pair, and so do the
two predictions for the same test image. Uncertainty therefore has two sources:

* training randomness  (which seeds were drawn), and
* test-set sampling    (which images were drawn).

`paired_bootstrap` resamples both: each replicate draws seeds with replacement and,
independently, test images with replacement, and recomputes the mean difference.
With 3 seeds the seed level is coarse, so we also report every per-seed difference.
"""
import numpy as np


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0,
                     alpha: float = 0.05) -> dict:
    """a, b: [n_seeds, n_images] metric values (rows = matched seeds, cols = matched images).

    Returns the mean of (a - b), a percentile CI from the two-level bootstrap, and the
    per-seed mean differences.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    d = a - b
    keep = ~np.isnan(d).any(axis=0)          # drop images with NaN in any seed (no edge pixels)
    d = d[:, keep]
    s, n = d.shape
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        si = rng.integers(0, s, s)
        ii = rng.integers(0, n, n)
        boots[i] = d[np.ix_(si, ii)].mean()
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "per_seed_diff": [float(x) for x in d.mean(axis=1)], "n_images": int(n),
            "n_seeds": int(s), "n_dropped_nan": int((~keep).sum())}


def wilcoxon_images(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value on per-image differences averaged over seeds.

    This test treats the images as the sampling unit; it ignores seed variance, so it is
    reported next to the bootstrap CI and not on its own. Returns NaN without SciPy.
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return float("nan")
    d = (np.asarray(a, float) - np.asarray(b, float)).mean(axis=0)
    d = d[~np.isnan(d)]
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(d).pvalue)


def holm(pvalues: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for a family {name: p}."""
    names = sorted(pvalues, key=lambda k: pvalues[k])
    m = len(names)
    adjusted, running = {}, 0.0
    for i, k in enumerate(names):
        p = pvalues[k]
        running = max(running, min(1.0, (m - i) * p)) if not np.isnan(p) else float("nan")
        adjusted[k] = running
    return adjusted


def mean_sd(values) -> tuple:
    v = np.asarray(values, float)
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else float("nan")
