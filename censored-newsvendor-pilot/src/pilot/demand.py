"""Demand distributions on the integers 0..M."""
from __future__ import annotations

import numpy as np
from scipy import stats


def truncated_negbin(mean: float, k: float, M: int) -> np.ndarray:
    """Negative binomial with the given mean and dispersion k, truncated to
    0..M and renormalised. Variance (before truncation) = mean + mean^2 / k."""
    p_success = k / (k + mean)
    pmf = stats.nbinom.pmf(np.arange(M + 1), k, p_success)
    return pmf / pmf.sum()


def sample_demand(p: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(len(p), size=n, p=p)
