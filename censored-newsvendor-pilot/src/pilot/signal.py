"""The AI signal: a coarse, noisy demand forecast.

A forecasting model predicts demand and we only keep which bucket the
prediction falls in (e.g. "low / medium / high"), which mimics the discrete
scores an LLM produces. With cut points c_1 < ... < c_{L-1} the signal is

    F = #{ cuts <= X },   X = centre(D) + sigma * Z,   Z ~ N(0, 1)

where centre(D) = D + mu - shrink * (D - lam)^+.

* mu is a forecast bias and sigma the forecast noise.
* shrink > 0 makes the forecast under-predict high demand. It is zero below
  lam, so it is invisible in uncensored data: we use it to study what happens
  when the analyst's error model is wrong exactly where it cannot be checked.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats


@dataclass
class BucketSignal:
    cuts: np.ndarray
    mu: float = 0.0
    sigma: float = 0.0
    shrink: float = 0.0
    lam: int = 0
    edges: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.cuts = np.asarray(self.cuts, dtype=float)
        self.edges = np.concatenate(([-np.inf], self.cuts, [np.inf]))

    @property
    def L(self) -> int:
        return len(self.cuts) + 1

    def centre(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        return y + self.mu - self.shrink * np.maximum(y - self.lam, 0.0)

    def kernel(self, ys: np.ndarray) -> np.ndarray:
        """K[f, i] = P(F = f | D = ys[i]); columns sum to one."""
        c = self.centre(ys)
        if self.sigma == 0:
            f = np.searchsorted(self.cuts, c, side="right")
            K = np.zeros((self.L, len(c)))
            K[f, np.arange(len(c))] = 1.0
            return K
        cdf = stats.norm.cdf((self.edges[:, None] - c[None, :]) / self.sigma)
        return np.diff(cdf, axis=0)

    def sample(self, d: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        x = self.centre(d) + self.sigma * rng.standard_normal(len(d))
        return np.searchsorted(self.cuts, x, side="right")


def fit_error_model(y: np.ndarray, f: np.ndarray, cuts: np.ndarray,
                    sigma_min: float = 0.25) -> tuple[float, float]:
    """Maximum-likelihood estimate of (mu, sigma) from pairs where demand was
    observed (no stockout). Assumes the shift model with no shrinkage."""
    y = np.asarray(y, dtype=float)
    f = np.asarray(f, dtype=int)
    edges = np.concatenate(([-np.inf], np.asarray(cuts, float), [np.inf]))
    lo, hi = edges[f], edges[f + 1]

    def nll(theta):
        mu, log_s = theta
        s = sigma_min + np.exp(log_s)
        prob = stats.norm.cdf((hi - y - mu) / s) - stats.norm.cdf((lo - y - mu) / s)
        return -np.sum(np.log(np.clip(prob, 1e-300, None)))

    best = None
    for start in ((0.0, np.log(2.0)), (0.0, np.log(8.0))):
        res = optimize.minimize(nll, start, method="Nelder-Mead",
                                options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
        if best is None or res.fun < best.fun:
            best = res
    mu, log_s = best.x
    return float(mu), float(sigma_min + np.exp(log_s))
