"""Newsvendor basics: costs, the critical quantile, regret, and the
Hssaine-Sinclair closed form for the censored (no-signal) case.

Demand D takes integer values 0..M. Ordering q costs
    b * (D - q)^+   (underage: lost sales)
  + h * (q - D)^+   (overage: leftover stock)
and the critical ratio is rho = b / (b + h).
"""
from __future__ import annotations

import numpy as np


def critical_ratio(b: float, h: float) -> float:
    return b / (b + h)


def unit_costs(q: float, M: int, b: float, h: float) -> np.ndarray:
    """Cost of ordering q for every demand value y = 0..M."""
    y = np.arange(M + 1)
    return b * np.maximum(y - q, 0.0) + h * np.maximum(q - y, 0.0)


def expected_cost(q: float, p: np.ndarray, b: float, h: float) -> float:
    return float(p @ unit_costs(q, len(p) - 1, b, h))


def critical_quantile(p: np.ndarray, rho: float, tol: float = 1e-12) -> int:
    """Smallest integer q with P(D <= q) >= rho: the optimal order."""
    cdf = np.cumsum(p)
    return int(np.searchsorted(cdf, rho - tol, side="left"))


def regret(q: float, p: np.ndarray, b: float, h: float) -> float:
    """Expected cost of q minus the best achievable expected cost under p."""
    q_star = critical_quantile(p, critical_ratio(b, h))
    return expected_cost(q, p, b, h) - expected_cost(q_star, p, b, h)


def hssaine_sinclair(F0: float, lam: int, M: int, b: float, h: float):
    """Closed-form minimax-regret order and minimax risk under censoring.

    F0 = P(D < lam) is identified from censored sales; nothing is known about
    how the remaining mass 1 - F0 is spread over lam..M.

    Returns (q_dagger, Delta). In the identifiable regime F0 >= rho the
    optimal order is identified and Delta = 0 (q_dagger is then None:
    use the critical quantile of the known part).

    Source: Hssaine & Sinclair (2024), "The Data-Driven Censored Newsvendor
    Problem", arXiv:2412.01763.
    """
    rho = critical_ratio(b, h)
    if F0 >= rho:
        return None, 0.0
    q = (b * M + h * lam - (b + h) * F0 * M) / ((b + h) * (1 - F0))
    delta = h * (b - (b + h) * F0) * (M - lam) / ((b + h) * (1 - F0))
    return q, delta
