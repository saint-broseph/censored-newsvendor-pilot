"""Simulated sales history and the Kaplan-Meier estimator for censored demand."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .demand import sample_demand
from .signal import BucketSignal


@dataclass
class History:
    order: np.ndarray      # q_t, order quantity in period t
    sales: np.ndarray      # min(D_t, q_t)
    stockout: np.ndarray   # True if D_t >= q_t (demand censored)
    signal: np.ndarray     # F_t, the AI forecast bucket
    demand: np.ndarray     # D_t: hidden from every estimator, kept for tests

    @property
    def n(self) -> int:
        return len(self.order)


def simulate_history(p: np.ndarray, order_levels, n: int, signal: BucketSignal,
                     rng: np.random.Generator) -> History:
    """n periods; each period's order is drawn uniformly from order_levels,
    independently of demand and of the forecast."""
    d = sample_demand(p, n, rng)
    q = rng.choice(np.asarray(order_levels), size=n)
    f = signal.sample(d, rng)
    return History(order=q, sales=np.minimum(d, q), stockout=d >= q,
                   signal=f, demand=d)


def kaplan_meier(sales: np.ndarray, stockout: np.ndarray, upto: int):
    """Product-limit estimate of P(D = y) for y < upto, and of P(D >= upto).

    A period with sales s and no stockout shows D = s. A stockout period with
    sales s shows only D >= s: we cannot tell D = s from D > s, so it is at
    risk only for y < s (censored "just before" s) and is never an event.
    """
    p_hat = np.zeros(upto)
    surv = 1.0                                  # P(D >= y) at the current y
    for y in range(upto):
        at_risk = np.sum((sales > y) | ((sales == y) & ~stockout))
        events = np.sum((sales == y) & ~stockout)
        hazard = events / at_risk if at_risk > 0 else 0.0
        p_hat[y] = surv * hazard
        surv *= 1.0 - hazard
    return p_hat, surv


def sales_pmf(p: np.ndarray, order_levels) -> np.ndarray:
    """Distribution of sales when each order level is used equally often."""
    out = np.zeros(len(p))
    for q in order_levels:
        out[:q] += p[:q]
        out[q] += p[q:].sum()
    return out / len(order_levels)


def sales_quantile(sales: np.ndarray, rho: float) -> int:
    """The naive policy: treat sales as if they were demand."""
    s = np.sort(sales)
    return int(s[int(np.ceil(rho * len(s))) - 1])
