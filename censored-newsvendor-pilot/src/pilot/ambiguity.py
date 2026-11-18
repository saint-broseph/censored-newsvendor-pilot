"""Ambiguity sets: every demand distribution consistent with what we observe.

Censored sales pin down P(D = y) for y below the observable boundary lam (the
largest past order) and the total tail mass m = P(D >= lam). How m is spread
over lam..M is unknown. Writing p_tail for that spread:

    no signal:   P0       = { p_tail >= 0 : sum(p_tail) = m }
    with signal: P(delta) = { p in P0 : || K p_tail - beta ||_1 <= delta }

beta[f] = P(F = f, D >= lam) is the forecast's distribution in stockout
periods, which we observe; K[f, i] = P(F = f | D = lam + i) is the forecast's
error model, which we must assume or estimate. delta = 0 trusts the error
model exactly; delta > 0 allows sampling error or misspecification.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

from .data import History, kaplan_meier
from .signal import BucketSignal, fit_error_model


class EmptyAmbiguitySet(ValueError):
    """No distribution satisfies the constraints at this delta."""


@dataclass
class AmbiguitySet:
    M: int
    lam: int
    p_low: np.ndarray                 # P(D = y), y = 0..lam-1
    tail_mass: float                  # P(D >= lam)
    K: np.ndarray | None = None       # (L, T) error model on the tail
    beta: np.ndarray | None = None    # (L,) P(F = f, D >= lam)
    delta: float = 0.0
    info: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return self.M - self.lam + 1

    @property
    def has_signal(self) -> bool:
        return self.K is not None

    @property
    def L(self) -> int:
        return 0 if self.K is None else self.K.shape[0]

    def without_signal(self) -> "AmbiguitySet":
        return AmbiguitySet(self.M, self.lam, self.p_low, self.tail_mass)

    def with_delta(self, delta: float) -> "AmbiguitySet":
        return AmbiguitySet(self.M, self.lam, self.p_low, self.tail_mass,
                            self.K, self.beta, delta, dict(self.info))

    def contains(self, p: np.ndarray, tol: float = 1e-7) -> bool:
        tail = p[self.lam:]
        ok = (np.allclose(p[:self.lam], self.p_low, atol=tol)
              and abs(tail.sum() - self.tail_mass) <= tol
              and np.all(tail >= -tol))
        if ok and self.has_signal:
            ok = np.abs(self.K @ tail - self.beta).sum() <= self.delta + tol
        return bool(ok)

    def min_delta(self) -> float:
        """Smallest delta for which the set is non-empty."""
        if not self.has_signal:
            return 0.0
        L, T = self.K.shape
        # variables [p_tail (T), u (L)]; minimise sum(u)
        c = np.concatenate([np.zeros(T), np.ones(L)])
        A_ub = np.block([[self.K, -np.eye(L)], [-self.K, -np.eye(L)]])
        b_ub = np.concatenate([self.beta, -self.beta])
        A_eq = np.concatenate([np.ones(T), np.zeros(L)])[None, :]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[self.tail_mass],
                      bounds=(0, None), method="highs")
        return float(res.fun)

    # ------------------------------------------------------------ builders
    @classmethod
    def population(cls, p: np.ndarray, lam: int, signal: BucketSignal | None = None,
                   model: BucketSignal | None = None, delta: float = 0.0) -> "AmbiguitySet":
        """Infinite-data version. beta comes from the true signal; K from the
        analyst's model (the true signal unless a different model is given)."""
        M = len(p) - 1
        p_low, m = p[:lam].copy(), float(p[lam:].sum())
        if signal is None:
            return cls(M, lam, p_low, m)
        ys = np.arange(lam, M + 1)
        beta = signal.kernel(ys) @ p[lam:]
        K = (model or signal).kernel(ys)
        return cls(M, lam, p_low, m, K, beta, delta)

    @classmethod
    def from_history(cls, hist: History, M: int, cuts=None, delta: float | None = None,
                     z: float = 1.0, sigma_min: float = 0.25) -> "AmbiguitySet":
        """Estimate the set from data.

        * p_low and the tail mass: Kaplan-Meier on all periods.
        * error model (mu, sigma): maximum likelihood on uncensored periods,
          then extrapolated to the tail (the untestable assumption).
        * beta: tail mass times the forecast distribution among stockouts at
          the largest order lam.
        * delta (if not given): z * m * sqrt(L / n_stockouts), a rough l1
          sampling allowance, raised if needed so the set is non-empty.
        """
        lam = int(hist.order.max())
        p_low, m = kaplan_meier(hist.sales, hist.stockout, lam)
        if cuts is None:
            return cls(M, lam, p_low, m)
        seen = ~hist.stockout
        mu, sigma = fit_error_model(hist.sales[seen], hist.signal[seen], cuts, sigma_min)
        model = BucketSignal(cuts, mu=mu, sigma=sigma, lam=lam)
        K = model.kernel(np.arange(lam, M + 1))
        stockouts_at_lam = (hist.order == lam) & hist.stockout
        n_s = int(stockouts_at_lam.sum())
        counts = np.bincount(hist.signal[stockouts_at_lam], minlength=model.L)
        beta = m * counts / max(n_s, 1)
        amb = cls(M, lam, p_low, m, K, beta, 0.0,
                  {"mu_hat": mu, "sigma_hat": sigma, "n_stockouts": n_s})
        d_min = amb.min_delta()
        if delta is None:
            delta = z * m * np.sqrt(model.L / max(n_s, 1))
        amb.delta = max(delta, d_min * 1.001 + 1e-9)
        amb.info["delta_min"] = d_min
        return amb
