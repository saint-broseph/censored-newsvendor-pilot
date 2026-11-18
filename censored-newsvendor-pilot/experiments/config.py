"""One place for every parameter used in the experiments."""
import numpy as np

from pilot import BucketSignal, expected_cost, critical_quantile, truncated_negbin

M = 80                       # largest possible demand
MEAN, DISPERSION = 30, 6     # negative binomial demand: mean 30, sd about 13
B, H = 4.0, 1.0              # underage and overage cost per unit: critical ratio 0.8
RHO = B / (B + H)
LAM = 34                     # observable boundary: the largest past order
ORDER_GAP = (8, 4, 0)        # past orders are lam-8, lam-4 and lam, used equally often

CUTS = {                     # bucket cut points for an L-level forecast
    2: [40],                                  # "will demand exceed 40?"
    3: [35, 50],
    5: [20, 30, 40, 50],                      # main signal: a 1-5 score
    8: [10, 20, 30, 40, 50, 60, 70],
}
MAIN_L, MAIN_SIGMA = 5, 4.0

P_TRUE = truncated_negbin(MEAN, DISPERSION, M)
Q_STAR = critical_quantile(P_TRUE, RHO)
C_STAR = expected_cost(Q_STAR, P_TRUE, B, H)   # optimal expected cost: the scale for "% excess cost"
SD = float(np.sqrt(P_TRUE @ np.arange(M + 1) ** 2 - (P_TRUE @ np.arange(M + 1)) ** 2))


N_EFF = 1000                 # population experiments allow for the sampling error of 1,000 periods


def order_levels(lam: int = LAM):
    return [lam - g for g in ORDER_GAP]


def delta_allowance(L: int, lam: int = LAM, n: int = N_EFF) -> float:
    """The same l1 allowance used on real data: m * sqrt(L / n_stockouts),
    with the expected number of stockouts at the largest order in n periods."""
    m = float(P_TRUE[lam:].sum())
    n_stockouts = n / len(ORDER_GAP) * m
    return m * float(np.sqrt(L / n_stockouts))


def signal(L: int = MAIN_L, sigma: float = MAIN_SIGMA, lam: int = LAM, shrink: float = 0.0):
    return BucketSignal(CUTS[L], sigma=sigma, lam=lam, shrink=shrink)
