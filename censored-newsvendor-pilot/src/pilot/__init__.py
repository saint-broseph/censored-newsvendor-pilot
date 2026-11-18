"""How much is an AI forecast worth when sales are censored?

A pilot study of the robust newsvendor with an AI signal used as a weak
shadow variable.
"""
from .ambiguity import AmbiguitySet, EmptyAmbiguitySet
from .data import History, kaplan_meier, sales_pmf, sales_quantile, simulate_history
from .demand import sample_demand, truncated_negbin
from .newsvendor import (critical_quantile, critical_ratio, expected_cost,
                         hssaine_sinclair, regret, unit_costs)
from .robust import RobustSolution, solve_robust, worst_case
from .signal import BucketSignal, fit_error_model

__all__ = [
    "AmbiguitySet", "EmptyAmbiguitySet", "History", "kaplan_meier", "sales_pmf", "sales_quantile",
    "simulate_history", "sample_demand", "truncated_negbin", "critical_quantile",
    "critical_ratio", "expected_cost", "hssaine_sinclair", "regret", "unit_costs",
    "RobustSolution", "solve_robust", "worst_case", "BucketSignal", "fit_error_model",
]
