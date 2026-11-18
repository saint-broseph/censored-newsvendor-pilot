import numpy as np

from pilot import critical_quantile, expected_cost, regret, unit_costs


def test_unit_costs():
    c = unit_costs(3, 6, b=4, h=1)
    assert np.allclose(c, [3, 2, 1, 0, 4, 8, 12])


def test_critical_quantile_and_zero_regret_at_optimum():
    p = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    q_star = critical_quantile(p, 0.8)            # cdf = .1 .3 .6 .85 1
    assert q_star == 3
    assert abs(regret(q_star, p, 4, 1)) < 1e-12
    for q in np.linspace(0, 4, 41):
        assert regret(q, p, 4, 1) >= -1e-12


def test_expected_cost_is_convex_piecewise_linear():
    p = np.array([0.2, 0.5, 0.3])
    costs = [expected_cost(q, p, 2, 1) for q in np.linspace(0, 2, 21)]
    assert np.all(np.diff(costs, 2) >= -1e-12)
