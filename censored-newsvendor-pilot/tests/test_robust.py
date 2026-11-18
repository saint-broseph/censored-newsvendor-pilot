"""The robust LP against two independent references:

1. the Hssaine-Sinclair closed form (no signal), and
2. brute force: the inner primal LPs solved directly (strong duality check),
   the robust-LP analogue of checking an MPC engine against a plaintext oracle.
"""
import numpy as np
import pytest

from pilot import (AmbiguitySet, BucketSignal, EmptyAmbiguitySet, hssaine_sinclair,
                   regret, solve_robust, truncated_negbin, worst_case)


@pytest.mark.parametrize("mean,lam,b,h", [
    (12, 12, 4, 1), (12, 10, 9, 1), (15, 14, 3, 2), (10, 9, 1, 1), (18, 16, 19, 1),
])
def test_matches_hssaine_sinclair_closed_form(mean, lam, b, h):
    M = 40
    p = truncated_negbin(mean, 5, M)
    amb = AmbiguitySet.population(p, lam)
    F0 = p[:lam].sum()
    q_hs, delta_hs = hssaine_sinclair(F0, lam, M, b, h)
    assert q_hs is not None, "test case should be in the unidentifiable regime"
    sol = solve_robust(amb, b, h)
    assert sol.q == pytest.approx(q_hs, abs=1e-6)
    assert sol.value == pytest.approx(delta_hs, rel=1e-7, abs=1e-7)


def test_identifiable_regime_has_zero_minimax_regret():
    M = 40
    p = truncated_negbin(10, 5, M)
    lam, b, h = 25, 1, 1                  # F0 = P(D < 25) >> rho = 0.5
    amb = AmbiguitySet.population(p, lam)
    assert hssaine_sinclair(p[:lam].sum(), lam, M, b, h) == (None, 0.0)
    sol = solve_robust(amb, b, h)
    assert sol.value == pytest.approx(0.0, abs=1e-8)
    assert regret(sol.q, p, b, h) == pytest.approx(0.0, abs=1e-8)


def small_instance(sigma=2.0, delta=0.0, shrink=0.0):
    M, lam = 25, 9
    p = truncated_negbin(9, 4, M)
    sig = BucketSignal([6, 11, 16], sigma=sigma, shrink=shrink, lam=lam)
    model = BucketSignal([6, 11, 16], sigma=sigma, lam=lam)
    return p, AmbiguitySet.population(p, lam, sig, model=model, delta=delta)


@pytest.mark.parametrize("criterion", ["regret", "cost"])
@pytest.mark.parametrize("sigma,delta", [(0.0, 0.0), (2.0, 0.0), (2.0, 0.05), (6.0, 0.02)])
def test_single_lp_equals_brute_force(criterion, sigma, delta):
    _, amb = small_instance(sigma, delta)
    b, h = 4, 1
    sol = solve_robust(amb, b, h, criterion)
    # strong duality: the certificate equals the primal worst case at q
    val, p_worst = worst_case(amb, sol.q, b, h, criterion)
    assert val == pytest.approx(sol.value, abs=1e-6)
    assert amb.contains(p_worst, tol=1e-6)
    # optimality: no other order does better in the worst case
    for q in np.linspace(0, amb.M, 101):
        assert worst_case(amb, q, b, h, criterion)[0] >= sol.value - 1e-6


def test_certificate_covers_true_regret_and_signal_helps():
    p, amb = small_instance(sigma=2.0)
    b, h = 4, 1
    assert amb.contains(p)
    sig = solve_robust(amb, b, h)
    nosig = solve_robust(amb.without_signal(), b, h)
    assert sig.value <= nosig.value + 1e-9
    assert sig.value >= regret(sig.q, p, b, h) - 1e-9


def test_certificate_grows_with_delta():
    _, amb = small_instance(sigma=3.0)
    values = [solve_robust(amb.with_delta(d), 4, 1).value for d in (0, 0.01, 0.05, 0.2, 1.0)]
    assert np.all(np.diff(values) >= -1e-9)
    assert values[-1] == pytest.approx(solve_robust(amb.without_signal(), 4, 1).value, abs=1e-7)


def test_misspecified_model_can_exclude_the_truth():
    p, amb = small_instance(sigma=1.0, shrink=0.6)
    delta_star = np.abs(amb.K @ p[amb.lam:] - amb.beta).sum()
    assert delta_star > 0.01
    assert not amb.contains(p)
    assert amb.with_delta(delta_star + 1e-9).contains(p, tol=1e-6)
    if amb.min_delta() > 1e-9:
        with pytest.raises(EmptyAmbiguitySet):
            solve_robust(amb, 4, 1)
