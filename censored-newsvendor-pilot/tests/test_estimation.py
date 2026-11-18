import numpy as np
import pytest

from pilot import (AmbiguitySet, BucketSignal, fit_error_model, kaplan_meier, sales_quantile,
                   simulate_history, truncated_negbin)


def test_kaplan_meier_without_censoring_is_the_empirical_pmf():
    sales = np.array([0, 1, 1, 2, 2, 2, 3, 4])
    p_hat, tail = kaplan_meier(sales, np.zeros(8, bool), upto=3)
    assert np.allclose(p_hat, [1 / 8, 2 / 8, 3 / 8])
    assert tail == pytest.approx(2 / 8)


def test_kaplan_meier_recovers_censored_demand():
    rng = np.random.default_rng(0)
    p = truncated_negbin(20, 5, 60)
    sig = BucketSignal([20, 30], sigma=3.0, lam=24)
    hist = simulate_history(p, [16, 20, 24], 60_000, sig, rng)
    p_hat, tail = kaplan_meier(hist.sales, hist.stockout, 24)
    assert np.abs(p_hat - p[:24]).max() < 0.006
    assert tail == pytest.approx(p[24:].sum(), abs=0.01)


def test_kernel_columns_sum_to_one_and_match_sampling():
    rng = np.random.default_rng(1)
    sig = BucketSignal([10, 20, 30], sigma=4.0, mu=1.0)
    K = sig.kernel(np.arange(41))
    assert np.allclose(K.sum(axis=0), 1.0)
    d = np.full(200_000, 18)
    freq = np.bincount(sig.sample(d, rng), minlength=4) / len(d)
    assert np.allclose(freq, K[:, 18], atol=0.005)
    K0 = BucketSignal([10, 20], sigma=0.0).kernel(np.array([9, 10, 19, 20]))
    assert np.array_equal(K0.argmax(axis=0), [0, 1, 1, 2])


def test_mle_recovers_error_model():
    rng = np.random.default_rng(2)
    cuts = np.array([10, 15, 20, 25, 30])
    y = rng.integers(5, 35, 30_000)
    f = BucketSignal(cuts, mu=1.5, sigma=4.0).sample(y, rng)
    mu, sigma = fit_error_model(y, f, cuts)
    assert mu == pytest.approx(1.5, abs=0.15)
    assert sigma == pytest.approx(4.0, abs=0.15)


def test_estimated_set_contains_truth_in_large_samples():
    rng = np.random.default_rng(3)
    M, lam = 50, 22
    p = truncated_negbin(18, 5, M)
    cuts = [15, 25, 35]
    hist = simulate_history(p, [14, 18, lam], 80_000, BucketSignal(cuts, sigma=3.0, lam=lam), rng)
    amb = AmbiguitySet.from_history(hist, M, cuts, z=2.0)
    assert amb.lam == lam
    assert amb.info["sigma_hat"] == pytest.approx(3.0, abs=0.2)
    assert np.abs(amb.p_low - p[:lam]).max() < 0.006
    tail = p[lam:] * amb.tail_mass / p[lam:].sum()     # truth rescaled to the estimated mass
    assert np.abs(amb.K @ tail - amb.beta).sum() <= amb.delta


def test_sales_quantile():
    assert sales_quantile(np.array([1, 2, 3, 4, 5]), 0.8) == 4
    assert sales_quantile(np.array([1, 2, 3, 4, 5]), 0.81) == 5
