import numpy as np
import pytest

from spxvol.density import density_stats, lognormal_density, rn_density, strike_grid
from spxvol.svi import fit_svi, svi_derivs, svi_g, svi_w

P = np.array([0.005, 0.05, -0.6, 0.0, 0.1])  # ATM ~24%, 20% OTM put ~31% at T=0.25
F, T, D = 5000.0, 0.25, 0.99


def test_fit_recovers_known_parameters():
    k = np.linspace(-0.4, 0.2, 40)
    w = svi_w(k, P)
    np.testing.assert_allclose(svi_w(k, fit_svi(k, w)), w, atol=1e-8)


def test_derivatives_match_finite_difference():
    k, h = np.linspace(-0.3, 0.3, 7), 1e-5
    w, w1, w2 = svi_derivs(k, P)
    fd1 = (svi_w(k + h, P) - svi_w(k - h, P)) / (2 * h)
    fd2 = (svi_w(k + h, P) - 2 * w + svi_w(k - h, P)) / h**2
    np.testing.assert_allclose(w1, fd1, rtol=1e-6)
    np.testing.assert_allclose(w2, fd2, rtol=1e-4)


def test_no_butterfly_arbitrage_for_sane_slice():
    assert np.all(svi_g(np.linspace(-1.5, 1.5, 301), P) >= 0)


def test_flat_svi_reproduces_lognormal():
    sigma = 0.2
    flat = [sigma**2 * T, 0.0, 0.0, 0.0, 0.1]  # b = 0 -> constant total variance
    K = strike_grid(F)
    q = rn_density(flat, F, T, D, K)
    ql = lognormal_density(F, T, sigma, K)
    np.testing.assert_allclose(q, ql, atol=1e-4 * ql.max())


def test_density_integrates_to_one_with_mean_forward():
    K = strike_grid(F)
    q = rn_density(P, F, T, D, K)
    assert np.all(q >= -1e-12)
    s = density_stats(K, q, F)
    assert s["mass"] == pytest.approx(1.0, abs=2e-3)
    assert s["mean"] == pytest.approx(F, rel=2e-3)
    assert s["skew"] < 0  # negative rho -> left-skewed log return
