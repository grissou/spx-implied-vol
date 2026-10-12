import numpy as np
import pytest

from spxvol.black import black_price, black_vega, implied_vol, implied_vol_vec

F, T, D = 3300.0, 0.5, 0.99


def test_put_call_parity():
    K = np.array([2500.0, 3300.0, 4000.0])
    sigma = 0.25
    c = black_price(F, K, T, sigma, D, True)
    p = black_price(F, K, T, sigma, D, False)
    np.testing.assert_allclose(c - p, D * (F - K), atol=1e-8)


def test_price_bounds_and_monotone_in_vol():
    K = 3000.0
    sig = np.linspace(0.05, 2.0, 50)
    c = black_price(F, K, T, sig, D, True)
    assert np.all(np.diff(c) > 0)
    assert c[0] > D * (F - K) and c[-1] < D * F


def test_zero_vol_is_intrinsic():
    assert black_price(F, 3000.0, T, 0.0, D, True) == pytest.approx(D * 300.0)
    assert black_price(F, 3000.0, T, 0.0, D, False) == 0.0


def test_vega_matches_finite_difference():
    K, s, h = 3100.0, 0.3, 1e-5
    fd = (black_price(F, K, T, s + h, D) - black_price(F, K, T, s - h, D)) / (2 * h)
    assert black_vega(F, K, T, s, D) == pytest.approx(fd, rel=1e-6)


@pytest.mark.parametrize("K", [2200.0, 3000.0, 3300.0, 3600.0, 4500.0])
@pytest.mark.parametrize("sigma", [0.08, 0.2, 0.6, 1.5])
def test_implied_vol_round_trip_otm(K, sigma):
    # OTM side only (puts below F, calls above), as in the pipeline: deep ITM options
    # have time value below float precision and carry no vol information.
    is_call = K >= F
    price = black_price(F, K, T, sigma, D, is_call)
    assert implied_vol(price, F, K, T, D, is_call) == pytest.approx(sigma, abs=1e-8)


def test_implied_vol_outside_bounds_is_nan():
    K = 3000.0
    assert np.isnan(implied_vol(D * (F - K) - 1.0, F, K, T, D, True))  # below intrinsic
    assert np.isnan(implied_vol(D * F + 1.0, F, K, T, D, True))        # above forward


def test_vectorised_wrapper():
    K = np.array([3000.0, 3300.0, 3600.0])
    sig = np.array([0.3, 0.2, 0.25])
    px = black_price(F, K, T, sig, D, True)
    np.testing.assert_allclose(implied_vol_vec(px, F, K, T, D, True), sig, atol=1e-8)
