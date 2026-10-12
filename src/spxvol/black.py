"""Black (1976) option pricing on the forward, and implied volatility by Brent's method.

Everything is expressed in terms of the forward F and discount factor D = e^{-rT}
rather than spot, r and q. On SPX we back F and D out of put-call parity per expiry,
so dividends and rates never need to be modelled separately.

    C = D [F N(d1) - K N(d2)],   P = D [K N(-d2) - F N(-d1)]
    d1 = (ln(F/K) + sigma^2 T / 2) / (sigma sqrt T),   d2 = d1 - sigma sqrt T
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

SIGMA_MIN, SIGMA_MAX = 1e-4, 5.0


def black_price(F, K, T, sigma, D=1.0, is_call=True):
    """Black-76 price. All arguments broadcast; sigma=0 returns discounted intrinsic."""
    F, K, T, sigma, D = map(np.asarray, (F, K, T, sigma, D))
    is_call = np.asarray(is_call, dtype=bool)
    sign = np.where(is_call, 1.0, -1.0)
    sT = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * sT**2) / sT
        d2 = d1 - sT
    price = D * sign * (F * norm.cdf(sign * d1) - K * norm.cdf(sign * d2))
    intrinsic = D * np.maximum(sign * (F - K), 0.0)
    return np.where(sT > 0, price, intrinsic)


def black_vega(F, K, T, sigma, D=1.0):
    """dPrice/dsigma, identical for calls and puts."""
    sT = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sT**2) / sT
    return D * F * norm.pdf(d1) * np.sqrt(T)


def implied_vol(price, F, K, T, D=1.0, is_call=True):
    """Invert Black-76 for sigma with Brent's method (scalar inputs).

    Returns NaN if the price lies outside the no-arbitrage bounds
    [discounted intrinsic, D*F for calls / D*K for puts], where no IV exists.
    """
    sign = 1.0 if is_call else -1.0
    lower = D * max(sign * (F - K), 0.0)
    upper = D * (F if is_call else K)
    if not (lower < price < upper):
        return np.nan
    f = lambda s: black_price(F, K, T, s, D, is_call) - price
    if f(SIGMA_MIN) > 0 or f(SIGMA_MAX) < 0:
        return np.nan
    return brentq(f, SIGMA_MIN, SIGMA_MAX, xtol=1e-10)


implied_vol_vec = np.vectorize(implied_vol, otypes=[float])
