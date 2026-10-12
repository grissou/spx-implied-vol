"""Raw SVI (Gatheral 2004) for one expiry: total variance w(k) in log-moneyness k = ln(K/F).

    w(k) = a + b [ rho (k - m) + sqrt((k - m)^2 + sigma^2) ],   b >= 0, |rho| < 1, sigma > 0

Butterfly (no-arbitrage) check is Gatheral & Jacquier (2014) g(k) >= 0, which is
proportional to the risk-neutral density in k.
"""

import numpy as np
from scipy.optimize import least_squares

LB = [-np.inf, 0.0, -0.999, -2.0, 1e-4]
UB = [np.inf, np.inf, 0.999, 2.0, 5.0]


def svi_w(k, p):
    a, b, rho, m, s = p
    x = k - m
    return a + b * (rho * x + np.sqrt(x**2 + s**2))


def svi_derivs(k, p):
    """w, dw/dk, d2w/dk2."""
    a, b, rho, m, s = p
    x = k - m
    r = np.sqrt(x**2 + s**2)
    return a + b * (rho * x + r), b * (rho + x / r), b * s**2 / r**3


def svi_g(k, p):
    """Gatheral's butterfly function; slice is free of butterfly arbitrage iff g >= 0."""
    w, w1, w2 = svi_derivs(k, p)
    return (1 - k * w1 / (2 * w)) ** 2 - w1**2 / 4 * (1 / w + 0.25) + w2 / 2


def fit_svi(k, w, weights=None):
    """Least-squares fit of raw SVI to market total variances. Returns (a, b, rho, m, sigma)."""
    k, w = np.asarray(k, float), np.asarray(w, float)
    sw = np.ones_like(k) if weights is None else np.sqrt(np.asarray(weights, float))

    def resid(p):
        return sw * (svi_w(k, p) - w)

    best = None
    for rho0 in (-0.7, -0.3, 0.0):
        for m0 in (-0.1, 0.0, 0.1):
            x0 = [0.5 * w.min(), 0.1, rho0, m0, 0.1]
            res = least_squares(resid, x0, bounds=(LB, UB))
            if best is None or res.cost < best.cost:
                best = res
    return best.x
