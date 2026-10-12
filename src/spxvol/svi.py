"""Raw SVI for one expiry: w(k) = a + b [rho (k - m) + sqrt((k - m)^2 + sigma^2)], k = ln(K/F)."""

import numpy as np
from scipy.optimize import least_squares


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
    """Gatheral-Jacquier butterfly function: no butterfly arbitrage iff g >= 0."""
    w, w1, w2 = svi_derivs(k, p)
    return (1 - k * w1 / (2 * w)) ** 2 - w1**2 / 4 * (1 / w + 0.25) + w2 / 2


def fit_svi(k, w, weights=None):
    """Weighted least squares in total variance. Bounds a >= 0 and m inside the quoted
    k range stop the fit landing on the far branch of the hyperbola (rho ~ +1)."""
    k, w = np.asarray(k, float), np.asarray(w, float)
    sw = np.ones_like(k) if weights is None else np.sqrt(np.asarray(weights, float))
    lb = [0.0, 0.0, -0.999, k.min(), 1e-4]
    ub = [np.inf, np.inf, 0.999, k.max(), 2.0]

    def resid(p):
        return sw * (svi_w(k, p) - w)

    best = None
    for rho0 in (-0.7, -0.3, 0.0):
        for m0 in (-0.1, 0.0, 0.1):
            x0 = [0.5 * w.min(), 0.1, rho0, np.clip(m0, k.min(), k.max()), 0.1]
            res = least_squares(resid, x0, bounds=(lb, ub))
            if best is None or res.cost < best.cost:
                best = res
    return best.x
