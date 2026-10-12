"""Breeden-Litzenberger density q(K) = (1/D) d2C/dK2 from SVI call prices on a fine strike grid."""

import numpy as np
from scipy.stats import norm

from .black import black_price
from .svi import svi_w


def strike_grid(F, lo=0.05, hi=4.0, n=8001):
    return np.linspace(lo * F, hi * F, n)


def rn_density(p, F, T, D, K):
    """Central differences of SVI-implied call prices."""
    sigma = np.sqrt(svi_w(np.log(K / F), p) / T)
    C = black_price(F, K, T, sigma, D, True)
    return np.gradient(np.gradient(C, K), K) / D


def lognormal_density(F, T, sigma, K):
    """Constant-vol benchmark with the same forward."""
    sT = sigma * np.sqrt(T)
    d2 = (np.log(F / K) - 0.5 * sT**2) / sT
    return norm.pdf(d2) / (K * sT)


def density_stats(K, q, F):
    """Mass, mean, log-return moments and left-tail probabilities."""
    tr = np.trapezoid
    x = np.log(K / F)
    mass = tr(q, K)
    mu = tr(x * q, K) / mass
    cm = lambda n: tr((x - mu) ** n * q, K) / mass
    var = cm(2)
    return {
        "mass": mass,
        "mean": tr(K * q, K) / mass,
        "log_vol": np.sqrt(var),
        "skew": cm(3) / var**1.5,
        "ex_kurt": cm(4) / var**2 - 3,
        "P(S<0.9F)": tr(q[K < 0.9 * F], K[K < 0.9 * F]),
        "P(S<0.8F)": tr(q[K < 0.8 * F], K[K < 0.8 * F]),
    }
