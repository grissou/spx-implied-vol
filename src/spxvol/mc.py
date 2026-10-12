"""Log-Euler Monte Carlo of k_t = ln(S_t / F_t): dk = -sigma^2/2 dt + sigma dW. Antithetic pairs, control variates."""

import numpy as np
from scipy.stats import norm


def simulate(sigma, t_grid, F_t, n_pairs, seed=0, fix_every=1):
    """sigma(k, j) is the vol over step j. Barrier monitoring and Asian fixings happen every `fix_every`
    steps. Returns terminal S, running min, arithmetic and geometric means over the fixings,
    for 2 n_pairs paths: first half +Z, second half -Z."""
    rng = np.random.default_rng(seed)
    n = len(t_grid) - 1
    k = np.zeros(2 * n_pairs)
    S_min = np.full(2 * n_pairs, F_t[0])
    S_sum = np.zeros(2 * n_pairs)
    logS_sum = np.zeros(2 * n_pairs)
    for j in range(n):
        dt = t_grid[j + 1] - t_grid[j]
        z = rng.standard_normal(n_pairs)
        s = sigma(k, j)
        k = k - 0.5 * s**2 * dt + s * np.sqrt(dt) * np.concatenate([z, -z])
        if (j + 1) % fix_every == 0:
            S = F_t[j + 1] * np.exp(k)
            S_min = np.minimum(S_min, S)
            S_sum += S
            logS_sum += np.log(S)
    m = n // fix_every
    return dict(S_T=S, S_min=S_min, A=S_sum / m, G=np.exp(logS_sum / m))


def estimate(y, control=None, control_mean=None, antithetic=True):
    """Mean and standard error of per-path payoffs y. With a control x of known mean, uses y - beta (x - E x)."""
    if control is not None:
        beta = np.cov(control, y)[0, 1] / np.var(control)
        y = y - beta * (control - control_mean)
    if antithetic:
        h = len(y) // 2
        y = 0.5 * (y[:h] + y[h:])
    return y.mean(), y.std(ddof=1) / np.sqrt(len(y))


def geometric_asian_call(F_t, t_grid, sigma, K, D):
    """Closed form under constant vol for discrete fixings t_1..t_n (pass the fixing grid)."""
    t, lnF = t_grid[1:], np.log(F_t[1:])
    n = len(t)
    V = sigma**2 * np.minimum.outer(t, t).sum() / n**2
    G = np.exp(lnF.mean() - 0.5 * sigma**2 * t.mean() + 0.5 * V)
    d1 = (np.log(G / K) + 0.5 * V) / np.sqrt(V)
    return D * (G * norm.cdf(d1) - K * norm.cdf(d1 - np.sqrt(V)))
