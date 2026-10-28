import numpy as np

from spxvol.black import black_price
from spxvol.mc import estimate, geometric_asian_call, simulate

F, T, D, SIGMA = 3300.0, 0.5, 0.99, 0.25
T_GRID = np.linspace(0.0, T, 51)
F_T = np.full(len(T_GRID), F)


def flat(k, j):
    return SIGMA


def test_flat_vol_call_matches_black_within_3_se():
    paths = simulate(flat, T_GRID, F_T, n_pairs=50_000, seed=1)
    for K in (3000.0, 3300.0, 3600.0):
        mean, se = estimate(D * np.maximum(paths["S_T"] - K, 0.0))
        assert abs(mean - black_price(F, K, T, SIGMA, D)) < 3 * se


def test_geometric_asian_matches_closed_form_within_3_se():
    paths = simulate(flat, T_GRID, F_T, n_pairs=50_000, seed=2)
    K = 3300.0
    mean, se = estimate(D * np.maximum(paths["G"] - K, 0.0))
    assert abs(mean - geometric_asian_call(F_T, T_GRID, SIGMA, K, D)) < 3 * se


def test_control_variate_shrinks_the_error():
    paths = simulate(flat, T_GRID, F_T, n_pairs=20_000, seed=3)
    K = 3300.0
    y = D * np.maximum(paths["A"] - K, 0.0)
    x = D * np.maximum(paths["G"] - K, 0.0)
    _, se_plain = estimate(y)
    _, se_cv = estimate(y, x, geometric_asian_call(F_T, T_GRID, SIGMA, K, D))
    assert se_cv < 0.2 * se_plain
