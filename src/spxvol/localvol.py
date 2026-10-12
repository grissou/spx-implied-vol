"""Step 4: Dupire local vol in total variance, sigma_loc^2 = (dw/dT) / g(k, T), on the surface k grid."""

import numpy as np


class LocalVol:
    """sigma_loc(k, t) for each t in t_grid, clipped to [sig_min, sig_max]; flat beyond the k grid."""

    def __init__(self, surface, t_grid, sig_min=0.05, sig_max=3.0, dT=1e-4):
        self.k, self.t = surface.k, np.asarray(t_grid, float)
        self.sig_min, self.sig_max = sig_min, sig_max
        k = self.k
        grid = []
        for t in self.t:
            w = surface.w(k, t)
            w_T = (surface.w(k, t + dT) - w) / dT
            w1 = np.gradient(w, k)
            w2 = np.gradient(w1, k)
            g = (1 - k * w1 / (2 * w)) ** 2 - w1**2 / 4 * (1 / w + 0.25) + w2 / 2
            ok = g > 1e-8
            var = np.interp(k, k[ok], w_T[ok] / g[ok]) if ok.any() else np.full_like(k, sig_max**2)
            grid.append(np.sqrt(np.clip(var, sig_min**2, sig_max**2)))
        self.sigma_grid = np.array(grid)

    def __call__(self, k, j):
        return np.interp(k, self.k, self.sigma_grid[j])
