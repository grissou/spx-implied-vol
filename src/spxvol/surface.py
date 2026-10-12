"""Step 2: SVI per expiry on the core strike range, arbitrage checks, total-variance surface."""

import numpy as np
import pandas as pd

from .svi import fit_svi, svi_g, svi_w

K_CORE = (-0.6, 0.3)
PARAMS = ["a", "b", "rho", "m", "sigma"]


def fit_slices(otm: pd.DataFrame) -> pd.DataFrame:
    """Precision-weighted SVI per expiry, with RMSE in vol points and min g(k)."""
    rows = []
    for exdate, g in otm.groupby("exdate"):
        c = g[(g.k >= K_CORE[0]) & (g.k <= K_CORE[1])]
        T = c["T"].iloc[0]
        p = fit_svi(c.k, c.w, 1 / c.w_sd.clip(lower=1e-6) ** 2)
        err_vp = 100 * (np.sqrt(svi_w(c.k, p) / T) - c.iv)
        rows.append(dict(exdate=exdate, T=T, F=c.F.iloc[0], D=c.D.iloc[0], n=len(c), k_min=c.k.min(), k_max=c.k.max(),
                         **dict(zip(PARAMS, p)), rmse_vp=np.sqrt(np.mean(err_vp**2)),
                         g_min=svi_g(np.linspace(c.k.min(), c.k.max(), 300), p).min()))
    return pd.DataFrame(rows)


def calendar_violations(slices: pd.DataFrame, k=np.linspace(*K_CORE, 181)) -> pd.DataFrame:
    """Adjacent expiries where w decreases in T on the core range."""
    P = slices[PARAMS].values
    rows = []
    for i in range(len(slices) - 1):
        dw = svi_w(k, P[i + 1]) - svi_w(k, P[i])
        if dw.min() < 0:
            rows.append(dict(T1=slices["T"].iloc[i], T2=slices["T"].iloc[i + 1], k=k[dw.argmin()], dw=dw.min()))
    return pd.DataFrame(rows, columns=["T1", "T2", "k", "dw"])


class Surface:
    """w(k, T): linear in T between expiries, running max in T, proportional to T before the first."""

    def __init__(self, slices: pd.DataFrame, k=np.linspace(*K_CORE, 181)):
        self.k = k
        self.T = slices["T"].values
        W = np.array([svi_w(k, p) for p in slices[PARAMS].values])
        self.W = np.maximum.accumulate(W, axis=0)
        self.adjusted = float(np.abs(self.W - W).max())

    def w(self, k, T):
        k, T = np.broadcast_arrays(np.asarray(k, float), np.asarray(T, float))
        out = np.empty_like(k)
        for idx, (kk, tt) in enumerate(zip(k.ravel(), T.ravel())):
            col = np.array([np.interp(kk, self.k, row) for row in self.W])
            if tt <= self.T[0]:
                out.flat[idx] = col[0] * tt / self.T[0]
            else:
                out.flat[idx] = np.interp(tt, self.T, col)
        return out

    def iv(self, k, T):
        return np.sqrt(self.w(k, T) / T)
