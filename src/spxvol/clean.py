"""Step 1: clean one day of SPX quotes, back out forwards and discount factors, recompute implied vols.

Filters (agreed): bid > 0, ask > bid, relative spread <= 50%, 7 <= days to expiry <= 730,
PM-settled contract kept where AM and PM share an expiry date, OTM options only.
"""

import numpy as np
import pandas as pd

from .black import implied_vol_vec

MAX_REL_SPREAD = 0.5
MIN_DAYS, MAX_DAYS = 7, 730
PARITY_BAND = 0.10          # strikes within +-10% of spot used for put-call parity
AM_SETTLE_OFFSET = 6.5 / 24  # AM expiries settle at the open, 6.5 hours before the close


def clean_quotes(options: pd.DataFrame) -> pd.DataFrame:
    o = options.rename(columns={"best_bid": "bid", "best_offer": "ask", "cp_flag": "cp"})
    o["mid"] = (o.bid + o.ask) / 2
    o["days"] = (o.exdate - o.date).dt.days
    o = o[(o.bid > 0) & (o.ask > o.bid) & ((o.ask - o.bid) / o.mid <= MAX_REL_SPREAD)]
    o = o[(o.days >= MIN_DAYS) & (o.days <= MAX_DAYS)]
    has_pm = o[o.am_settlement == 0].exdate.unique()
    o = o[(o.am_settlement == 0) | ~o.exdate.isin(has_pm)]
    o["T"] = (o.days - AM_SETTLE_OFFSET * o.am_settlement) / 365
    cols = ["exdate", "T", "cp", "strike", "bid", "ask", "mid", "impl_volatility", "am_settlement"]
    return o[cols].sort_values(["exdate", "strike"]).reset_index(drop=True)


def zero_rate(zero_curve: pd.DataFrame, T) -> np.ndarray:
    """Continuously compounded rate (decimal) interpolated linearly in days."""
    return np.interp(np.asarray(T) * 365, zero_curve.days, zero_curve.rate) / 100


def forwards(q: pd.DataFrame, spot: float, zero_curve: pd.DataFrame, div_yield: float) -> pd.DataFrame:
    """Per expiry: F and D from C - P = D (F - K) by regression, plus the zero-curve / dividend benchmarks."""
    rows = []
    for exdate, g in q.groupby("exdate"):
        piv = g.pivot(index="strike", columns="cp", values="mid").dropna()
        piv = piv[abs(piv.index / spot - 1) <= PARITY_BAND]
        if len(piv) < 3:
            continue
        slope, intercept = np.polyfit(piv.index, piv.C - piv.P, 1)
        rows.append(dict(exdate=exdate, T=g["T"].iloc[0], n_pairs=len(piv), F=-intercept / slope, D=-slope))
    f = pd.DataFrame(rows)
    r = zero_rate(zero_curve, f["T"])
    f["D_zero"] = np.exp(-r * f["T"])
    f["F_div"] = spot * np.exp((r - div_yield / 100) * f["T"])
    return f


def otm_quotes(q: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    """OTM quotes (puts below F, calls above) with own Black-76 implied vol, log-moneyness and total variance."""
    q = q.merge(f[["exdate", "F", "D"]], on="exdate")
    q = q[((q.cp == "P") & (q.strike < q.F)) | ((q.cp == "C") & (q.strike > q.F))].copy()
    q["iv"] = implied_vol_vec(q.mid, q.F, q.strike, q["T"], q.D, q.cp == "C")
    q = q.dropna(subset=["iv"])
    q["k"] = np.log(q.strike / q.F)
    q["w"] = q.iv**2 * q["T"]
    return q.reset_index(drop=True)
