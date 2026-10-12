"""Step 1: filter quotes, forward and discount factor per expiry, own implied vols."""

import numpy as np
import pandas as pd

from .black import black_vega, implied_vol_vec

MAX_REL_SPREAD = 0.5
MIN_DAYS, MAX_DAYS = 7, 730
PARITY_BAND = 0.10          # strikes within +-10% of spot used for put-call parity
AM_SETTLE_OFFSET = 6.5 / 24  # AM-settled expiries end at the open


def clean_quotes(options: pd.DataFrame) -> pd.DataFrame:
    """bid > 0, ask > bid, relative spread <= 50%, 7 to 730 days, PM contract where AM and PM coexist."""
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
    """Per expiry: D from the zero curve, F from C - P = D (F - K) near the money.
    F_reg, D_reg (joint regression) and F_div = S exp((r - q) T) are kept as checks."""
    rows = []
    for exdate, g in q.groupby("exdate"):
        piv = g.pivot(index="strike", columns="cp", values="mid").dropna()
        piv = piv[abs(piv.index / spot - 1) <= PARITY_BAND]
        if len(piv) < 3:
            continue
        T = g["T"].iloc[0]
        D = float(np.exp(-zero_rate(zero_curve, T) * T))
        slope, intercept = np.polyfit(piv.index, piv.C - piv.P, 1)
        rows.append(dict(exdate=exdate, T=T, n_pairs=len(piv), D=D, F=(piv.index + (piv.C - piv.P) / D).mean(),
                         F_reg=-intercept / slope, D_reg=-slope))
    f = pd.DataFrame(rows)
    f["F_div"] = spot * np.exp((zero_rate(zero_curve, f["T"]) - div_yield / 100) * f["T"])
    return f


def otm_quotes(q: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    """OTM quotes with own implied vol, k, total variance w and its bid-ask uncertainty w_sd."""
    q = q.merge(f[["exdate", "F", "D"]], on="exdate")
    q = q[((q.cp == "P") & (q.strike < q.F)) | ((q.cp == "C") & (q.strike > q.F))].copy()
    q["iv"] = implied_vol_vec(q.mid, q.F, q.strike, q["T"], q.D, q.cp == "C")
    q = q.dropna(subset=["iv"])
    q["k"] = np.log(q.strike / q.F)
    q["w"] = q.iv**2 * q["T"]
    vega = black_vega(q.F, q.strike, q["T"], q.iv, q.D)
    q["w_sd"] = q.iv * q["T"] * (q.ask - q.bid) / vega
    return q.reset_index(drop=True)
