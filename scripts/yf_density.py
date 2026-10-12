"""Risk-neutral density of SPX from today's Yahoo Finance option chain (fallback while WRDS is blocked).

Usage: python scripts/yf_density.py [EXPIRY]          e.g. 2026-12-18
Caches the raw chain in data/yf/ and writes figures/yf_density_<expiry>.png.
"""

import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spxvol.black import implied_vol_vec
from spxvol.density import density_stats, lognormal_density, rn_density, strike_grid
from spxvol.svi import fit_svi, svi_g, svi_w

ROOT = Path(__file__).resolve().parents[1]
MAX_REL_SPREAD = 0.5   # drop quotes with (ask - bid) / mid above this
ATM_BAND = 0.04        # strikes within +-4% of spot used for put-call parity


def fetch_chain(expiry: str, today: str) -> tuple[pd.DataFrame, float]:
    cache = ROOT / "data" / "yf" / f"spx_{today}_{expiry}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        return df, df.attrs.get("spot", float(df["spot"].iloc[0]))
    import yfinance as yf
    t = yf.Ticker("^SPX")
    spot = float(t.history(period="5d")["Close"].iloc[-1])
    ch = t.option_chain(expiry)
    df = pd.concat([ch.calls.assign(cp="C"), ch.puts.assign(cp="P")])
    df = df[["cp", "strike", "bid", "ask", "impliedVolatility"]].assign(spot=spot)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df, spot


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df.bid > 0) & (df.ask > df.bid)].copy()
    df["mid"] = (df.bid + df.ask) / 2
    return df[(df.ask - df.bid) / df.mid <= MAX_REL_SPREAD]


def forward_from_parity(df: pd.DataFrame, spot: float) -> tuple[float, float]:
    """C - P = D (F - K): regress C - P on K near the money; slope = -D, intercept = D F."""
    piv = df.pivot(index="strike", columns="cp", values="mid").dropna()
    piv = piv[abs(piv.index / spot - 1) < ATM_BAND]
    slope, intercept = np.polyfit(piv.index, piv.C - piv.P, 1)
    D = -slope
    return intercept / D, D


def main(expiry: str):
    today = date.today().isoformat()
    raw, spot = fetch_chain(expiry, today)
    T = (date.fromisoformat(expiry) - date.today()).days / 365
    q = clean(raw)
    F, D = forward_from_parity(q, spot)
    r = -np.log(D) / T

    otm = q[((q.cp == "P") & (q.strike < F)) | ((q.cp == "C") & (q.strike > F))].copy()
    otm["iv"] = implied_vol_vec(otm.mid, F, otm.strike, T, D, otm.cp == "C")
    otm = otm.dropna(subset=["iv"]).sort_values("strike")
    otm["k"] = np.log(otm.strike / F)

    p = fit_svi(otm.k, otm.iv**2 * T)
    fit_iv = np.sqrt(svi_w(otm.k, p) / T)
    rmse_vp = 100 * np.sqrt(np.mean((fit_iv - otm.iv) ** 2))
    kk = np.linspace(otm.k.min(), otm.k.max(), 400)
    g_min = svi_g(kk, p).min()

    K = strike_grid(F, 0.4, 1.6)
    dens = rn_density(p, F, T, D, K)
    atm_vol = np.sqrt(svi_w(0.0, p) / T)
    logn = lognormal_density(F, T, atm_vol, K)
    stats = pd.DataFrame({"risk-neutral": density_stats(K, dens, F),
                          "lognormal": density_stats(K, logn, F)})

    print(f"SPX {today} -> {expiry}   T = {T:.3f}y   spot = {spot:.1f}   F = {F:.1f}   "
          f"D = {D:.4f} (r = {100*r:.2f}%)")
    print(f"quotes: {len(raw)} raw -> {len(q)} after bid/ask/spread filter -> {len(otm)} OTM with IV")
    print(f"SVI (a, b, rho, m, sigma) = {np.round(p, 4)}   RMSE = {rmse_vp:.2f} vol pts   "
          f"min g(k) on quote range = {g_min:.3f} ({'no' if g_min >= 0 else 'BUTTERFLY'} arbitrage)")
    print(f"ATM vol = {100*atm_vol:.2f}%")
    print(stats.round(4).to_string())

    blue, orange, ink, muted = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="#fcfcfb")
    for ax in (ax1, ax2):
        ax.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#e6e5e1", lw=0.8)
        ax.tick_params(colors=muted)

    ax1.scatter(otm.strike, 100 * otm.iv, s=14, color=ink, alpha=0.6, label="market mid IV (OTM)")
    Kf = F * np.exp(kk)
    ax1.plot(Kf, 100 * np.sqrt(svi_w(kk, p) / T), color=blue, lw=2, label="SVI fit")
    ax1.axvline(F, color=muted, lw=1, ls=":")
    ax1.set_xlabel("strike"); ax1.set_ylabel("implied vol (%)")
    ax1.set_title(f"SPX {expiry} smile  (RMSE {rmse_vp:.2f} vol pts)", loc="left", color=ink)
    ax1.legend(frameon=False)

    ax2.plot(K, dens * F, color=blue, lw=2, label="risk-neutral (SVI + Breeden-Litzenberger)")
    ax2.plot(K, logn * F, color=orange, lw=2, label=f"lognormal at ATM vol {100*atm_vol:.1f}%")
    left = K < 0.8 * F
    ax2.fill_between(K[left], 0, dens[left] * F, color=blue, alpha=0.18, lw=0)
    ax2.axvline(F, color=muted, lw=1, ls=":")
    ax2.set_xlim(0.6 * F, 1.3 * F)
    ax2.set_xlabel("index level at expiry"); ax2.set_ylabel("density (per unit of F)")
    ax2.set_title(f"P(fall > 20%): RN {100*stats.loc['P(S<0.8F)','risk-neutral']:.2f}%  "
                  f"vs lognormal {100*stats.loc['P(S<0.8F)','lognormal']:.2f}%", loc="left", color=ink)
    ax2.legend(frameon=False)
    fig.tight_layout()
    out = ROOT / "figures" / f"yf_density_{expiry}.png"
    fig.savefig(out, dpi=150)
    print("saved", out.relative_to(ROOT))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2026-12-18")
