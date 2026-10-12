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

BLUE, ORANGE, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"


def fetch_chain(expiry: str, today: str) -> pd.DataFrame:
    """Calls and puts for one expiry with the last index close in column `spot`."""
    cache = ROOT / "data" / "yf" / f"spx_{today}_{expiry}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    import yfinance as yf
    t = yf.Ticker("^SPX")
    spot = float(t.history(period="5d")["Close"].iloc[-1])
    ch = t.option_chain(expiry)
    df = pd.concat([ch.calls.assign(cp="C"), ch.puts.assign(cp="P")])
    df = df[["cp", "strike", "bid", "ask"]].assign(spot=spot)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df.bid > 0) & (df.ask > df.bid)].copy()
    df["mid"] = (df.bid + df.ask) / 2
    return df[(df.ask - df.bid) / df.mid <= MAX_REL_SPREAD]


def forward_from_parity(df: pd.DataFrame, spot: float) -> tuple[float, float]:
    """C - P = D (F - K): regress C - P on K near the money; slope = -D, intercept = D F."""
    piv = df.pivot(index="strike", columns="cp", values="mid").dropna()
    piv = piv[abs(piv.index / spot - 1) < ATM_BAND]
    slope, intercept = np.polyfit(piv.index, piv.C - piv.P, 1)
    return -intercept / slope, -slope


def build(expiry: str) -> dict:
    today = date.today()
    raw = fetch_chain(expiry, today.isoformat())
    T = (date.fromisoformat(expiry) - today).days / 365
    q = clean(raw)
    F, D = forward_from_parity(q, raw.spot.iloc[0])

    otm = q[((q.cp == "P") & (q.strike < F)) | ((q.cp == "C") & (q.strike > F))].copy()
    otm["iv"] = implied_vol_vec(otm.mid, F, otm.strike, T, D, otm.cp == "C")
    otm = otm.dropna(subset=["iv"]).sort_values("strike")
    otm["k"] = np.log(otm.strike / F)

    p = fit_svi(otm.k, otm.iv**2 * T)
    K = strike_grid(F, 0.4, 1.6)
    dens = rn_density(p, F, T, D, K)
    atm_vol = np.sqrt(svi_w(0.0, p) / T)
    logn = lognormal_density(F, T, atm_vol, K)
    return dict(
        expiry=expiry, today=today, T=T, spot=raw.spot.iloc[0], F=F, D=D,
        n_raw=len(raw), n_clean=len(q), otm=otm, p=p, K=K, dens=dens, atm_vol=atm_vol, logn=logn,
        rmse_vp=100 * np.sqrt(np.mean((np.sqrt(svi_w(otm.k, p) / T) - otm.iv) ** 2)),
        g_min=svi_g(np.linspace(otm.k.min(), otm.k.max(), 400), p).min(),
        stats=pd.DataFrame({"risk-neutral": density_stats(K, dens, F), "lognormal": density_stats(K, logn, F)}),
    )


def report(r: dict) -> None:
    print(f"SPX {r['today']} -> {r['expiry']}   T = {r['T']:.3f}y   spot = {r['spot']:.1f}   "
          f"F = {r['F']:.1f}   D = {r['D']:.4f} (r = {-100 * np.log(r['D']) / r['T']:.2f}%)")
    print(f"quotes: {r['n_raw']} raw -> {r['n_clean']} after bid/ask/spread filter -> {len(r['otm'])} OTM with IV")
    print(f"SVI (a, b, rho, m, sigma) = {np.round(r['p'], 4)}   RMSE = {r['rmse_vp']:.2f} vol pts   "
          f"min g(k) = {r['g_min']:.3f} ({'no' if r['g_min'] >= 0 else 'BUTTERFLY'} arbitrage)")
    print(f"ATM vol = {100 * r['atm_vol']:.2f}%")
    print(r["stats"].round(4).to_string())


def plot(r: dict, out: Path) -> None:
    otm, p, F, T, K = r["otm"], r["p"], r["F"], r["T"], r["K"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e5e1", lw=0.8)
        ax.tick_params(colors=MUTED)
        ax.axvline(F, color=MUTED, lw=1, ls=":")

    kk = np.linspace(otm.k.min(), otm.k.max(), 400)
    ax1.scatter(otm.strike, 100 * otm.iv, s=14, color=INK, alpha=0.6, label="market mid IV (OTM)")
    ax1.plot(F * np.exp(kk), 100 * np.sqrt(svi_w(kk, p) / T), color=BLUE, lw=2, label="SVI fit")
    ax1.set(xlabel="strike", ylabel="implied vol (%)")
    ax1.set_title(f"SPX {r['expiry']} smile  (RMSE {r['rmse_vp']:.2f} vol pts)", loc="left", color=INK)
    ax1.legend(frameon=False)

    tail = r["stats"].loc["P(S<0.8F)"]
    ax2.plot(K, r["dens"] * F, color=BLUE, lw=2, label="risk-neutral (SVI + Breeden-Litzenberger)")
    ax2.plot(K, r["logn"] * F, color=ORANGE, lw=2, label=f"lognormal at ATM vol {100 * r['atm_vol']:.1f}%")
    left = K < 0.8 * F
    ax2.fill_between(K[left], 0, r["dens"][left] * F, color=BLUE, alpha=0.18, lw=0)
    ax2.set(xlabel="index level at expiry", ylabel="density (per unit of F)", xlim=(0.6 * F, 1.3 * F))
    ax2.set_title(f"P(fall > 20%): RN {100 * tail['risk-neutral']:.2f}%  vs lognormal {100 * tail['lognormal']:.2f}%",
                  loc="left", color=INK)
    ax2.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("saved", out.relative_to(ROOT))


if __name__ == "__main__":
    expiry = sys.argv[1] if len(sys.argv) > 1 else "2026-12-18"
    r = build(expiry)
    report(r)
    plot(r, ROOT / "figures" / f"yf_density_{expiry}.png")
