"""Rebuild every figure from the cached WRDS data:  python scripts/build.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spxvol.black import implied_vol_vec
from spxvol.clean import clean_quotes, forwards, otm_quotes
from spxvol.data import load_day

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
DATES = {"calm 2020-02-19": "2020-02-19", "stress 2020-03-16": "2020-03-16"}
COLOR = {"calm 2020-02-19": "#2a78d6", "stress 2020-03-16": "#eb6834"}
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def style(fig, axes):
    fig.set_facecolor(SURFACE)
    for ax in np.ravel(axes):
        ax.set_facecolor(SURFACE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e5e1", lw=0.8)
        ax.tick_params(colors=MUTED)
        ax.title.set_color(INK)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print("saved", name)


def step1():
    """Clean quotes, forwards and discount factors, own IV vs OptionMetrics. Returns per-date results."""
    res = {}
    for label, date in DATES.items():
        raw = load_day(date)
        spot, dy = raw["spot"].close.iloc[0], raw["div_yield"].rate.iloc[0]
        q = clean_quotes(raw["options"])
        f = forwards(q, spot, raw["zero_curve"], dy)
        otm = otm_quotes(q, f)
        res[label] = dict(spot=spot, q=q, f=f, otm=otm)
        print(f"{label}: {len(raw['options'])} raw -> {len(q)} clean -> {len(otm)} OTM with IV, {len(f)} expiries")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    style(fig, (ax1, ax2))
    for label, r in res.items():
        f, c = r["f"], COLOR[label]
        ax1.plot(f["T"], 100 * (f.F / r["spot"] - 1), "o-", color=c, lw=2, ms=4, label=f"{label}: put-call parity")
        ax1.plot(f["T"], 100 * (f.F_div / r["spot"] - 1), "--", color=c, lw=1.5, label=f"{label}: S exp((r-q)T), OptionMetrics q")
        ax2.plot(f["T"], -100 * np.log(f.D) / f["T"], "o-", color=c, lw=2, ms=4, label=f"{label}: parity-implied")
        ax2.plot(f["T"], -100 * np.log(f.D_zero) / f["T"], "--", color=c, lw=1.5, label=f"{label}: zero curve")
    ax1.set(xlabel="T (years)", ylabel="F / S - 1  (%)", title="Forward from put-call parity vs dividend-yield benchmark")
    ax2.set(xlabel="T (years)", ylabel="rate (%)", ylim=(-2, 4), title="Discount rate from parity vs zero curve")
    ax1.legend(frameon=False, fontsize=8)
    ax2.legend(frameon=False, fontsize=8)
    save(fig, "step1_forwards.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    style(fig, axes)
    for ax, (label, r) in zip(axes, res.items()):
        o = r["otm"]
        diff = 100 * (o.iv - o.impl_volatility)
        ax.scatter(o.k, diff, s=6, color=COLOR[label], alpha=0.5)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set(xlabel="log-moneyness k = ln(K/F)", title=f"{label}: own IV - OptionMetrics IV (median {diff.median():+.2f} vol pts)")
    axes[0].set_ylabel("vol points")
    save(fig, "step1_iv_vs_optionmetrics.png")
    return res


if __name__ == "__main__":
    step1()
