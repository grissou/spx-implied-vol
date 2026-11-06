"""python scripts/build.py [TICKER]  -> pulls today's chain (US market hours), rebuilds results/ and figures/"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

from spxvol.clean import clean_quotes, forwards, otm_quotes
from spxvol.data import load_chain
from spxvol.density import density_stats, lognormal_density, rn_density, strike_grid
from spxvol.surface import K_CORE, PARAMS, calendar_violations, fit_slices
from spxvol.svi import svi_w

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"      # diagnostics, named <step>_<content>
RESULTS = ROOT / "results"  # main outputs: density figure and density stats
BLUE, ORANGE, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"


def style(fig, axes):
    fig.set_facecolor(SURFACE)
    for ax in np.ravel(axes):
        ax.set_facecolor(SURFACE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e5e1", lw=0.8)
        ax.tick_params(colors=MUTED)
        ax.title.set_color(INK)


def save(fig, name, folder=FIG, tight=True):
    folder.mkdir(exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(folder / name, dpi=150)
    plt.close(fig)
    print("saved", (folder / name).relative_to(ROOT))


def step1(ticker):
    """Clean quotes, forward and discount factor per expiry, implied vols."""
    raw = load_chain(ticker)
    day = str(raw["spot"].date.iloc[0])[:10]
    spot, dy = raw["spot"].close.iloc[0], raw["div_yield"].rate.iloc[0]
    q = clean_quotes(raw["options"])
    f = forwards(q, spot, raw["zero_curve"], dy)
    otm = otm_quotes(q, f)
    print(f"{ticker} {day}: spot {spot:.2f}, rate {raw['zero_curve'].rate.iloc[0]:.2f}%, "
          f"{len(raw['options'])} raw -> {len(q)} clean -> {len(otm)} OTM with IV, {len(f)} expiries")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    style(fig, (ax1, ax2))
    ax1.plot(f["T"], 100 * (f.F / spot - 1), "o-", color=BLUE, lw=2, ms=4, label="put-call parity")
    ax1.plot(f["T"], 100 * (f.F_div / spot - 1), "--", color=MUTED, lw=1.5, label="S exp((r - q) T)")
    ax1.set(xlabel="T (years)", ylabel="F / S - 1  (%)", title="Forward from put-call parity vs rate benchmark")
    ax2.plot(f["T"], -100 * np.log(f.D_reg) / f["T"], "o-", color=BLUE, lw=2, ms=4, label="parity regression")
    ax2.plot(f["T"], -100 * np.log(f.D) / f["T"], "--", color=MUTED, lw=1.5, label="T-bill rate (used)")
    ax2.set(xlabel="T (years)", ylabel="rate (%)", ylim=(-5, 10), title="Discount rate implied by parity vs T-bill")
    ax1.legend(frameon=False)
    ax2.legend(frameon=False)
    save(fig, "1_forwards.png")
    return dict(ticker=ticker, day=day, spot=spot, q=q, f=f, otm=otm)


def step2(r):
    """SVI per expiry with butterfly and calendar checks."""
    r["slices"] = s = fit_slices(r["otm"])
    cal = calendar_violations(s)
    print(f"{len(s)} slices, RMSE {s.rmse_vp.mean():.2f} vol pts (max {s.rmse_vp.max():.2f}), "
          f"butterfly violations {int((s.g_min < 0).sum())}, calendar violations {len(cal)} "
          f"(worst dw {cal.dw.min() if len(cal) else 0:.4f})")
    otm = r["otm"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharex=True)
    style(fig, axes)
    for ax, days in zip(axes, (30, 91, 182, 365)):
        row = s.iloc[(s["T"] * 365 - days).abs().argmin()]
        g = otm[otm.exdate == row.exdate]
        kk = np.linspace(max(g.k.min(), -1.0), min(g.k.max(), 0.6), 300)
        p = row[PARAMS].values.astype(float)
        ax.axvspan(*K_CORE, color="#e6e5e1", alpha=0.5, lw=0)
        ax.scatter(g.k, 100 * g.iv, s=8, color=INK, alpha=0.6)
        ax.plot(kk, 100 * np.sqrt(svi_w(kk, p) / row["T"]), color=BLUE, lw=2)
        ax.set(xlim=(-1.0, 0.6), xlabel="k = ln(K/F)", title=f"{row.exdate.date()}, T={row['T']:.2f}, RMSE {row.rmse_vp:.2f}")
        ax.title.set_fontsize(10)
    axes[0].set_ylabel("implied vol (%)")
    fig.suptitle(f"{r['ticker']} {r['day']}: market implied vol (dots) and SVI fit (line); shaded = fitted strike range",
                 color=INK, x=0.01, ha="left")
    save(fig, "2_svi_smiles.png")

    fig = plt.figure(figsize=(7, 5.5), facecolor=SURFACE)
    ax = fig.add_subplot(projection="3d")
    kgrid = np.linspace(*K_CORE, 181)
    s1 = s[s["T"] <= 1.0]
    K, T = np.meshgrid(kgrid, s1["T"].values)
    W = np.array([svi_w(kgrid, p) for p in s1[PARAMS].values])
    ax.plot_surface(K, T, 100 * np.sqrt(W / s1["T"].values[:, None]), cmap="Blues", rstride=1, cstride=2, lw=0)
    ax.set(xlabel="k = ln(K/F)", ylabel="T (years)", zlabel="implied vol (%)", title=f"{r['ticker']} SVI surface, expiries to 1 year")
    ax.set_facecolor(SURFACE)
    ax.view_init(elev=22, azim=-55)
    fig.subplots_adjust(left=0, right=1, bottom=0.02, top=0.95)
    save(fig, "2_svi_surface.png", tight=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    style(fig, (ax1, ax2))
    ax1.plot(s["T"], s.rmse_vp, "o-", color=BLUE, lw=2, ms=4)
    ax1.set(xlabel="T (years)", ylabel="RMSE (vol points)", title="SVI fit error per expiry")
    ax2.plot(s["T"], s.g_min, "o-", color=BLUE, lw=2, ms=4)
    ax2.axhline(0, color=MUTED, lw=1)
    ax2.set(xlabel="T (years)", ylabel="min g(k)", title="Butterfly check: min g(k) per expiry (>= 0 is arbitrage-free)")
    save(fig, "2_svi_fit_quality.png")
    return r


def step3(r, targets=(30, 91, 182)):
    """Risk-neutral density and cumulative probability vs lognormal at the expiries nearest 1, 3 and 6 months."""
    s, stats = r["slices"], []
    fig, axes = plt.subplots(2, len(targets), figsize=(14, 7.5), sharex="col")
    style(fig, axes)
    for col, days in enumerate(targets):
        ax, axc = axes[:, col]
        row = s.iloc[(s["T"] * 365 - days).abs().argmin()]
        p, F, T, D = row[PARAMS].values.astype(float), row.F, row["T"], row.D
        K = strike_grid(F)
        q = rn_density(p, F, T, D, K)
        atm = np.sqrt(svi_w(0.0, p) / T)
        ln = lognormal_density(F, T, atm, K)
        for name, dens in (("risk-neutral", q), ("lognormal", ln)):
            stats.append(dict(expiry=row.exdate.date(), T=round(T, 3), model=name, atm_vol=atm, **density_stats(K, dens, F)))
        ax.plot(K / F, q * F, color=BLUE, lw=2, label="market-implied")
        ax.plot(K / F, ln * F, color=ORANGE, lw=1.5, ls="--", label=f"lognormal, ATM vol {100 * atm:.0f}%")
        ax.set(xlim=(0.4, 1.8), title=f"{row.exdate.date()}, about {days} days")
        ax.legend(frameon=False)
        cdf = cumulative_trapezoid(q, K, initial=0)
        axc.plot(K / F, cdf, color=BLUE, lw=2)
        axc.plot(K / F, cumulative_trapezoid(ln, K, initial=0), color=ORANGE, lw=1.5, ls="--")
        axc.axhline(1, color=MUTED, lw=0.8)
        axc.set(xlabel="price at expiry / forward", ylim=(0, 1.05), title=f"cumulative probability, total {cdf[-1]:.3f}")
        axc.title.set_fontsize(10)
    axes[0, 0].set_ylabel("probability density")
    axes[1, 0].set_ylabel("P(price below this level)")
    fig.suptitle(f"{r['ticker']} {r['day']}: what the options market implies for the share price at expiry",
                 color=INK, x=0.01, ha="left")
    save(fig, "risk_neutral_density.png", RESULTS)
    st = pd.DataFrame(stats)
    st.round(4).to_csv(RESULTS / "density_stats.csv", index=False)
    print(st.round(3).to_string(index=False))
    return st


if __name__ == "__main__":
    step3(step2(step1(sys.argv[1] if len(sys.argv) > 1 else "MU")))
