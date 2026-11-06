"""python scripts/build.py  -> rebuilds results/ (main outputs) and figures/ (diagnostics) from data/"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

from spxvol.clean import clean_quotes, forwards, otm_quotes
from spxvol.data import load_day
from spxvol.density import density_stats, lognormal_density, rn_density, strike_grid
from spxvol.surface import K_CORE, PARAMS, calendar_violations, fit_slices
from spxvol.svi import svi_w

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"      # diagnostics, named <step>_<content>
RESULTS = ROOT / "results"  # main outputs: density figure and density stats
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


def save(fig, name, folder=FIG, tight=True):
    folder.mkdir(exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(folder / name, dpi=150)
    plt.close(fig)
    print("saved", (folder / name).relative_to(ROOT))


def step1():
    """Clean quotes, forwards, own implied vols vs OptionMetrics."""
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
        ax1.plot(f["T"], 100 * (f.F / r["spot"] - 1), "o-", color=c, lw=2, ms=4, label=f"{label}: parity")
        ax1.plot(f["T"], 100 * (f.F_div / r["spot"] - 1), "--", color=c, lw=1.5, label=f"{label}: S exp((r-q)T)")
        ax2.plot(f["T"], -100 * np.log(f.D_reg) / f["T"], "o-", color=c, lw=2, ms=4, label=f"{label}: regression")
        ax2.plot(f["T"], -100 * np.log(f.D) / f["T"], "--", color=c, lw=1.5, label=f"{label}: zero curve")
    ax1.set(xlabel="T (years)", ylabel="F / S - 1  (%)", title="Forward: parity vs dividend-yield benchmark")
    ax2.set(xlabel="T (years)", ylabel="rate (%)", ylim=(-2, 4), title="Discount rate: regression vs zero curve")
    ax1.legend(frameon=False, fontsize=8)
    ax2.legend(frameon=False, fontsize=8)
    save(fig, "1_forwards_vs_zero_curve.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    style(fig, axes)
    for ax, (label, r) in zip(axes, res.items()):
        o = r["otm"]
        diff = 100 * (o.iv - o.impl_volatility)
        ax.scatter(o.k, diff, s=6, color=COLOR[label], alpha=0.5)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set(xlim=(-1, 0.5), xlabel="k = ln(K/F)", title=f"{label}: median {diff.median():+.2f} vol pts")
    axes[0].set_ylabel("own IV minus OptionMetrics IV (vol points)")
    fig.suptitle("Own implied vol minus OptionMetrics' for the same quote: opposite signs on puts and calls mean a different forward",
                 color=INK, x=0.01, ha="left", fontsize=11)
    save(fig, "1_implied_vol_vs_optionmetrics.png")
    return res


def step2(res):
    """SVI per expiry with butterfly and calendar checks."""
    for label, r in res.items():
        r["slices"] = s = fit_slices(r["otm"])
        cal = calendar_violations(s)
        print(f"{label}: {len(s)} slices, RMSE {s.rmse_vp.mean():.2f} vol pts (max {s.rmse_vp.max():.2f}), "
              f"butterfly violations {int((s.g_min < 0).sum())}, calendar violations {len(cal)} "
              f"(worst dw {cal.dw.min() if len(cal) else 0:.4f})")

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharex=True)
    style(fig, axes)
    for row_axes, (label, r) in zip(axes, res.items()):
        s, otm = r["slices"], r["otm"]
        for ax, days in zip(row_axes, (30, 91, 182, 365)):
            row = s.iloc[(s["T"] * 365 - days).abs().argmin()]
            g = otm[otm.exdate == row.exdate]
            kk = np.linspace(max(g.k.min(), -1.0), g.k.max(), 300)
            p = row[PARAMS].values.astype(float)
            ax.axvspan(*K_CORE, color="#e6e5e1", alpha=0.5, lw=0)
            ax.scatter(g.k, 100 * g.iv, s=6, color=INK, alpha=0.5)
            ax.plot(kk, 100 * np.sqrt(svi_w(kk, p) / row["T"]), color=COLOR[label], lw=2)
            ax.set(xlim=(-1.0, 0.4), title=f"{label.split()[0]}, {row.exdate.date()}, T={row['T']:.2f}, RMSE {row.rmse_vp:.2f}")
            ax.title.set_fontsize(10)
        row_axes[0].set_ylabel("implied vol (%)")
    for ax in axes[1]:
        ax.set_xlabel("k = ln(K/F)")
    fig.suptitle("Market implied vol (dots) and SVI fit (line); shaded = fitted strike range", color=INK, x=0.01, ha="left")
    save(fig, "2_svi_smiles.png")

    fig = plt.figure(figsize=(12, 5), facecolor=SURFACE)
    kgrid = np.linspace(*K_CORE, 181)
    for i, (label, r) in enumerate(res.items()):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        s = r["slices"][r["slices"]["T"] <= 1.0]
        K, T = np.meshgrid(kgrid, s["T"].values)
        W = np.array([svi_w(kgrid, p) for p in s[PARAMS].values])
        ax.plot_surface(K, T, 100 * np.sqrt(W / s["T"].values[:, None]), cmap="Blues" if "calm" in label else "Oranges",
                        rstride=1, cstride=2, lw=0, antialiased=True)
        ax.set(xlabel="k = ln(K/F)", ylabel="T (years)", zlabel="implied vol (%)", title=f"{label}, expiries to 1 year")
        ax.set_facecolor(SURFACE)
        ax.view_init(elev=22, azim=-55)
    fig.subplots_adjust(left=0, right=1, bottom=0.02, top=0.95, wspace=0)
    save(fig, "2_svi_surface.png", tight=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    style(fig, (ax1, ax2))
    for label, r in res.items():
        s = r["slices"]
        ax1.plot(s["T"], s.rmse_vp, "o-", color=COLOR[label], lw=2, ms=4, label=label)
        ax2.plot(s["T"], s.g_min, "o-", color=COLOR[label], lw=2, ms=4, label=label)
    ax1.set(xlabel="T (years)", ylabel="RMSE (vol points)", title="SVI fit error per expiry")
    ax2.axhline(0, color=MUTED, lw=1)
    ax2.set(xlabel="T (years)", ylabel="min g(k)", title="Butterfly check: min g(k) per expiry")
    ax1.legend(frameon=False)
    ax2.legend(frameon=False)
    save(fig, "2_svi_fit_quality.png")
    return res


def step3(res, targets=(30, 91)):
    """Risk-neutral density and cumulative probability vs lognormal at the expiries nearest 1 and 3 months."""
    stats = []
    fig, axes = plt.subplots(2, len(targets), figsize=(12, 8), sharex="col")
    style(fig, axes)
    for col, days in enumerate(targets):
        ax, axc = axes[:, col]
        for label, r in res.items():
            s = r["slices"]
            row = s.iloc[(s["T"] * 365 - days).abs().argmin()]
            p, F, T, D = row[PARAMS].values.astype(float), row.F, row["T"], row.D
            K = strike_grid(F)
            q = rn_density(p, F, T, D, K)
            atm = np.sqrt(svi_w(0.0, p) / T)
            ln = lognormal_density(F, T, atm, K)
            for name, dens in (("risk-neutral", q), ("lognormal", ln)):
                stats.append(dict(day=label, expiry=row.exdate.date(), T=round(T, 3), model=name, atm_vol=atm,
                                  **density_stats(K, dens, F)))
            ax.plot(K / F, q * F, color=COLOR[label], lw=2, label=f"{label} ({row.exdate.date()})")
            ax.plot(K / F, ln * F, color=COLOR[label], lw=1.2, ls="--", label=f"lognormal, ATM vol {100 * atm:.0f}%")
            cdf = cumulative_trapezoid(q, K, initial=0)
            axc.plot(K / F, cdf, color=COLOR[label], lw=2)
            axc.plot(K / F, cumulative_trapezoid(ln, K, initial=0), color=COLOR[label], lw=1.2, ls="--")
            for x in (0.8, 0.9):
                pr = np.interp(x, K / F, cdf)
                axc.plot([x], [pr], "o", color=COLOR[label], ms=5)
                axc.annotate(f"{100 * pr:.1f}%", (x, pr), xytext=(-6, 6), textcoords="offset points", ha="right",
                             fontsize=8, color=COLOR[label], bbox=dict(fc=SURFACE, ec="none", pad=1))
            axc.set_title(f"cumulative probability, total {cdf[-1]:.3f}", fontsize=10, loc="left")
        ax.set(xlim=(0.5, 1.3), title=f"about {days} days to expiry")
        ax.legend(frameon=False, fontsize=8)
        for x in (0.8, 0.9):
            axc.axvline(x, color=MUTED, lw=0.8, ls=":")
        axc.axhline(1, color=MUTED, lw=0.8)
        axc.set(xlabel="index at expiry / forward", ylim=(0, 1.05))
    axes[0, 0].set_ylabel("probability density")
    axes[1, 0].set_ylabel("P(index below this level)")
    save(fig, "risk_neutral_density.png", RESULTS)
    st = pd.DataFrame(stats)
    st.round(4).to_csv(RESULTS / "density_stats.csv", index=False)
    print(st.round(3).to_string(index=False))
    return st


if __name__ == "__main__":
    step3(step2(step1()))
