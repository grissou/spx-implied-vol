"""python scripts/build.py  -> rebuilds every figure from data/"""

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




def step2(res):
    from spxvol.surface import K_CORE, Surface, calendar_violations, fit_slices
    from spxvol.svi import svi_w

    for label, r in res.items():
        r["slices"] = s = fit_slices(r["otm"])
        r["surface"] = Surface(s)
        cal = calendar_violations(s)
        print(f"{label}: {len(s)} slices, RMSE {s.rmse_vp.mean():.2f} vol pts (max {s.rmse_vp.max():.2f}), "
              f"butterfly violations {int((s.g_min < 0).sum())}, calendar violations {len(cal)} "
              f"(worst dw {cal.dw.min() if len(cal) else 0:.4f}), surface adjustment {r['surface'].adjusted:.4f}")

    for label, r in res.items():
        s, otm = r["slices"], r["otm"]
        pick = s.iloc[np.linspace(0, len(s) - 1, 8).round().astype(int)]
        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        style(fig, axes)
        for ax, (_, row) in zip(axes.ravel(), pick.iterrows()):
            g = otm[otm.exdate == row.exdate]
            kk = np.linspace(g.k.min(), g.k.max(), 300)
            p = row[["a", "b", "rho", "m", "sigma"]].values.astype(float)
            ax.axvspan(*K_CORE, color="#e6e5e1", alpha=0.5, lw=0)
            ax.scatter(g.k, 100 * g.iv, s=6, color=INK, alpha=0.5)
            ax.plot(kk, 100 * np.sqrt(svi_w(kk, p) / row["T"]), color=COLOR[label], lw=2)
            ax.set_title(f"{row.exdate.date()}  T={row['T']:.2f}  RMSE {row.rmse_vp:.2f}", fontsize=10)
            ax.set_xlabel("k = ln(K/F)")
        axes[0, 0].set_ylabel("implied vol (%)"); axes[1, 0].set_ylabel("implied vol (%)")
        fig.suptitle(f"{label}: market IV vs SVI (fitted on shaded range)", color=INK, x=0.01, ha="left")
        save(fig, f"step2_smiles_{label.split()[1]}.png")

    fig = plt.figure(figsize=(13, 5.5), facecolor=SURFACE)
    for i, (label, r) in enumerate(res.items()):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        sf = r["surface"]
        K, T = np.meshgrid(sf.k, sf.T)
        ax.plot_surface(K, T, 100 * np.sqrt(sf.W / sf.T[:, None]), cmap="Blues" if "calm" in label else "Oranges",
                        rstride=1, cstride=4, lw=0, antialiased=True)
        ax.set(xlabel="k", ylabel="T (years)", zlabel="IV (%)", title=label)
        ax.set_facecolor(SURFACE); ax.view_init(elev=25, azim=-130)
    save(fig, "step2_surface.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    style(fig, (ax1, ax2))
    for label, r in res.items():
        s = r["slices"]
        ax1.plot(s["T"], s.rmse_vp, "o-", color=COLOR[label], lw=2, ms=4, label=label)
        ax2.plot(s["T"], s.g_min, "o-", color=COLOR[label], lw=2, ms=4, label=label)
    ax1.set(xlabel="T (years)", ylabel="RMSE (vol points)", title="SVI fit error per expiry")
    ax2.axhline(0, color=MUTED, lw=1)
    ax2.set(xlabel="T (years)", ylabel="min g(k)", title="Butterfly check: min g(k) per expiry")
    ax1.legend(frameon=False); ax2.legend(frameon=False)
    save(fig, "step2_fit_quality.png")
    return res


def step3(res, targets=(30, 91)):
    """Risk-neutral density vs lognormal at the expiries nearest 1 and 3 months."""
    from spxvol.density import density_stats, lognormal_density, rn_density, strike_grid
    from spxvol.surface import PARAMS
    from spxvol.svi import svi_w

    stats = []
    fig, axes = plt.subplots(1, len(targets), figsize=(12, 4.4), sharey=True)
    style(fig, axes)
    for ax, days in zip(axes, targets):
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
        ax.set(xlim=(0.5, 1.3), xlabel="K / F", title=f"about {days} days to expiry")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("density of S_T / F")
    save(fig, "step3_density.png")
    st = pd.DataFrame(stats)
    print(st.round(3).to_string(index=False))
    return st


if __name__ == "__main__":
    step3(step2(step1()))
