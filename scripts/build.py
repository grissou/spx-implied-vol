"""python scripts/build.py  -> rebuilds results/ (main outputs) and figures/ (diagnostics) from data/"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spxvol.black import black_price, implied_vol_vec
from spxvol.clean import clean_quotes, forwards, otm_quotes
from spxvol.data import load_day
from spxvol.density import density_stats, lognormal_density, rn_density, strike_grid
from spxvol.localvol import LocalVol
from spxvol.mc import estimate, geometric_asian_call, simulate
from spxvol.surface import K_CORE, PARAMS, Surface, calendar_violations, fit_slices
from spxvol.svi import svi_w

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"      # diagnostics, named <step>_<content>
RESULTS = ROOT / "results"  # main outputs: density figure, density stats, exotic prices
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
    """SVI per expiry, arbitrage checks, surface grid."""
    for label, r in res.items():
        r["slices"] = s = fit_slices(r["otm"])
        r["surface"] = Surface(s)
        cal = calendar_violations(s)
        print(f"{label}: {len(s)} slices, RMSE {s.rmse_vp.mean():.2f} vol pts (max {s.rmse_vp.max():.2f}), "
              f"butterfly violations {int((s.g_min < 0).sum())}, calendar violations {len(cal)} "
              f"(worst dw {cal.dw.min() if len(cal) else 0:.4f}), surface adjustment {r['surface'].adjusted:.4f}")

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
    for i, (label, r) in enumerate(res.items()):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        sf = r["surface"]
        keep = sf.T <= 1.0
        K, T = np.meshgrid(sf.k, sf.T[keep])
        ax.plot_surface(K, T, 100 * np.sqrt(sf.W[keep] / sf.T[keep, None]), cmap="Blues" if "calm" in label else "Oranges",
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
    from scipy.integrate import cumulative_trapezoid

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


def step4(res, target_days=91, barrier=0.9, n_pairs=100_000, steps_per_day=8):
    """Dupire local vol, Monte Carlo validation on vanillas, then a down-and-out call and an arithmetic Asian.
    Barrier monitoring and Asian fixings are daily; the finer time steps keep the log-Euler bias below one s.e."""
    rows, conv = [], {}
    fig_lv, axes_lv = plt.subplots(1, 2, figsize=(12, 4.2))
    fig_val, axes_val = plt.subplots(1, 2, figsize=(12, 4.2))
    style(fig_lv, axes_lv)
    style(fig_val, axes_val)
    for (label, r), ax_lv, ax_val in zip(res.items(), axes_lv, axes_val):
        s, sf, S0 = r["slices"], r["surface"], r["spot"]
        row = s.iloc[(s["T"] * 365 - target_days).abs().argmin()]
        T, F, D = row["T"], row.F, row.D
        days = int(round(T * 365))
        n = days * steps_per_day
        t = np.linspace(0, T, n + 1)
        F_t = np.exp(np.interp(t, np.r_[0, s["T"]], np.r_[np.log(S0), np.log(s.F)]))
        lv = LocalVol(Surface(s, k=np.linspace(-1.5, 1.0, 501)), t[:-1] + T / n / 2)
        atm = sf.iv(0.0, T)
        sig_ts = np.sqrt(np.diff([sf.w(0.0, ti) for ti in t]) / np.diff(t))  # ATM term structure, no skew
        models = {"local vol": lv, "BS term structure": lambda k, j: np.full_like(k, sig_ts[j]),
                  "Black-Scholes": lambda k, j: np.full_like(k, atm)}
        sim = {name: simulate(fn, t, F_t, n_pairs, fix_every=steps_per_day) for name, fn in models.items()}
        clipped = np.mean((lv.sigma_grid <= lv.sig_min) | (lv.sigma_grid >= lv.sig_max))
        print(f"{label}: T={T:.3f} ({row.exdate.date()}), F={F:.1f}, ATM vol {100*atm:.1f}%, "
              f"{days} days x {steps_per_day} steps x {2*n_pairs} paths, local vol clipped on {100*clipped:.1f}% of the grid")

        kc = (lv.k >= -0.6) & (lv.k <= 0.3)
        im = ax_lv.pcolormesh(lv.k[kc], lv.t, 100 * lv.sigma_grid[:, kc], cmap="Blues" if "calm" in label else "Oranges", shading="auto")
        fig_lv.colorbar(im, ax=ax_lv, label="local vol (%)")
        ax_lv.set(xlabel="k = ln(S / F_t)", ylabel="t (years)", title=f"{label}: Dupire local vol")

        ks = np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2])
        K = F * np.exp(ks)
        mc = [estimate(D * np.maximum(sim["local vol"]["S_T"] - Ki, 0)) for Ki in K]
        px, se = np.array([m[0] for m in mc]), np.array([m[1] for m in mc])
        iv_mc, iv_lo, iv_hi = (implied_vol_vec(px + c * se, F, K, T, D, True) for c in (0, -2, 2))
        iv_svi = sf.iv(ks, T)
        ax_val.plot(ks, 100 * iv_svi, color=INK, lw=2, label="SVI surface")
        ax_val.errorbar(ks, 100 * iv_mc, yerr=100 * np.vstack([iv_mc - iv_lo, iv_hi - iv_mc]), fmt="o", color=COLOR[label],
                        ms=5, capsize=3, label="MC under local vol (2 s.e.)")
        ax_val.set(xlabel="k", ylabel="implied vol (%)", title=f"{label}: European calls, T={T:.2f}")
        ax_val.legend(frameon=False)
        print("  validation |MC - SVI| in s.e.:", np.round(np.abs(px - black_price(F, K, T, iv_svi, D)) / se, 1))

        B = barrier * S0
        eu_ref = black_price(F, F, T, atm, D)  # all three models price the ATM European identically
        for model, o in sim.items():
            eu = D * np.maximum(o["S_T"] - F, 0)
            dob = eu * (o["S_min"] > B)
            asian = D * np.maximum(o["A"] - F, 0)
            out = {"European": estimate(eu), "down-and-out call": estimate(dob, eu, eu_ref)}
            if model == "Black-Scholes":
                geo = D * np.maximum(o["G"] - F, 0)
                out["arithmetic Asian"] = estimate(asian, geo, geometric_asian_call(F_t[::steps_per_day], t[::steps_per_day], atm, F, D))
            else:
                out["arithmetic Asian"] = estimate(asian, eu, eu_ref)
            for name, (p, e) in out.items():
                rows.append(dict(day=label, model=model, option=name, price_pct_F=100 * p / F, se_pct_F=100 * e / F))
            if model == "local vol":
                conv[label] = dict(y=100 * dob / F, x=100 * eu / F, ref=100 * eu_ref / F)
    save(fig_lv, "4_local_vol_surface.png")
    save(fig_val, "4_monte_carlo_validation.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    style(fig, axes)
    for ax, (label, c) in zip(axes, conv.items()):
        m = np.unique(np.logspace(2, np.log10(n_pairs), 40).astype(int))
        pair = lambda v, mi: np.concatenate([v[:mi], v[n_pairs:n_pairs + mi]])
        runs = {"crude": [estimate(c["y"][:mi], antithetic=False) for mi in m],
                "antithetic": [estimate(pair(c["y"], mi)) for mi in m],
                "antithetic + control": [estimate(pair(c["y"], mi), pair(c["x"], mi), c["ref"]) for mi in m]}
        for (name, est), col in zip(runs.items(), (MUTED, INK, COLOR[label])):
            est = np.array(est)
            ax.fill_between(m, est[:, 0] - 2 * est[:, 1], est[:, 0] + 2 * est[:, 1], alpha=0.15, lw=0, color=col)
            ax.plot(m, est[:, 0], lw=1.5, color=col, label=f"{name} (s.e. {est[-1, 1]:.3f})")
        ax.set(xscale="log", xlabel="paths (pairs for antithetic)", ylabel="price (% of forward)", title=f"{label}: down-and-out call under local vol")
        ax.legend(frameon=False, fontsize=8)
    save(fig, "4_barrier_convergence.png")

    tab = pd.DataFrame(rows)
    tab.round(4).to_csv(RESULTS / "exotic_prices.csv", index=False)
    print(tab.pivot(index=["day", "option"], columns="model", values=["price_pct_F", "se_pct_F"]).round(3).to_string())
    return tab


if __name__ == "__main__":
    r = step1()
    step2(r)
    step3(r)
    step4(r)
