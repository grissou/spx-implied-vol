"""The probability distribution of a stock's future price implied by its option prices (OptionMetrics via WRDS).

Breeden-Litzenberger: the probability density of the price at expiry is the second derivative of
the call price in strike, q(K) = (1/D) d2C/dK2. Quotes are too coarse and noisy to differentiate
twice, so a smooth smile (SVI) is fitted to each expiry first and the fitted call prices are
differentiated instead.

    python implied_distribution.py SPX 2020-02-19

For the expiries nearest 1, 3 and 6 months it draws the implied distribution, marks where the
price actually closed on each expiry date, and prints the probability the market had priced for
ending there or beyond. Option quotes come from WRDS on first use (credentials in ~/.pgpass), then
from data/; the realised closes come from Yahoo. Requires numpy, pandas, scipy, matplotlib,
pyarrow, wrds, yfinance.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

DATA = Path(__file__).parent / "data"
SECIDS = {"SPX": 108105}  # OptionMetrics ids; other tickers are looked up
YAHOO = {"SPX": "^GSPC"}  # Yahoo symbols where they differ


# ---------------------------------------------------------------- data
def load(ticker, date):
    """One day of option quotes, the close and the zero curve."""
    cache = DATA / f"{ticker}_{date}"
    if not (cache / "options.parquet").exists():
        import wrds

        db = wrds.Connection()
        secid = SECIDS.get(ticker) or int(db.raw_sql(
            f"select secid from optionm.opprcd{date[:4]} where date = '{date}' and secid in "
            f"(select secid from optionm.securd where ticker = '{ticker}') group by secid order by count(*) desc limit 1").secid[0])
        sql = {
            "options": f"select date, exdate, cp_flag, strike_price, best_bid, best_offer, am_settlement "
                       f"from optionm.opprcd{date[:4]} where secid = {secid} and date = '{date}'",
            "spot": f"select date, close from optionm.secprd where secid = {secid} and date = '{date}'",
            "zero_curve": f"select date, days, rate from optionm.zerocd where date = '{date}' order by days",
        }
        cache.mkdir(parents=True, exist_ok=True)
        for name, q in sql.items():
            df = db.raw_sql(q, date_cols=["date"])
            if name == "options":
                df["strike"] = df.pop("strike_price") / 1000  # OptionMetrics stores strikes x1000
            df.to_parquet(cache / f"{name}.parquet", index=False)
    o, spot, zc = (pd.read_parquet(cache / f"{n}.parquet") for n in ("options", "spot", "zero_curve"))
    o["exdate"] = pd.to_datetime(o.exdate)
    return o, float(spot.close.iloc[0]), zc


def clean(o):
    """Live quotes only, 7 days to 3 years out, PM-settled contract where AM and PM share a date."""
    o = o.rename(columns={"best_bid": "bid", "best_offer": "ask", "cp_flag": "cp"})
    o["mid"] = (o.bid + o.ask) / 2
    o["days"] = (o.exdate - o.date).dt.days
    o = o[(o.bid > 0) & (o.ask > o.bid) & ((o.ask - o.bid) / o.mid <= 0.5) & (o.days >= 7) & (o.days <= 1100)]
    has_pm = o[o.am_settlement == 0].exdate.unique()
    o = o[(o.am_settlement == 0) | ~o.exdate.isin(has_pm)]
    o["T"] = (o.days - 6.5 / 24 * o.am_settlement) / 365  # AM contracts expire at the open
    return o


def realised_close(ticker, exdates):
    """Actual close on (or just before) each expiry date, from Yahoo."""
    import yfinance as yf

    h = yf.Ticker(YAHOO.get(ticker, ticker)).history(start=min(exdates) - pd.Timedelta(days=7), end=max(exdates) + pd.Timedelta(days=3))["Close"]
    h.index = h.index.tz_localize(None).normalize()
    return {d: float(h[h.index <= d].iloc[-1]) for d in exdates}


# ---------------------------------------------------------------- pricing
def black(F, K, T, sigma, D, call=True):
    """Black-76: option price given the forward F and discount factor D."""
    sT = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sT**2) / sT
    d2 = d1 - sT
    if call:
        return D * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_vol(price, F, K, T, D, call):
    """Invert Black-76 by bisection (Brent). NaN if the price is outside its no-arbitrage bounds."""
    lo = D * max((F - K) if call else (K - F), 0)
    hi = D * (F if call else K)
    if not lo < price < hi:
        return np.nan
    return brentq(lambda s: black(F, K, T, s, D, call) - price, 1e-4, 5.0)


def forward(g, D):
    """Put-call parity C - P = D (F - K): average F over call-put pairs near the money."""
    piv = g.pivot(index="strike", columns="cp", values="mid").dropna()
    piv = piv[abs(piv.index / piv.index[len(piv) // 2] - 1) < 0.1]
    return float((piv.index + (piv.C - piv.P) / D).mean())


# ---------------------------------------------------------------- smile
def svi(k, p):
    """Total variance w = sigma^2 T as a function of log-moneyness k = ln(K/F)."""
    a, b, rho, m, s = p
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s**2))


def fit_svi(k, w, weight, sigma_min):
    """Weighted least squares. Bounds keep the fit on the right branch and the kink from collapsing."""
    lb = [0, 0, -0.999, k.min(), sigma_min]
    ub = [np.inf, np.inf, 0.999, k.max(), 2]
    best = None
    for rho0 in (-0.7, -0.3, 0.0):
        x0 = [0.5 * w.min(), 0.1, rho0, np.clip(0, k.min(), k.max()), max(0.1, sigma_min)]
        r = least_squares(lambda p: np.sqrt(weight) * (svi(k, p) - w), x0, bounds=(lb, ub))
        if best is None or r.cost < best.cost:
            best = r
    return best.x


# ---------------------------------------------------------------- distribution
def density(p, F, T, D):
    """Breeden-Litzenberger on a fine strike grid: second difference of the fitted call prices."""
    K = np.linspace(0.05 * F, 4 * F, 8001)
    sigma = np.sqrt(svi(np.log(K / F), p) / T)
    C = black(F, K, T, sigma, D)
    return K, np.gradient(np.gradient(C, K), K) / D


def main(ticker, date):
    o, spot, zc = load(ticker, date)
    q = clean(o)
    picks = [q["T"].iloc[(q["T"] * 365 - d).abs().argmin()] for d in (30, 91, 182)]
    actual = realised_close(ticker, [q[q["T"] == T].exdate.iloc[0] for T in picks])
    print(f"{ticker} on {date}: close {spot:.2f}")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, T in zip(axes, picks):
        g = q[q["T"] == T].copy()
        exdate = g.exdate.iloc[0]
        D = float(np.exp(-np.interp(T * 365, zc.days, zc.rate) / 100 * T))
        F = forward(g, D)
        g = g[((g.cp == "P") & (g.strike < F)) | ((g.cp == "C") & (g.strike > F))]  # out of the money
        g["iv"] = [implied_vol(r.mid, F, r.strike, T, D, r.cp == "C") for r in g.itertuples()]
        g = g.dropna(subset=["iv"])
        g["k"] = np.log(g.strike / F)
        g = g[(g.k > -0.6) & (g.k < 0.3)]
        w = g.iv**2 * T
        vega = D * F * norm.pdf((-g.k + 0.5 * w) / np.sqrt(w)) * np.sqrt(T)
        w_sd = g.iv * T * (g.ask - g.bid) / vega  # uncertainty in w from the bid-ask spread
        p = fit_svi(g.k.values, w.values, 1 / w_sd.clip(lower=1e-6).values ** 2, 0.25 * np.sqrt(np.interp(0, g.k, w)))
        K, dens = density(p, F, T, D)
        atm = np.sqrt(svi(0, p) / T)
        d2 = (np.log(F / K) - 0.5 * atm**2 * T) / (atm * np.sqrt(T))
        logn = norm.pdf(d2) / (K * atm * np.sqrt(T))

        S_T = actual[exdate]
        fell = S_T < spot
        side = K <= S_T if fell else K >= S_T
        prob, prob_ln = np.trapezoid(dens[side], K[side]), np.trapezoid(logn[side], K[side])
        word = "below" if fell else "above"
        print(f"  expiry {exdate.date()} ({T:.2f}y): ATM vol {100 * atm:.0f}%, fit RMSE "
              f"{100 * np.sqrt(np.mean((np.sqrt(svi(g.k, p) / T) - g.iv) ** 2)):.2f} vol pts, mass {np.trapezoid(dens, K):.3f}; "
              f"actual close {S_T:.0f} ({100 * (S_T / spot - 1):+.0f}%); market-implied P({word}) = {100 * prob:.2f}%, lognormal {100 * prob_ln:.2f}%")

        ax.plot(K / spot, dens * spot, color="#2a78d6", lw=2, label="implied by option prices")
        ax.plot(K / spot, logn * spot, color="#eb6834", lw=1.5, ls="--", label=f"lognormal at ATM vol {100 * atm:.0f}%")
        ax.axvline(S_T / spot, color="#0b0b0b", lw=1.2, ls=":")
        ax.text(0.02, 0.97, f"dotted line: actual close {S_T:.0f} ({100 * (S_T / spot - 1):+.0f}%)\nmarket-implied P({word}) = {100 * prob:.2f}%",
                transform=ax.transAxes, va="top", fontsize=9, color="#0b0b0b")
        ax.set(xlim=(0.5, 1.4), xlabel=f"price at expiry / close on {date}", title=f"expiry {exdate.date()}, {T:.2f} years ahead")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper right", fontsize=9)
    axes[0].set_ylabel("probability density")
    fig.suptitle(f"{ticker} on {date} (close {spot:.0f}): distribution of the price at expiry implied by option prices, and what happened",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(f"implied_distribution_{ticker}_{date}.png", dpi=150)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
