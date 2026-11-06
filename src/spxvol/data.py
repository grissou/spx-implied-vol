"""One snapshot of a stock's option chain from Yahoo Finance, cached as parquet. No filtering here."""

from datetime import date
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
NAMES = ["options", "spot", "zero_curve", "div_yield"]


def _fetch(ticker: str, today: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    t = yf.Ticker(ticker)
    spot = float(t.history(period="5d")["Close"].iloc[-1])
    rate = float(yf.Ticker("^IRX").history(period="5d")["Close"].iloc[-1])  # 13-week T-bill, percent
    chains = []
    for exp in t.options:
        ch = t.option_chain(exp)
        for df, cp in ((ch.calls, "C"), (ch.puts, "P")):
            chains.append(df[["strike", "bid", "ask", "impliedVolatility"]].assign(cp_flag=cp, exdate=exp))
    opt = pd.concat(chains, ignore_index=True)
    opt = opt.rename(columns={"bid": "best_bid", "ask": "best_offer", "impliedVolatility": "impl_volatility"})
    opt["date"] = pd.Timestamp(today)
    opt["exdate"] = pd.to_datetime(opt.exdate)
    opt["am_settlement"] = 0  # equity options settle on the close
    days = [7, 30, 91, 182, 365, 730]
    return {
        "options": opt,
        "spot": pd.DataFrame({"date": [today], "close": [spot]}),
        "zero_curve": pd.DataFrame({"date": today, "days": days, "rate": rate}),  # flat curve
        "div_yield": pd.DataFrame({"date": [today], "rate": [100 * (t.info.get("dividendYield") or 0)]}),
    }


def load_chain(ticker: str, today: str | None = None) -> dict[str, pd.DataFrame]:
    """{'options', 'spot', 'zero_curve', 'div_yield'} for one ticker and day, from data/<ticker>_<day>/ or Yahoo."""
    today = today or date.today().isoformat()
    cache = DATA_DIR / f"{ticker}_{today}"
    if all((cache / f"{n}.parquet").exists() for n in NAMES):
        return {n: pd.read_parquet(cache / f"{n}.parquet") for n in NAMES}
    out = _fetch(ticker, today)
    if (out["options"].best_bid > 0).sum() < 100:
        raise ValueError(f"{ticker} chain has no live bids: is the US market open?")
    cache.mkdir(parents=True, exist_ok=True)
    for n in NAMES:
        out[n].to_parquet(cache / f"{n}.parquet", index=False)
    return out
