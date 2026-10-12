"""Pull one trading day of SPX option data from WRDS OptionMetrics and cache it as parquet.

No filtering happens here beyond secid and date: cleaning is a separate, explicit step.
Credentials come from ~/.pgpass; nothing is stored in code.
"""

from pathlib import Path

import pandas as pd

SPX_SECID = 108105
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

OPTION_COLS = [
    "date", "exdate", "cp_flag", "strike_price", "best_bid", "best_offer",
    "volume", "open_interest", "impl_volatility", "delta", "am_settlement",
    "exercise_style", "ss_flag", "symbol",
]


def _query_day(db, date: str) -> dict[str, pd.DataFrame]:
    year = date[:4]
    cols = ", ".join(OPTION_COLS)
    q = {
        "options": f"""
            select {cols} from optionm.opprcd{year}
            where secid = {SPX_SECID} and date = '{date}'""",
        "spot": f"""
            select date, close from optionm.secprd
            where secid = {SPX_SECID} and date = '{date}'""",
        "zero_curve": f"""
            select date, days, rate from optionm.zerocd
            where date = '{date}' order by days""",
        "div_yield": f"""
            select date, rate from optionm.idxdvd
            where secid = {SPX_SECID} and date = '{date}'""",
    }
    out = {name: db.raw_sql(sql, date_cols=["date"]) for name, sql in q.items()}
    opt = out["options"]
    opt["exdate"] = pd.to_datetime(opt["exdate"])
    opt["strike"] = opt["strike_price"] / 1000.0  # OptionMetrics stores strike x1000
    out["options"] = opt.drop(columns="strike_price")
    return out


def load_day(date: str, db=None) -> dict[str, pd.DataFrame]:
    """Return {'options', 'spot', 'zero_curve', 'div_yield'} for one date (YYYY-MM-DD).

    Reads from data/<date>/ if cached, otherwise queries WRDS (opening a
    connection if `db` is None) and writes the cache.
    """
    cache = DATA_DIR / date
    names = ["options", "spot", "zero_curve", "div_yield"]
    if all((cache / f"{n}.parquet").exists() for n in names):
        return {n: pd.read_parquet(cache / f"{n}.parquet") for n in names}

    if db is None:
        import wrds
        db = wrds.Connection()
    out = _query_day(db, date)
    if out["options"].empty:
        raise ValueError(f"No SPX options on {date}: not a trading day or outside subscription")
    cache.mkdir(parents=True, exist_ok=True)
    for n in names:
        out[n].to_parquet(cache / f"{n}.parquet", index=False)
    return out
