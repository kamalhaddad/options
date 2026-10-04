"""Download free daily option chains from DoltHub into optlab's CSV layout.

Uses the post-no-preference/options and /stocks databases through DoltHub's
SQL API, one query per trading day (the tables are keyed by date first, so
these are fast). Output columns: underlying_symbol, quote_date, expiration,
strike, option_type, bid, ask, underlying_price, implied_volatility, delta.

    python scripts/fetch_dolthub.py --start 2019-01-01 --end 2026-09-30 \\
        --symbols SPY T PFE --out data/dolthub_chains.csv.gz
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/{db}/master?q={q}"


def sql(db: str, q: str, tries: int = 4) -> list[dict]:
    url = API.format(db=db, q=urllib.parse.quote(q))
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                d = json.load(r)
            if d.get("query_execution_status") == "Success":
                return d.get("rows", [])
            raise RuntimeError(d.get("query_execution_message"))
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2**attempt)
    return []


def fetch_day(day: str, symbols: list[str]) -> pd.DataFrame:
    names = ",".join(f"'{s}'" for s in symbols)
    opts = sql(
        "options",
        "SELECT date, act_symbol, expiration, strike, call_put, bid, ask, vol, delta "
        f"FROM option_chain WHERE date='{day}' AND act_symbol IN ({names})",
    )
    if not opts:
        return pd.DataFrame()
    px = sql("stocks", f"SELECT act_symbol, close FROM ohlcv WHERE date='{day}' AND act_symbol IN ({names})")
    closes = {r["act_symbol"]: float(r["close"]) for r in px}
    df = pd.DataFrame(opts)
    df["underlying_price"] = df["act_symbol"].map(closes)
    df = df.dropna(subset=["underlying_price", "bid", "ask"])
    return pd.DataFrame(
        {
            "underlying_symbol": df["act_symbol"],
            "quote_date": df["date"],
            "expiration": df["expiration"],
            "strike": df["strike"].astype(float),
            "option_type": df["call_put"].str[0].str.lower(),
            "bid": df["bid"].astype(float),
            "ask": df["ask"].astype(float),
            "underlying_price": df["underlying_price"],
            "implied_volatility": pd.to_numeric(df["vol"], errors="coerce"),
            "delta": pd.to_numeric(df["delta"], errors="coerce"),
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(args.start, args.end)]
    frames, done = [], 0
    with ThreadPoolExecutor(args.workers) as pool:
        for df in pool.map(lambda d: fetch_day(d, args.symbols), days):
            done += 1
            if not df.empty:
                frames.append(df)
            if done % 100 == 0:
                print(f"{done}/{len(days)} days, {len(frames)} with data", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(args.out, index=False)
    by_sym = out.groupby("underlying_symbol").quote_date.agg(["min", "max", "nunique"])
    print(f"Wrote {len(out):,} rows to {args.out}")
    print(by_sym.to_string())


if __name__ == "__main__":
    main()
