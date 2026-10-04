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


def sql(db: str, q: str, tries: int = 6) -> list[dict]:
    url = API.format(db=db, q=urllib.parse.quote(q))
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                d = json.load(r)
            if d.get("query_execution_status") == "Success":
                return d.get("rows", [])
            raise RuntimeError(f"{d.get('query_execution_status')}: {d.get('query_execution_message')}")
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(min(2 ** (attempt + 1), 60))
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


def safe_fetch(day: str, symbols: list[str]) -> tuple[str, pd.DataFrame | None]:
    """One day's chains, or None after repeated errors (logged, retried later)."""
    try:
        return day, fetch_day(day, symbols)
    except Exception as e:  # noqa: BLE001 - one bad day should not sink the download
        print(f"{day}: {type(e).__name__}: {e}", flush=True)
        return day, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-failed", type=float, default=0.02, help="fail if more than this share of days errors")
    args = ap.parse_args()

    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(args.start, args.end)]
    frames, failed, done = [], [], 0
    with ThreadPoolExecutor(args.workers) as pool:
        for day, df in pool.map(lambda d: safe_fetch(d, args.symbols), days):
            done += 1
            if df is None:
                failed.append(day)
            elif not df.empty:
                frames.append(df)
            if done % 100 == 0:
                print(f"{done}/{len(days)} days, {len(frames)} with data, {len(failed)} failed", flush=True)
    # Second pass, one at a time, for days that errored (usually rate limits).
    still_failed = []
    for day in failed:
        time.sleep(5)
        _, df = safe_fetch(day, args.symbols)
        if df is None:
            still_failed.append(day)
        elif not df.empty:
            frames.append(df)
    if still_failed:
        print(f"{len(still_failed)} days could not be fetched: {', '.join(still_failed)}")
    if len(still_failed) > args.max_failed * len(days):
        raise SystemExit("Too many days failed to download")
    out = pd.concat(frames, ignore_index=True).sort_values(["quote_date", "underlying_symbol"])
    out.to_csv(args.out, index=False)
    by_sym = out.groupby("underlying_symbol").quote_date.agg(["min", "max", "nunique"])
    print(f"Wrote {len(out):,} rows to {args.out}")
    print(by_sym.to_string())


if __name__ == "__main__":
    main()
