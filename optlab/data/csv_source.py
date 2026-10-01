"""Historical data from files you download or buy.

Option chains use the column names of the Optopsy backtester, which most
vendor exports can be renamed to:

    underlying_symbol, quote_date, expiration, strike, option_type, bid, ask,
    underlying_price, and optionally implied_volatility, delta

`option_type` may be c/p or call/put. Missing iv and delta are computed with
Black-Scholes. Pass `rename={"vendor_col": "our_col"}` for other layouts.

VIX and VIX3M come from Cboe's free daily history files (VIX_History.csv,
VIX3M_History.csv), which have DATE and CLOSE columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import CHAIN_COLUMNS, MarketData, enrich_chain

CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"


def read_cboe_index(source: str | Path) -> pd.Series:
    """Read a Cboe daily index history file (path or URL) into a close series."""
    df = pd.read_csv(source)
    df.columns = [c.strip().upper() for c in df.columns]
    close_col = "CLOSE" if "CLOSE" in df.columns else df.columns[-1]
    s = pd.Series(df[close_col].astype(float).values, index=pd.to_datetime(df["DATE"]))
    return s.sort_index()


def cboe_index(name: str) -> pd.Series:
    """Download a Cboe index history, e.g. "VIX", "VIX3M" or "PUT"."""
    return read_cboe_index(CBOE_URL.format(name=name))


def read_chains(paths: list[str | Path], rename: dict[str, str] | None = None) -> pd.DataFrame:
    frames = [pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if rename:
        df = df.rename(columns=rename)
    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["kind"] = df["option_type"].astype(str).str[0].str.upper()
    df = df.rename(columns={"underlying_price": "underlying", "implied_volatility": "iv"})
    return df


class CsvMarket(MarketData):
    def __init__(
        self,
        chains: pd.DataFrame,
        vix: pd.Series,
        vix3m: pd.Series,
        closes: dict[str, pd.Series] | None = None,
    ):
        self._chains = {
            (sym, date): g.drop(columns=["underlying_symbol", "quote_date"])
            for (sym, date), g in chains.groupby(["underlying_symbol", "quote_date"])
        }
        derived = chains.groupby(["underlying_symbol", "quote_date"])["underlying"].first()
        self._closes = {sym: derived.loc[sym].sort_index() for sym in derived.index.get_level_values(0).unique()}
        self._closes.update(closes or {})
        self._dates = pd.DatetimeIndex(sorted(chains["quote_date"].unique()))
        self._vix = vix
        self._vix3m = vix3m
        self._enriched: dict[tuple[str, pd.Timestamp], pd.DataFrame] = {}

    @classmethod
    def from_files(
        cls,
        chain_paths: list[str | Path],
        vix_path: str | Path,
        vix3m_path: str | Path,
        rename: dict[str, str] | None = None,
    ) -> CsvMarket:
        return cls(read_chains(chain_paths, rename), read_cboe_index(vix_path), read_cboe_index(vix3m_path))

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self._dates

    def closes(self, symbol: str) -> pd.Series:
        return self._closes[symbol]

    def vix(self) -> pd.Series:
        return self._vix

    def vix3m(self) -> pd.Series:
        return self._vix3m

    def _full_chain(self, date: pd.Timestamp, symbol: str) -> pd.DataFrame:
        key = (symbol, date)
        if key not in self._enriched:
            raw = self._chains.get(key)
            self._enriched[key] = (
                pd.DataFrame(columns=CHAIN_COLUMNS) if raw is None else enrich_chain(raw.reset_index(drop=True), date)
            )
        return self._enriched[key]

    def chain(self, date, symbol, dte_min=0, dte_max=365):
        ch = self._full_chain(date, symbol)
        dte = (ch.expiration - date).dt.days
        return ch[(dte >= dte_min) & (dte <= dte_max)].reset_index(drop=True)

    def quote(self, date, symbol, expiration, strike, kind):
        ch = self._full_chain(date, symbol)
        row = ch[(ch.expiration == expiration) & np.isclose(ch.strike, strike) & (ch.kind == kind)]
        if row.empty:
            return None
        return float(row.bid.iloc[0]), float(row.ask.iloc[0])
