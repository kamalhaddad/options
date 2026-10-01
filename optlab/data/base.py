"""The data interface every source implements.

The backtester and scanner only talk to `MarketData`, so a paid historical
feed (ORATS, Cboe DataShop, Polygon) can replace the free or synthetic
sources by implementing these few methods, or by exporting to the CSV
layout `CsvMarket` reads.

Chain frames always have these columns:
    expiration (Timestamp), strike (float), kind ("C" or "P"),
    bid, ask, mid, iv, delta, underlying
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ..pricing import bs_delta, implied_vol

CHAIN_COLUMNS = ["expiration", "strike", "kind", "bid", "ask", "mid", "iv", "delta", "underlying"]


class MarketData(ABC):
    """Daily end-of-day market data for a set of symbols."""

    @property
    @abstractmethod
    def dates(self) -> pd.DatetimeIndex:
        """Trading days with data, ascending."""

    @abstractmethod
    def closes(self, symbol: str) -> pd.Series:
        """Daily closing prices indexed by date."""

    @abstractmethod
    def vix(self) -> pd.Series:
        """VIX closes (30-day S&P 500 implied vol, in points)."""

    @abstractmethod
    def vix3m(self) -> pd.Series:
        """VIX3M closes (93-day S&P 500 implied vol, in points)."""

    @abstractmethod
    def chain(self, date: pd.Timestamp, symbol: str, dte_min: int = 0, dte_max: int = 365) -> pd.DataFrame:
        """Option chain for `symbol` on `date`, limited to expirations in the DTE window."""

    def quote(
        self, date: pd.Timestamp, symbol: str, expiration: pd.Timestamp, strike: float, kind: str
    ) -> tuple[float, float] | None:
        """Bid and ask of one contract, or None when it is not quoted."""
        dte = (expiration - date).days
        ch = self.chain(date, symbol, dte, dte)
        row = ch[(ch.expiration == expiration) & np.isclose(ch.strike, strike) & (ch.kind == kind)]
        if row.empty:
            return None
        return float(row.bid.iloc[0]), float(row.ask.iloc[0])

    def atm_iv(self, date: pd.Timestamp, symbol: str, dte: int = 30) -> float | None:
        """At-the-money implied vol (as a decimal) near `dte` days out."""
        ch = self.chain(date, symbol, max(dte - 15, 1), dte + 20)
        ch = ch.dropna(subset=["iv"])
        if ch.empty:
            return None
        ch = ch.assign(dte_gap=(ch.expiration - date).dt.days.sub(dte).abs())
        ch = ch[ch.dte_gap == ch.dte_gap.min()]
        spot = float(ch.underlying.iloc[0])
        ch = ch.assign(dist=(ch.strike - spot).abs())
        return float(ch.nsmallest(4, "dist").iv.mean())


def enrich_chain(df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """Add mid, and fill missing iv and delta from Black-Scholes."""
    df = df.copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    if "iv" not in df:
        df["iv"] = np.nan
    if "delta" not in df:
        df["delta"] = np.nan
    t = (df["expiration"] - date).dt.days.clip(lower=0) / 365.0
    for i in df.index[df["iv"].isna()]:
        df.at[i, "iv"] = implied_vol(df.at[i, "mid"], df.at[i, "underlying"], df.at[i, "strike"], t[i], df.at[i, "kind"])
    for i in df.index[df["delta"].isna() & df["iv"].notna()]:
        df.at[i, "delta"] = bs_delta(df.at[i, "underlying"], df.at[i, "strike"], t[i], df.at[i, "iv"], df.at[i, "kind"])
    return df[CHAIN_COLUMNS]
