"""Free, delayed data from Yahoo Finance for the daily scanner.

Yahoo gives price history for stocks and Cboe volatility indices and today's
option chains, but no option history, so this source is for scanning only.
Install the extra with `pip install -e .[live]`.
"""

from __future__ import annotations

import pandas as pd

from .base import CHAIN_COLUMNS, MarketData, enrich_chain


class YahooMarket(MarketData):
    def __init__(self, symbols: list[str], history: str = "2y"):
        import yfinance as yf  # optional dependency

        self._yf = yf
        tickers = sorted(set(symbols) | {"^VIX", "^VIX3M"})
        raw = yf.download(tickers, period=history, auto_adjust=True, progress=False)["Close"]
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        self._prices = raw.dropna(how="all")
        self._cache: dict[str, pd.DataFrame] = {}

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self._prices.index)

    def closes(self, symbol: str) -> pd.Series:
        return self._prices[symbol].dropna()

    def vix(self) -> pd.Series:
        return self._prices["^VIX"].dropna()

    def vix3m(self) -> pd.Series:
        return self._prices["^VIX3M"].dropna()

    def _today_chain(self, symbol: str, max_dte: int) -> pd.DataFrame:
        if symbol in self._cache:
            return self._cache[symbol]
        today = self.dates[-1]
        ticker = self._yf.Ticker(symbol)
        spot = float(self.closes(symbol).iloc[-1])
        frames = []
        for exp in ticker.options:
            exp_ts = pd.Timestamp(exp)
            if (exp_ts - today).days > max_dte:
                break
            oc = ticker.option_chain(exp)
            for kind, df in (("C", oc.calls), ("P", oc.puts)):
                frames.append(
                    pd.DataFrame(
                        {
                            "expiration": exp_ts,
                            "strike": df["strike"].astype(float),
                            "kind": kind,
                            "bid": df["bid"].astype(float),
                            "ask": df["ask"].astype(float),
                            "underlying": spot,
                        }
                    )
                )
        if not frames:
            chain = pd.DataFrame(columns=CHAIN_COLUMNS)
        else:
            chain = pd.concat(frames, ignore_index=True)
            chain = chain[(chain.ask > 0) & (chain.bid >= 0)]
            chain = enrich_chain(chain.reset_index(drop=True), today)
        self._cache[symbol] = chain
        return chain

    def chain(self, date, symbol, dte_min=0, dte_max=365):
        if date != self.dates[-1]:
            raise ValueError("Yahoo only has today's option chain")
        ch = self._today_chain(symbol, 130)
        dte = (ch.expiration - date).dt.days
        return ch[(dte >= dte_min) & (dte <= dte_max)].reset_index(drop=True)
