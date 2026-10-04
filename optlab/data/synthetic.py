"""A simulated market for testing the machinery, not for judging the strategy.

Prices follow a stochastic-volatility model with crash jumps, and option
quotes come from Black-Scholes with a put skew, a volatility risk premium
and a bid-ask spread. That is enough to exercise every rule (regime flips,
stops, the stress test) with no data subscription. Results on this data say
nothing about real-world profitability: use real chains for that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..pricing import bs_price
from .base import CHAIN_COLUMNS, MarketData, empty_chain


@dataclass
class SyntheticParams:
    start: str = "2010-01-01"
    end: str = "2024-12-31"
    seed: int = 7
    s0: float = 200.0
    stock_s0: float = 50.0
    drift: float = 0.08
    kappa: float = 3.0  # variance mean reversion per year
    theta: float = 0.13**2  # long-run variance
    vol_of_var: float = 0.35
    rho: float = -0.7
    jump_per_year: float = 0.25
    jump_mean: float = -0.04
    jump_var_add: float = 0.04  # variance spike when a crash hits
    vrp_mult: float = 1.10  # implied vol = realized expectation x this, + add
    vrp_add: float = 0.02
    term_premium: float = 0.015  # extra implied vol at 3 months in calm markets
    skew: float = 0.15
    smile: float = 0.03
    index_half_spread: float = 0.02  # share of mid
    stock_half_spread: float = 0.04


def _fridays_between(date: pd.Timestamp, dte_min: int, dte_max: int) -> list[pd.Timestamp]:
    first = date + pd.Timedelta(days=dte_min)
    first += pd.Timedelta(days=(4 - first.weekday()) % 7)
    out = []
    d = first
    while (d - date).days <= dte_max:
        out.append(d)
        d += pd.Timedelta(days=7)
    return out


class SyntheticMarket(MarketData):
    """Index "SPY" plus any number of stocks with a beta to it."""

    def __init__(self, stocks: list[str] | None = None, params: SyntheticParams | None = None):
        self.p = params or SyntheticParams()
        rng = np.random.default_rng(self.p.seed)
        self._dates = pd.bdate_range(self.p.start, self.p.end)
        n = len(self._dates)
        dt = 1.0 / 252.0
        p = self.p

        var = np.empty(n)
        px = np.empty(n)
        var[0], px[0] = p.theta, p.s0
        z1 = rng.standard_normal(n)
        z2 = p.rho * z1 + math.sqrt(1 - p.rho**2) * rng.standard_normal(n)
        jumps = rng.random(n) < p.jump_per_year * dt
        for i in range(1, n):
            v = var[i - 1]
            dv = p.kappa * (p.theta - v) * dt + p.vol_of_var * math.sqrt(v * dt) * z2[i]
            ret = (p.drift - 0.5 * v) * dt + math.sqrt(v * dt) * z1[i]
            nv = max(v + dv, 0.0025)
            if jumps[i]:
                ret += p.jump_mean * (0.5 + rng.random())
                nv += p.jump_var_add
            var[i] = nv
            px[i] = px[i - 1] * math.exp(ret)
        self._index_var = pd.Series(var, self._dates)
        self._closes: dict[str, pd.Series] = {"SPY": pd.Series(px, self._dates)}
        self._iv_level: dict[str, pd.Series] = {}

        def expected_vol(days: float) -> np.ndarray:
            k = p.kappa * days / 365.0
            avg = p.theta + (var - p.theta) * (1 - math.exp(-k)) / k
            return np.sqrt(avg)

        iv30 = expected_vol(30) * p.vrp_mult + p.vrp_add
        iv93 = expected_vol(93) * p.vrp_mult + p.vrp_add + p.term_premium
        self._vix = pd.Series(iv30 * 100, self._dates)
        self._vix3m = pd.Series(iv93 * 100, self._dates)
        self._iv_level["SPY"] = pd.Series(iv30, self._dates)

        index_ret = np.diff(np.log(px), prepend=np.log(px[0]))
        for sym in stocks or []:
            beta = 0.8 + 0.6 * rng.random()
            idio = 0.15 + 0.25 * rng.random()
            # Slow-moving richness: some names' options trade rich, some cheap.
            rich = np.cumsum(rng.standard_normal(n)) * 0.002
            rich = 1.0 + 0.15 * np.tanh(rich - rich.mean())
            r = beta * index_ret + idio * math.sqrt(dt) * rng.standard_normal(n) - 0.5 * idio**2 * dt
            self._closes[sym] = pd.Series(p.stock_s0 * np.exp(np.cumsum(r)), self._dates)
            total = np.sqrt(beta**2 * expected_vol(30) ** 2 + idio**2)
            self._iv_level[sym] = pd.Series(total * rich * p.vrp_mult + p.vrp_add, self._dates)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self._dates

    @property
    def symbols(self) -> list[str]:
        return list(self._closes)

    def closes(self, symbol: str) -> pd.Series:
        return self._closes[symbol]

    def vix(self) -> pd.Series:
        return self._vix

    def vix3m(self) -> pd.Series:
        return self._vix3m

    def atm_iv(self, date: pd.Timestamp, symbol: str, dte: int = 30) -> float | None:
        return float(self._iv_level[symbol].loc[date])

    def _iv(self, symbol: str, date: pd.Timestamp, spot: float, strike: np.ndarray, t: np.ndarray) -> np.ndarray:
        atm = float(self._iv_level[symbol].loc[date])
        z = np.log(strike / spot) / (atm * np.sqrt(np.maximum(t, 1 / 365)))
        mult = 1 - self.p.skew * z + self.p.smile * z * z
        return atm * np.clip(mult, 0.6, 2.5)

    def _quotes(self, symbol: str, mid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        share = self.p.index_half_spread if symbol == "SPY" else self.p.stock_half_spread
        half = np.maximum(0.01, mid * share)
        bid = np.maximum(np.round(mid - half, 2), 0.0)
        ask = np.round(mid + half, 2)
        return bid, ask

    def chain(self, date: pd.Timestamp, symbol: str, dte_min: int = 0, dte_max: int = 365) -> pd.DataFrame:
        spot = float(self._closes[symbol].loc[date])
        step = 1.0 if symbol == "SPY" or spot < 300 else 5.0
        if spot < 50:
            step = 0.5
        strikes = np.arange(math.floor(spot * 0.5 / step) * step, spot * 1.4, step)
        rows = []
        for exp in _fridays_between(date, max(dte_min, 1), dte_max):
            t = np.full(len(strikes), (exp - date).days / 365.0)
            iv = self._iv(symbol, date, spot, strikes, t)
            for kind in ("P", "C"):
                mids = np.array([bs_price(spot, k, t[0], v, kind) for k, v in zip(strikes, iv)])
                keep = mids >= 0.01
                bid, ask = self._quotes(symbol, mids[keep])
                d1 = (np.log(spot / strikes[keep]) + 0.5 * iv[keep] ** 2 * t[0]) / (iv[keep] * math.sqrt(t[0]))
                delta = norm.cdf(d1) if kind == "C" else norm.cdf(d1) - 1
                rows.append(
                    pd.DataFrame(
                        {
                            "expiration": exp,
                            "strike": strikes[keep],
                            "kind": kind,
                            "bid": bid,
                            "ask": ask,
                            "mid": (bid + ask) / 2,
                            "iv": iv[keep],
                            "delta": delta,
                            "underlying": spot,
                        }
                    )
                )
        if not rows:
            return empty_chain()
        return pd.concat(rows, ignore_index=True)[CHAIN_COLUMNS]

    def quote(self, date, symbol, expiration, strike, kind):
        spot = float(self._closes[symbol].loc[date])
        t = max((expiration - date).days, 0) / 365.0
        iv = float(self._iv(symbol, date, spot, np.array([strike]), np.array([t]))[0])
        mid = bs_price(spot, strike, t, iv, kind)
        bid, ask = self._quotes(symbol, np.array([mid]))
        return float(bid[0]), float(ask[0])
