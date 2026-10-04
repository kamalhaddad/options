"""Playbook signals: regime, valuation and direction.

Every function looks only at data up to and including `date`, so the
backtester cannot peek at the future.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import RegimeConfig

GREEN, YELLOW, RED = "green", "yellow", "red"
_RANK = {GREEN: 0, YELLOW: 1, RED: 2}


def _upto(s: pd.Series, date: pd.Timestamp) -> pd.Series:
    return s.loc[:date].dropna()


def realized_vol(closes: pd.Series, date: pd.Timestamp, days: int = 20) -> float | None:
    """Annualized close-to-close volatility over the last `days` returns (decimal)."""
    s = _upto(closes, date)
    if len(s) < days + 1:
        return None
    r = np.log(s.iloc[-(days + 1) :]).diff().dropna()
    return float(r.std(ddof=1) * math.sqrt(252))


@dataclass
class Regime:
    color: str
    size: float
    checks: dict[str, str] = field(default_factory=dict)
    values: dict[str, float] = field(default_factory=dict)


def regime(date: pd.Timestamp, index_closes: pd.Series, vix: pd.Series, vix3m: pd.Series, cfg: RegimeConfig) -> Regime:
    """The master switch. The worst of the four checks sets the color."""
    checks: dict[str, str] = {}
    values: dict[str, float] = {}

    v = _upto(vix, date)
    v3 = _upto(vix3m, date)
    if v.empty or v3.empty:
        return Regime(RED, cfg.size_red, {"data": RED}, {})
    vix_now, ratio = float(v.iloc[-1]), float(v.iloc[-1] / v3.iloc[-1])
    values["vix"], values["vix_vix3m"] = vix_now, ratio

    if ratio > cfg.term_red_above:
        checks["term_structure"] = RED
    elif ratio >= cfg.term_green_below:
        checks["term_structure"] = YELLOW
    else:
        checks["term_structure"] = GREEN

    rising = len(v) > 1 and vix_now > float(v.iloc[-2])
    if vix_now > cfg.vix_red_above and rising:
        checks["vix_level"] = RED
    elif vix_now < cfg.vix_thin_below or vix_now > cfg.vix_green_high:
        checks["vix_level"] = YELLOW
    elif vix_now < cfg.vix_green_low:
        checks["vix_level"] = YELLOW  # 13 to 15: thin but not the worst
    else:
        checks["vix_level"] = GREEN

    px = _upto(index_closes, date)
    if len(px) >= cfg.trend_ma_days + 20:
        ma = px.rolling(cfg.trend_ma_days).mean()
        last, ma_now, ma_prev = float(px.iloc[-1]), float(ma.iloc[-1]), float(ma.iloc[-21])
        gap = last / ma_now - 1
        values["spy_vs_200d"] = gap
        if gap > cfg.trend_band:
            checks["trend"] = GREEN
        elif gap < -cfg.trend_band and ma_now < ma_prev:
            checks["trend"] = RED
        else:
            checks["trend"] = YELLOW
    else:
        checks["trend"] = YELLOW

    rv = realized_vol(index_closes, date, cfg.rv_days)
    if rv is not None:
        gap = vix_now - rv * 100
        values["vix_minus_rv"] = gap
        if gap < 0:
            checks["realized_vs_implied"] = RED
        elif gap < cfg.rv_gap_green:
            checks["realized_vs_implied"] = YELLOW
        else:
            checks["realized_vs_implied"] = GREEN

    color = max(checks.values(), key=_RANK.__getitem__)
    size = {GREEN: cfg.size_green, YELLOW: cfg.size_yellow, RED: cfg.size_red}[color]
    return Regime(color, size, checks, values)


def iv_rank(iv_history: pd.Series, date: pd.Timestamp, lookback: int = 252) -> float | None:
    """(current - 52w low) / (52w high - 52w low), in 0..100."""
    s = _upto(iv_history, date).iloc[-lookback:]
    if len(s) < 20:
        return None
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-9:
        return 50.0
    return 100.0 * (float(s.iloc[-1]) - lo) / (hi - lo)


def iv_minus_rv(iv_now: float, closes: pd.Series, date: pd.Timestamp, days: int = 20) -> float | None:
    """30-day implied minus 20-day realized, in vol points."""
    rv = realized_vol(closes, date, days)
    return None if rv is None else (iv_now - rv) * 100


def trend(closes: pd.Series, date: pd.Timestamp) -> str:
    """bullish, bearish or neutral from the 50- and 200-day averages."""
    s = _upto(closes, date)
    if len(s) < 200:
        return "neutral"
    last, ma50, ma200 = float(s.iloc[-1]), float(s.iloc[-50:].mean()), float(s.iloc[-200:].mean())
    if last > ma50 > ma200:
        return "bullish"
    if last < ma50 < ma200:
        return "bearish"
    return "neutral"


def momentum_12_1(closes: pd.Series, date: pd.Timestamp) -> float | None:
    """12-month return excluding the latest month."""
    s = _upto(closes, date)
    if len(s) < 253:
        return None
    return float(s.iloc[-22] / s.iloc[-253] - 1)


def straddle_momentum(iv_history: pd.Series, closes: pd.Series, date: pd.Timestamp, hold: int = 21) -> float | None:
    """Average return of buying a 1-month at-the-money straddle over the past year.

    Approximates each straddle's cost as 0.8 x IV x sqrt(T) of the stock price
    and its payoff as the absolute move over the month. Positive means
    straddles paid off, and the playbook says not to sell premium there
    (option momentum, Heston, Jones and Khorram 2023).
    """
    px = _upto(closes, date)
    ivs = _upto(iv_history, date)
    if len(px) < 252 + hold or ivs.empty:
        return None
    window = px.iloc[-(252 + hold) :]
    rets = []
    for i in range(0, 252, hold):
        start_date = window.index[i]
        iv = ivs.loc[:start_date]
        if iv.empty:
            continue
        cost = 0.8 * float(iv.iloc[-1]) * math.sqrt(hold / 252)
        if not cost > 0:
            continue
        move = abs(float(window.iloc[i + hold] / window.iloc[i] - 1))
        rets.append(move / cost - 1)
    return float(np.mean(rets)) if rets else None
