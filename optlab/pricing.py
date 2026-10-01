"""Black-Scholes pricing and Greeks, used to fill gaps in option data.

Free data sources rarely include Greeks, and the synthetic source has no
market at all, so every delta in this project comes from these formulas.
Rates and dividends default to zero; at 30-120 days that error is small
next to bid-ask spreads.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq
from scipy.stats import norm

MIN_T = 1.0 / 365.0 / 24.0  # one hour, avoids division by zero at expiry


def _d1_d2(spot: float, strike: float, t: float, vol: float, rate: float) -> tuple[float, float]:
    t = max(t, MIN_T)
    vol = max(vol, 1e-4)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    return d1, d1 - vol * math.sqrt(t)


def bs_price(spot: float, strike: float, t: float, vol: float, kind: str, rate: float = 0.0) -> float:
    """Price of a European call ("C") or put ("P")."""
    if t <= 0:
        return max(spot - strike, 0.0) if kind == "C" else max(strike - spot, 0.0)
    d1, d2 = _d1_d2(spot, strike, t, vol, rate)
    disc = math.exp(-rate * t)
    if kind == "C":
        return spot * norm.cdf(d1) - strike * disc * norm.cdf(d2)
    return strike * disc * norm.cdf(-d2) - spot * norm.cdf(-d1)


def bs_delta(spot: float, strike: float, t: float, vol: float, kind: str, rate: float = 0.0) -> float:
    """Delta per share: 0..1 for calls, -1..0 for puts."""
    if t <= 0:
        if kind == "C":
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0
    d1, _ = _d1_d2(spot, strike, t, vol, rate)
    return norm.cdf(d1) if kind == "C" else norm.cdf(d1) - 1.0


def bs_vega(spot: float, strike: float, t: float, vol: float, rate: float = 0.0) -> float:
    """Price change per 1.00 (100 vol points) change in volatility, per share."""
    if t <= 0:
        return 0.0
    d1, _ = _d1_d2(spot, strike, t, vol, rate)
    return spot * norm.pdf(d1) * math.sqrt(max(t, MIN_T))


def implied_vol(price: float, spot: float, strike: float, t: float, kind: str, rate: float = 0.0) -> float | None:
    """Volatility that reproduces `price`, or None if no solution exists."""
    intrinsic = max(spot - strike, 0.0) if kind == "C" else max(strike - spot, 0.0)
    if t <= 0 or price <= intrinsic + 1e-6:
        return None
    try:
        return brentq(lambda v: bs_price(spot, strike, t, v, kind, rate) - price, 1e-3, 5.0, xtol=1e-6)
    except ValueError:
        return None


def strike_for_delta(spot: float, t: float, vol: float, kind: str, target_delta: float, rate: float = 0.0) -> float:
    """Continuous strike whose delta equals `target_delta` (absolute value, e.g. 0.30)."""
    t = max(t, MIN_T)
    target = abs(target_delta)
    # Invert N(d1): calls have delta N(d1), puts N(d1) - 1.
    d1 = norm.ppf(target) if kind == "C" else norm.ppf(1.0 - target)
    return spot * math.exp(-(d1 * vol * math.sqrt(t)) + (rate + 0.5 * vol * vol) * t)
