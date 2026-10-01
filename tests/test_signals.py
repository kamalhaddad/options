import numpy as np
import pandas as pd
import pytest

from optlab import signals as sig
from optlab.config import RegimeConfig


def _series(values, end="2020-12-31"):
    idx = pd.bdate_range(end=end, periods=len(values))
    return pd.Series(values, idx, dtype=float)


def _calm_market(n=300):
    rng = np.random.default_rng(0)
    px = _series(100 * np.exp(np.cumsum(0.0005 + 0.005 * rng.standard_normal(n))))
    return px


def test_green_when_calm_and_contango():
    px = _calm_market()
    vix = _series([18.0] * len(px))
    vix3m = _series([21.0] * len(px))
    r = sig.regime(px.index[-1], px, vix, vix3m, RegimeConfig())
    assert r.checks["term_structure"] == sig.GREEN
    assert r.checks["vix_level"] == sig.GREEN
    assert r.checks["realized_vs_implied"] == sig.GREEN


def test_backwardation_is_red():
    px = _calm_market()
    vix = _series([35.0] * len(px))
    vix3m = _series([30.0] * len(px))
    r = sig.regime(px.index[-1], px, vix, vix3m, RegimeConfig())
    assert r.color == sig.RED and r.size == 0


def test_iv_rank_bounds():
    s = _series(np.linspace(0.1, 0.3, 100))
    assert sig.iv_rank(s, s.index[-1]) == pytest.approx(100)
    assert sig.iv_rank(s, s.index[0]) is None  # not enough history


def test_trend_bullish_on_rising_prices():
    s = _series(np.linspace(50, 100, 260))
    assert sig.trend(s, s.index[-1]) == "bullish"
    assert sig.trend(s[::-1].set_axis(s.index), s.index[-1]) == "bearish"


def test_signals_ignore_the_future():
    s = _calm_market(400)
    date = s.index[300]
    future_changed = s.copy()
    future_changed.iloc[301:] *= 3
    assert sig.trend(s, date) == sig.trend(future_changed, date)
    assert sig.realized_vol(s, date) == sig.realized_vol(future_changed, date)
