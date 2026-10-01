import pandas as pd

from optlab.config import Config
from optlab.planner import Planner
from optlab.trades import Leg, Position

EXP = pd.Timestamp("2021-03-19")


def _short(entry=-0.30):
    legs = [Leg(EXP, 95, "P", -1), Leg(EXP, 94, "P", 1)]
    return Position("SPY", "put_spread", "index", legs, 1, pd.Timestamp("2021-02-01"), entry, 70)


def test_short_premium_exits(market):
    p = Planner(market, Config())
    pos = _short()
    early = pd.Timestamp("2021-02-05")
    assert p.exit_reason(early, pos, -0.14).startswith("profit")
    assert p.exit_reason(early, pos, -0.90).startswith("stop")
    assert p.exit_reason(early, pos, -0.25) is None
    assert p.exit_reason(EXP - pd.Timedelta(days=21), pos, -0.25) == "21 DTE"
    assert p.exit_reason(EXP, pos, -0.25) == "expired"


def test_long_option_exits(market):
    p = Planner(market, Config())
    legs = [Leg(EXP, 95, "C", 1)]
    pos = Position("SPY", "long_call", "directional", legs, 1, pd.Timestamp("2021-01-04"), 2.0, 200)
    early = pd.Timestamp("2021-01-15")
    assert p.exit_reason(early, pos, 4.1).startswith("profit")
    assert p.exit_reason(early, pos, 0.9).startswith("stop")
    assert p.exit_reason(early, pos, 2.2) is None
