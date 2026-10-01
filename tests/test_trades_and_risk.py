import pandas as pd
import pytest

from optlab.config import FillConfig, RiskConfig
from optlab.risk import check_book, contracts_for, drawdown_multiplier, selling_halted, stress_loss
from optlab.trades import Leg, Position, fill_price, max_loss_per_contract

EXP = pd.Timestamp("2021-02-19")
DATE = pd.Timestamp("2021-01-08")


def put_spread(contracts=1):
    legs = [Leg(EXP, 95, "P", -1, 1.00, 1.10, -0.3, 0.25), Leg(EXP, 94, "P", +1, 0.70, 0.78, -0.25, 0.26)]
    entry = fill_price(legs, opening=True, fills=FillConfig())
    return Position(
        "SPY",
        "put_spread",
        "index",
        legs,
        contracts,
        DATE,
        entry,
        max_loss_per_contract("put_spread", legs, entry, 1.0),
    )


def test_fill_pays_part_of_the_spread():
    pos = put_spread()
    mid_credit = 1.05 - 0.74
    assert -pos.entry_price == pytest.approx(mid_credit - 0.25 * 0.10 - 0.25 * 0.08)
    assert pos.max_loss == pytest.approx((1 + pos.entry_price) * 100)


def test_stress_loss_is_positive_and_capped_by_width():
    pos = put_spread(2)
    loss = stress_loss([pos], {"SPY": 100.0}, DATE, RiskConfig())
    assert 0 < loss <= 2 * 100 * 1.0


def test_book_cap_rejects_oversized_book():
    pos = put_spread(30)
    chk = check_book([pos], {"SPY": 100.0}, DATE, 5000, RiskConfig())
    assert not chk.ok


def test_contracts_for_budget():
    assert contracts_for(70, 100) == 1
    assert contracts_for(120, 100) == 0
    assert contracts_for(0, 100) == 0


def test_drawdown_breakers():
    cfg = RiskConfig()
    assert drawdown_multiplier(95, 100, cfg) == 1.0
    assert drawdown_multiplier(89, 100, cfg) == 0.5


def test_selling_halt_waits_for_contango():
    idx = pd.bdate_range("2021-01-01", periods=10)
    cfg = RiskConfig()
    backwardation = pd.Series(30.0, idx), pd.Series(25.0, idx)
    contango = pd.Series(18.0, idx), pd.Series(21.0, idx)
    assert selling_halted(84, 100, *backwardation, idx[-1], cfg)
    assert not selling_halted(84, 100, *contango, idx[-1], cfg)
    assert not selling_halted(90, 100, *backwardation, idx[-1], cfg)
