"""Portfolio limits from the playbook's sizing section.

Checks run on the whole book with the candidate trade included, so many
small trades that are really one bet cannot sneak past the caps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .config import RiskConfig
from .pricing import bs_price, bs_vega
from .trades import MULT, Position


def _leg_years(leg, date: pd.Timestamp) -> float:
    return max((leg.expiration - date).days, 0) / 365.0


def stress_loss(book: list[Position], spots: dict[str, float], date: pd.Timestamp, cfg: RiskConfig) -> float:
    """Dollars lost if every underlying gaps by `stress_spot_move` and every IV multiplies."""
    loss = 0.0
    for pos in book:
        spot = spots[pos.symbol]
        shocked = spot * (1 + cfg.stress_spot_move)
        for leg in pos.legs:
            t = _leg_years(leg, date)
            iv = max(leg.iv, 0.05)
            now = bs_price(spot, leg.strike, t, iv, leg.kind)
            after = bs_price(shocked, leg.strike, t, iv * cfg.stress_vol_multiplier, leg.kind)
            loss -= leg.qty * (after - now) * MULT * pos.contracts
    return loss


def vega_loss(book: list[Position], spots: dict[str, float], date: pd.Timestamp, cfg: RiskConfig) -> float:
    """Dollars lost if implied vol rises `vega_shock_points` on everything at once."""
    loss = 0.0
    for pos in book:
        for leg in pos.legs:
            vega = bs_vega(spots[pos.symbol], leg.strike, _leg_years(leg, date), max(leg.iv, 0.05))
            loss -= leg.qty * vega * cfg.vega_shock_points / 100 * MULT * pos.contracts
    return loss


@dataclass
class RiskCheck:
    ok: bool
    reason: str = ""
    max_loss_total: float = 0.0
    stress: float = 0.0
    vega: float = 0.0


def check_book(
    book: list[Position], spots: dict[str, float], date: pd.Timestamp, equity: float, cfg: RiskConfig
) -> RiskCheck:
    total = sum(p.risk_dollars for p in book if p.sleeve != "hedge" and p.notes.get("risk_basis") != "stress")
    stress = stress_loss(book, spots, date, cfg)
    vega = vega_loss(book, spots, date, cfg)
    res = RiskCheck(True, "", total, stress, vega)
    if total > cfg.portfolio_max_loss_pct * equity + 1e-6:
        res.ok, res.reason = False, f"combined max loss ${total:,.0f} over {cfg.portfolio_max_loss_pct:.0%} cap"
    elif stress > cfg.stress_max_loss_pct * equity:
        res.ok, res.reason = False, f"stress loss ${stress:,.0f} over {cfg.stress_max_loss_pct:.0%} cap"
    elif vega > cfg.vega_max_loss_pct * equity:
        res.ok, res.reason = False, f"vega shock ${vega:,.0f} over {cfg.vega_max_loss_pct:.0%} cap"
    return res


def contracts_for(max_loss_per_contract: float, budget: float) -> int:
    """How many contracts keep total max loss within `budget`."""
    if max_loss_per_contract <= 0:
        return 0
    return int(math.floor(budget / max_loss_per_contract + 1e-9))


def drawdown_multiplier(equity: float, peak: float, cfg: RiskConfig) -> float:
    """Size multiplier from the drawdown circuit breaker (see `selling_halted` for the stop)."""
    dd = 1 - equity / peak if peak > 0 else 0.0
    return 0.5 if dd >= cfg.drawdown_half_size else 1.0


def selling_halted(
    equity: float, peak: float, vix: pd.Series, vix3m: pd.Series, date: pd.Timestamp, cfg: RiskConfig
) -> bool:
    """No new short premium in a deep drawdown until VIX has been in contango five straight days."""
    if peak <= 0 or 1 - equity / peak < cfg.drawdown_stop_selling:
        return False
    ratio = (vix.loc[:date] / vix3m.loc[:date]).dropna().iloc[-5:]
    return not (len(ratio) == 5 and bool((ratio < 1.0).all()))
