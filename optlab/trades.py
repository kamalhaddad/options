"""Positions and the trade structures the playbook uses.

Prices are per share; dollar amounts multiply by 100 per contract. A
position's value is what it is worth to us: negative for a credit trade we
would have to buy back, positive for options we own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import EntryConfig, FillConfig

MULT = 100


@dataclass
class Leg:
    expiration: pd.Timestamp
    strike: float
    kind: str  # "C" or "P"
    qty: int  # +1 long, -1 short, per unit of the position
    bid: float = 0.0
    ask: float = 0.0
    delta: float = 0.0
    iv: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


@dataclass
class Position:
    symbol: str
    structure: str  # put_spread, call_spread, iron_condor, csp, long_call, long_put, hedge_put
    sleeve: str  # index, single_stock, directional, hedge
    legs: list[Leg]
    contracts: int
    entry_date: pd.Timestamp
    entry_price: float  # net per share at fill, signed: negative = credit received
    max_loss: float  # dollars per contract
    id: int = 0
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    commissions: float = 0.0
    notes: dict = field(default_factory=dict)

    @property
    def is_short_premium(self) -> bool:
        return self.entry_price < 0

    @property
    def credit(self) -> float:
        return -self.entry_price if self.is_short_premium else 0.0

    @property
    def expiration(self) -> pd.Timestamp:
        return min(leg.expiration for leg in self.legs)

    def dte(self, date: pd.Timestamp) -> int:
        return (self.expiration - date).days

    @property
    def risk_dollars(self) -> float:
        return self.max_loss * self.contracts

    def pnl(self, value_now: float) -> float:
        """Dollar P&L if closed at `value_now` per share, before exit commissions."""
        return (value_now - self.entry_price) * MULT * self.contracts - self.commissions


def fill_price(legs: list[Leg], opening: bool, fills: FillConfig) -> float:
    """Net per-share price of trading all legs, paying part of each spread.

    Opening buys the long legs and sells the short ones; closing does the
    reverse. The result is the position's value at that fill.
    """
    total = 0.0
    for leg in legs:
        width = max(leg.ask - leg.bid, 0.0)
        buying = (leg.qty > 0) == opening
        px = leg.mid + fills.spread_share * width if buying else leg.mid - fills.spread_share * width
        total += leg.qty * max(px, 0.0)
    return total


def mid_value(legs: list[Leg]) -> float:
    return sum(leg.qty * leg.mid for leg in legs)


# ---------------------------------------------------------------- selection


def pick_expiration(chain: pd.DataFrame, date: pd.Timestamp, dte_min: int, dte_max: int) -> pd.Timestamp | None:
    """The expiration in the window closest to its far end, to collect the most decay."""
    exps = sorted(e for e in chain.expiration.unique() if dte_min <= (e - date).days <= dte_max)
    return exps[-1] if exps else None


def _leg_from_row(row: pd.Series, qty: int) -> Leg:
    return Leg(
        row.expiration,
        float(row.strike),
        row.kind,
        qty,
        float(row.bid),
        float(row.ask),
        float(row.delta),
        float(row.iv),
    )


def pick_by_delta(chain: pd.DataFrame, expiration: pd.Timestamp, kind: str, target: float) -> pd.Series | None:
    side = chain[(chain.expiration == expiration) & (chain.kind == kind) & chain.delta.notna() & (chain.bid > 0)]
    if side.empty:
        return None
    return side.loc[(side.delta.abs() - abs(target)).abs().idxmin()]


def pick_wing(chain: pd.DataFrame, short: pd.Series, width: float) -> pd.Series | None:
    """Long leg `width` dollars further out of the money than `short`."""
    side = chain[(chain.expiration == short.expiration) & (chain.kind == short.kind)]
    if short.kind == "P":
        side = side[side.strike <= short.strike - width + 1e-9]
        return None if side.empty else side.loc[side.strike.idxmax()]
    side = side[side.strike >= short.strike + width - 1e-9]
    return None if side.empty else side.loc[side.strike.idxmin()]


def _quote_ok(row: pd.Series, max_pct: float) -> bool:
    mid = (row.bid + row.ask) / 2
    return mid > 0 and (row.ask - row.bid) / mid <= max_pct


# ---------------------------------------------------------------- builders
# Each returns (legs, structure, max_loss_per_contract_estimate) or a reason string.


def vertical_spread(chain, date, kind, cfg: EntryConfig, delta: float):
    exp = pick_expiration(chain, date, cfg.short_dte_min, cfg.short_dte_max)
    if exp is None:
        return f"no expiration in {cfg.short_dte_min}-{cfg.short_dte_max} DTE"
    short = pick_by_delta(chain, exp, kind, delta)
    if short is None:
        return "no strike near target delta"
    wing = pick_wing(chain, short, cfg.wing_width)
    if wing is None:
        return "no wing strike"
    if not (_quote_ok(short, cfg.max_spread_pct_of_mid) and _quote_ok(wing, max(cfg.max_spread_pct_of_mid, 0.5))):
        return "bid-ask spread too wide"
    legs = [_leg_from_row(short, -1), _leg_from_row(wing, +1)]
    width = abs(short.strike - wing.strike)
    credit = -mid_value(legs)
    if credit / width < cfg.min_credit_to_width_spread:
        return f"credit {credit:.2f} under {cfg.min_credit_to_width_spread:.0%} of {width:g} width"
    return legs, "put_spread" if kind == "P" else "call_spread", width


def iron_condor(chain, date, cfg: EntryConfig):
    exp = pick_expiration(chain, date, cfg.short_dte_min, cfg.short_dte_max)
    if exp is None:
        return f"no expiration in {cfg.short_dte_min}-{cfg.short_dte_max} DTE"
    legs = []
    widths = []
    for kind in ("P", "C"):
        short = pick_by_delta(chain, exp, kind, cfg.condor_delta)
        wing = None if short is None else pick_wing(chain, short, cfg.wing_width)
        if short is None or wing is None:
            return "missing condor strikes"
        if not _quote_ok(short, cfg.max_spread_pct_of_mid):
            return "bid-ask spread too wide"
        legs += [_leg_from_row(short, -1), _leg_from_row(wing, +1)]
        widths.append(abs(short.strike - wing.strike))
    credit = -mid_value(legs)
    if credit / max(widths) < cfg.min_credit_to_width_condor:
        return f"condor credit {credit:.2f} under {cfg.min_credit_to_width_condor:.0%} of width"
    return legs, "iron_condor", max(widths)


def cash_secured_put(chain, date, cfg: EntryConfig):
    exp = pick_expiration(chain, date, cfg.short_dte_min, cfg.short_dte_max)
    if exp is None:
        return "no expiration in window"
    short = pick_by_delta(chain, exp, "P", cfg.put_spread_delta)
    if short is None or not _quote_ok(short, cfg.max_spread_pct_of_mid):
        return "no liquid put near target delta"
    return [_leg_from_row(short, -1)], "csp", float(short.strike)


def long_option(chain, date, kind, cfg: EntryConfig, delta: float, dte_min: int, dte_max: int, structure: str):
    exp = pick_expiration(chain, date, dte_min, dte_max)
    if exp is None:
        return f"no expiration in {dte_min}-{dte_max} DTE"
    row = pick_by_delta(chain, exp, kind, delta)
    if row is None:
        return "no strike near target delta"
    if structure != "hedge_put" and not _quote_ok(row, cfg.max_spread_pct_of_mid):
        return "bid-ask spread too wide"
    return [_leg_from_row(row, +1)], structure, 0.0


def max_loss_per_contract(structure: str, legs: list[Leg], entry_price: float, width: float) -> float:
    """Worst-case dollars lost per contract, given the actual fill.

    For credit trades `width` is the spread width, or the strike for a
    cash-secured put; you lose it minus the credit. Debits lose what was paid.
    """
    if entry_price >= 0:
        return entry_price * MULT
    return (width + entry_price) * MULT
