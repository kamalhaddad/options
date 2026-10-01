"""Day-by-day backtest of the playbook on any `MarketData` source."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import Config
from .data.base import MarketData
from .metrics import summarize
from .planner import BookState, Planner
from .pricing import bs_price
from .risk import selling_halted
from .trades import MULT, Position


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: list[Position]
    regimes: pd.Series
    skips: list[tuple[pd.Timestamp, str, str, str]] = field(default_factory=list)

    def trade_frame(self) -> pd.DataFrame:
        rows = []
        for p in self.trades:
            rows.append(
                {
                    "id": p.id,
                    "symbol": p.symbol,
                    "sleeve": p.sleeve,
                    "structure": p.structure,
                    "entry_date": p.entry_date,
                    "exit_date": p.exit_date,
                    "contracts": p.contracts,
                    "strikes": "/".join(f"{leg.strike:g}{leg.kind}" for leg in p.legs),
                    "expiration": p.expiration,
                    "entry_price": round(p.entry_price, 4),
                    "exit_price": None if p.exit_price is None else round(p.exit_price, 4),
                    "max_loss": round(p.risk_dollars, 2),
                    "pnl": None if p.exit_price is None else round(p.pnl(p.exit_price), 2),
                    "exit_reason": p.exit_reason,
                }
            )
        return pd.DataFrame(rows)

    def stats(self) -> dict[str, float]:
        closed = [p.pnl(p.exit_price) for p in self.trades if p.exit_price is not None and p.sleeve != "hedge"]
        return summarize(self.equity, closed)


def run_backtest(
    market: MarketData,
    cfg: Config,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> BacktestResult:
    planner = Planner(market, cfg)
    dates = market.dates
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]

    cash = cfg.account.starting_capital
    peak = cash
    book: list[Position] = []
    closed: list[Position] = []
    hedge_budget = 0.0
    equity_rows, regime_rows, skips = {}, {}, []
    next_id = 1
    last_week = None

    for date in dates:
        week = date.isocalendar()[:2]
        is_entry_day = week != last_week and date.weekday() >= cfg.entry.entry_weekday
        if is_entry_day:
            last_week = week

        equity = cash + sum(planner.mark(date, p) * MULT * p.contracts for p in book)
        peak = max(peak, equity)
        halted = selling_halted(equity, peak, market.vix(), market.vix3m(), date, cfg.risk)

        state = BookState(cash, book, equity, peak, hedge_budget, halted)
        plan = planner.plan(date, state, entries=is_entry_day)
        regime_rows[date] = plan.regime.color

        for ex in plan.exits:
            pos = ex.position
            commission = (
                0.0 if ex.reason == "expired" else cfg.account.commission_per_contract * len(pos.legs) * pos.contracts
            )
            cash += ex.value * MULT * pos.contracts - commission
            pos.commissions += commission
            pos.exit_date, pos.exit_price, pos.exit_reason = date, ex.value, ex.reason
            book.remove(pos)
            closed.append(pos)

        for prop in plan.proposals:
            pos = prop.position
            pos.id = next_id
            next_id += 1
            cash -= pos.entry_price * MULT * pos.contracts + pos.commissions
            if pos.sleeve == "hedge":
                hedge_budget -= pos.entry_price * MULT * pos.contracts + pos.commissions
            elif pos.is_short_premium:
                hedge_budget += cfg.hedge.premium_share * pos.credit * MULT * pos.contracts
            book.append(pos)
        skips += [(date, s.symbol, s.sleeve, s.reason) for s in plan.skips]

        equity_rows[date] = cash + sum(planner.mark(date, p) * MULT * p.contracts for p in book)

    if len(dates):
        for pos in book:  # leave open trades marked at mid
            pos.exit_reason = "open at end"
        closed += book

    return BacktestResult(pd.Series(equity_rows, dtype=float), closed, pd.Series(regime_rows), skips)


def simple_putwrite(market: MarketData, symbol: str = "SPY", start=None, end=None, capital: float = 1.0) -> pd.Series:
    """Benchmark in the style of the Cboe PUT index, priced on the same data.

    Each month sells one at-the-money put about 30 days out, fully
    cash-secured, holds to expiry, then repeats. Uses Black-Scholes at the
    source's ATM IV so it works on any `MarketData`, including synthetic.
    """
    closes = market.closes(symbol)
    dates = market.dates
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    equity = {}
    value = capital
    strike = expiry = premium = None
    base = value
    for date in dates:
        spot = float(closes.loc[:date].iloc[-1])
        if expiry is not None and date >= expiry:
            value = base * (1 + (premium - max(strike - spot, 0.0)) / strike)
            expiry = None
        if expiry is None:
            iv = market.atm_iv(date, symbol) or 0.2
            expiry = date + pd.Timedelta(days=30)
            strike, premium, base = spot, bs_price(spot, spot, 30 / 365, iv, "P"), value
        t = max((expiry - date).days, 0) / 365
        iv = market.atm_iv(date, symbol) or 0.2
        open_value = bs_price(spot, strike, t, iv, "P")
        equity[date] = base * (1 + (premium - open_value) / strike)
    return pd.Series(equity, dtype=float)
