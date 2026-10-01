"""Turns signals into exits and new trades for one day.

The backtester calls this every day over history; the scanner calls it once
for today. Sharing it means the trades you paper trade are exactly the
trades that were backtested.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import signals as sig
from . import trades as tr
from .config import Config
from .data.base import MarketData
from .pricing import implied_vol
from .risk import check_book, contracts_for, drawdown_multiplier
from .trades import MULT, Leg, Position


@dataclass
class Proposal:
    position: Position
    why: str


@dataclass
class Skip:
    symbol: str
    sleeve: str
    reason: str


@dataclass
class Exit:
    position: Position
    reason: str
    value: float  # per-share fill value


@dataclass
class DayPlan:
    date: pd.Timestamp
    regime: sig.Regime
    exits: list[Exit] = field(default_factory=list)
    proposals: list[Proposal] = field(default_factory=list)
    skips: list[Skip] = field(default_factory=list)
    signals: dict[str, dict] = field(default_factory=dict)


@dataclass
class BookState:
    """What the planner needs to know about the account."""

    cash: float
    positions: list[Position]
    equity: float
    peak_equity: float
    hedge_budget: float = 0.0
    selling_halted: bool = False


class Planner:
    def __init__(self, market: MarketData, cfg: Config):
        self.m = market
        self.cfg = cfg
        self.iv_hist: dict[str, dict[pd.Timestamp, float]] = {}

    # ------------------------------------------------------------ inputs

    def index_symbol(self) -> str:
        return self.cfg.universe.index[0]

    def iv_series(self, symbol: str) -> pd.Series:
        if symbol == "SPY":
            return self.m.vix() / 100.0
        h = self.iv_hist.get(symbol, {})
        return pd.Series(h, dtype=float).sort_index()

    def record_iv(self, date: pd.Timestamp, symbol: str) -> float | None:
        if symbol == "SPY":
            v = self.m.vix().loc[:date]
            return float(v.iloc[-1]) / 100 if not v.empty else None
        iv = self.m.atm_iv(date, symbol)
        if iv is not None:
            self.iv_hist.setdefault(symbol, {})[date] = iv
        return iv

    def spot(self, date: pd.Timestamp, symbol: str) -> float:
        return float(self.m.closes(symbol).loc[:date].iloc[-1])

    # ------------------------------------------------------------ marking

    def mark(self, date: pd.Timestamp, pos: Position) -> float:
        """Refresh each leg's quote and return the position's mid value per share."""
        spot = self.spot(date, pos.symbol)
        for leg in pos.legs:
            if date >= leg.expiration:
                intrinsic = max(spot - leg.strike, 0.0) if leg.kind == "C" else max(leg.strike - spot, 0.0)
                leg.bid = leg.ask = intrinsic
                continue
            q = self.m.quote(date, pos.symbol, leg.expiration, leg.strike, leg.kind)
            if q is None:
                continue  # keep the last quote
            leg.bid, leg.ask = q
            iv = implied_vol(leg.mid, spot, leg.strike, (leg.expiration - date).days / 365, leg.kind)
            if iv:
                leg.iv = iv
        return tr.mid_value(pos.legs)

    def exit_reason(self, date: pd.Timestamp, pos: Position, value: float) -> str | None:
        ex = self.cfg.exits
        if date >= pos.expiration:
            return "expired"
        if pos.sleeve == "hedge":
            return "hedge roll" if pos.dte(date) <= self.cfg.hedge.roll_dte else None
        if pos.is_short_premium:
            cost_to_close = -value
            credit = pos.credit
            if cost_to_close <= credit * (1 - ex.short_profit_target):
                return f"profit target {ex.short_profit_target:.0%}"
            if cost_to_close - credit >= ex.short_stop_multiple * credit:
                return f"stop at {ex.short_stop_multiple:g}x credit"
            if pos.dte(date) <= ex.short_exit_dte:
                return f"{ex.short_exit_dte} DTE"
            return None
        gain = value / pos.entry_price - 1 if pos.entry_price > 0 else 0.0
        if gain >= ex.long_profit_target:
            return f"profit +{ex.long_profit_target:.0%}"
        if gain <= -ex.long_stop:
            return f"stop -{ex.long_stop:.0%}"
        if pos.dte(date) <= ex.long_exit_dte:
            return f"{ex.long_exit_dte} DTE"
        return None

    def exits(self, date: pd.Timestamp, book: list[Position]) -> list[Exit]:
        out = []
        for pos in book:
            value = self.mark(date, pos)
            reason = self.exit_reason(date, pos, value)
            if reason:
                fill = value if reason == "expired" else tr.fill_price(pos.legs, opening=False, fills=self.cfg.fills)
                out.append(Exit(pos, reason, fill))
        return out

    # ------------------------------------------------------------ entries

    def _build(self, date, symbol, sleeve, built, why) -> Position | Skip:
        if isinstance(built, str):
            return Skip(symbol, sleeve, built)
        legs, structure, width = built
        entry = tr.fill_price(legs, opening=True, fills=self.cfg.fills)
        if structure in ("put_spread", "call_spread", "iron_condor", "csp") and entry >= 0:
            return Skip(symbol, sleeve, "no credit after costs")
        max_loss = tr.max_loss_per_contract(structure, legs, entry, width)
        pos = Position(symbol, structure, sleeve, legs, 0, date, entry, max_loss)
        pos.notes["why"] = why
        return pos

    def _size_and_check(self, date, pos: Position, budget: float, state: BookState, book: list[Position]):
        n = contracts_for(pos.max_loss, budget)
        if n < 1:
            return Skip(pos.symbol, pos.sleeve, f"one contract risks ${pos.max_loss:,.0f}, budget ${budget:,.0f}")
        reserved = sum(p.risk_dollars for p in book if p.is_short_premium)
        free_cash = state.cash - reserved
        spots = {s: self.spot(date, s) for s in {p.symbol for p in book} | {pos.symbol}}
        reason = ""
        while n >= 1:
            pos.contracts = n
            need = pos.max_loss * n if pos.is_short_premium else pos.entry_price * MULT * n
            need += self._commission(pos)
            if need > free_cash:
                reason = f"needs ${need:,.0f} cash, ${free_cash:,.0f} free"
            else:
                chk = check_book(book + [pos], spots, date, state.equity, self.cfg.risk)
                if chk.ok:
                    return pos
                reason = chk.reason
            n -= 1
        return Skip(pos.symbol, pos.sleeve, reason)

    def _commission(self, pos: Position) -> float:
        return self.cfg.account.commission_per_contract * len(pos.legs) * pos.contracts

    def _sleeve_room(
        self, sleeve: str, share: float, state: BookState, book: list[Position], scale: float = 1.0
    ) -> float:
        """Max-loss budget left in a sleeve. `scale` shrinks it in yellow regimes and drawdowns."""
        used = sum(p.risk_dollars for p in book if p.sleeve == sleeve)
        return share * self.cfg.risk.portfolio_max_loss_pct * state.equity * scale - used

    def plan(self, date: pd.Timestamp, state: BookState, entries: bool = True) -> DayPlan:
        cfg = self.cfg
        idx = self.index_symbol()
        reg = sig.regime(date, self.m.closes(idx), self.m.vix(), self.m.vix3m(), cfg.regime)
        plan = DayPlan(date, reg)
        plan.exits = self.exits(date, state.positions)
        if not entries:
            return plan

        closing = {id(e.position) for e in plan.exits}
        book = [p for p in state.positions if id(p) not in closing]
        dd_mult = drawdown_multiplier(state.equity, state.peak_equity, cfg.risk)
        selling_ok = not state.selling_halted
        size = reg.size * dd_mult
        # The regime and drawdown multipliers shrink each sleeve's total budget,
        # not the per-trade cap: at $5,000, half of a 2% cap ($50) is less than
        # one $1-wide spread risks, so halving per trade would mean no trades.
        per_trade = cfg.risk.per_trade_max_loss_pct * state.equity

        info: dict[str, dict] = {}
        for sym in cfg.universe.index + cfg.universe.stocks:
            iv = self.record_iv(date, sym)
            if iv is None:
                continue
            closes = self.m.closes(sym)
            ivs = self.iv_series(sym)
            info[sym] = {
                "iv": iv,
                "ivr": sig.iv_rank(ivs, date),
                "iv_rv": sig.iv_minus_rv(iv, closes, date),
                "trend": sig.trend(closes, date),
                "mom": sig.momentum_12_1(closes, date),
                "straddle_mom": sig.straddle_momentum(ivs, closes, date),
                "price": self.spot(date, sym),
            }
        plan.signals = info

        def add(result, budget):
            if isinstance(result, Skip):
                plan.skips.append(result)
                return
            sized = self._size_and_check(date, result, budget, state, book)
            if isinstance(sized, Skip):
                plan.skips.append(sized)
                return
            sized.commissions = self._commission(sized)
            book.append(sized)
            plan.proposals.append(Proposal(sized, sized.notes.get("why", "")))

        open_syms = {(p.symbol, p.sleeve) for p in book}

        # Sleeve 1: index premium, one new rung per symbol per week.
        for sym in cfg.universe.index:
            s = info.get(sym)
            if s is None:
                continue
            if size <= 0 or not selling_ok:
                why = f"regime {reg.color}" if size <= 0 else "drawdown breaker"
                plan.skips.append(Skip(sym, "index", f"no new short premium ({why})"))
                continue
            if s["ivr"] is None or s["ivr"] < cfg.entry.ivr_sell_min:
                ivr = "n/a (not enough history)" if s["ivr"] is None else f"{s['ivr']:.0f}"
                plan.skips.append(Skip(sym, "index", f"IV Rank {ivr} under {cfg.entry.ivr_sell_min:g}"))
                continue
            built = self._short_premium(date, sym, s["trend"])
            why = f"regime {reg.color}, IV Rank {s['ivr']:.0f}, trend {s['trend']}"
            room = self._sleeve_room("index", cfg.risk.sleeve_index, state, book, size)
            add(self._build(date, sym, "index", built, why), min(per_trade, room))

        # Sleeve 2: single-stock premium, richest names first.
        if size > 0 and selling_ok:
            ranked = [
                (sym, s)
                for sym, s in info.items()
                if sym in cfg.universe.stocks
                and (sym, "single_stock") not in open_syms
                and s["ivr"] is not None
                and s["ivr"] >= cfg.entry.ivr_sell_min
                and (s["iv_rv"] or 0) > 0
                and s["trend"] != "bearish"
                and (s["straddle_mom"] is None or s["straddle_mom"] <= 0)
                and s["price"] >= cfg.universe.min_price
            ]
            ranked.sort(key=lambda kv: (kv[1]["ivr"], kv[1]["iv_rv"]), reverse=True)
            for sym, s in ranked:
                room = self._sleeve_room("single_stock", cfg.risk.sleeve_single_stock, state, book, size)
                if room < 1:
                    break
                built = self._short_premium(date, sym, "bullish")
                why = f"IV Rank {s['ivr']:.0f}, IV-RV {s['iv_rv']:+.1f} pts"
                add(self._build(date, sym, "single_stock", built, why), min(per_trade, room))

        # Sleeve 3: directional buying, only when options are cheap.
        long_budget = cfg.risk.long_premium_max_pct * state.equity
        for sym, s in info.items():
            if (sym, "directional") in open_syms or s["ivr"] is None or s["mom"] is None:
                continue
            cheap = s["ivr"] <= cfg.entry.ivr_buy_max and (s["iv_rv"] or 0) < 0
            if not cheap:
                continue
            if s["trend"] == "bullish" and s["mom"] > 0 and reg.color != sig.RED:
                kind, structure = "C", "long_call"
            elif s["trend"] == "bearish" and s["mom"] < 0:
                kind, structure = "P", "long_put"
            else:
                continue
            chain = self.m.chain(date, sym, cfg.entry.long_dte_min, cfg.entry.long_dte_max)
            built = tr.long_option(
                chain,
                date,
                kind,
                cfg.entry,
                cfg.entry.long_delta,
                cfg.entry.long_dte_min,
                cfg.entry.long_dte_max,
                structure,
            )
            why = f"IV Rank {s['ivr']:.0f}, IV-RV {s['iv_rv']:+.1f}, trend {s['trend']}, 12-1 mom {s['mom']:+.0%}"
            room = self._sleeve_room("directional", cfg.risk.sleeve_directional, state, book, dd_mult)
            add(self._build(date, sym, "directional", built, why), min(long_budget, room))

        # Sleeve 4: tail hedge, paid from a share of premium collected.
        h = cfg.hedge
        if h.enabled and not any(p.sleeve == "hedge" for p in book) and h.symbol in info:
            chain = self.m.chain(date, h.symbol, h.dte_min, h.dte_max)
            built = tr.long_option(chain, date, "P", cfg.entry, h.delta, h.dte_min, h.dte_max, "hedge_put")
            pos = self._build(date, h.symbol, "hedge", built, "tail hedge")
            if isinstance(pos, Skip):
                plan.skips.append(pos)
            else:
                budget = state.hedge_budget
                n = contracts_for(pos.entry_price * MULT + cfg.account.commission_per_contract, budget)
                if n >= 1:
                    pos.contracts = n
                    pos.commissions = self._commission(pos)
                    book.append(pos)
                    plan.proposals.append(Proposal(pos, f"tail hedge from ${budget:,.0f} saved premium"))
                else:
                    cost = pos.entry_price * MULT
                    plan.skips.append(Skip(h.symbol, "hedge", f"saving up: ${budget:,.0f} of ${cost:,.0f} for one put"))
        return plan

    def _short_premium(self, date, sym, direction):
        e = self.cfg.entry
        chain = self.m.chain(date, sym, e.short_dte_min, e.short_dte_max)
        if not self.cfg.account.spreads_allowed:
            if direction == "bearish":
                return "bearish and spreads not allowed"
            return tr.cash_secured_put(chain, date, e)
        if direction == "bullish":
            return tr.vertical_spread(chain, date, "P", e, e.put_spread_delta)
        if direction == "bearish":
            return tr.vertical_spread(chain, date, "C", e, e.call_spread_delta)
        return tr.iron_condor(chain, date, e)


def leg_summary(leg: Leg) -> str:
    side = "Buy" if leg.qty > 0 else "Sell"
    return f"{side} {leg.expiration:%Y-%m-%d} {leg.strike:g}{leg.kind}"
