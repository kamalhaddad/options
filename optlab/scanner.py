"""Daily scanner: today's regime, exits due, and new trade suggestions.

It never places orders. `scan` only reports; `paper` also records the
suggestions in a paper journal at the modeled fill price.
"""

from __future__ import annotations

import pandas as pd

from .config import Config
from .data.base import MarketData
from .journal import Journal
from .planner import BookState, DayPlan, Planner, leg_summary
from .risk import selling_halted
from .trades import MULT


def run_day(market: MarketData, cfg: Config, journal: Journal, record: bool) -> DayPlan:
    """Plan today for the journal's book; when `record`, apply it to the journal."""
    date = market.dates[-1]
    planner = Planner(market, cfg)
    planner.iv_hist = {s: {pd.Timestamp(d): v for d, v in h.items()} for s, h in journal.iv_history.items()}

    equity = journal.cash + sum(planner.mark(date, p) * MULT * p.contracts for p in journal.open)
    peak = max(journal.peak_equity, equity)
    year, wk = date.isocalendar()[:2]
    week = f"{year}-{wk:02d}"
    entry_day = week != journal.last_entry_week and date.weekday() >= cfg.entry.entry_weekday
    halted = selling_halted(equity, peak, market.vix(), market.vix3m(), date, cfg.risk)

    state = BookState(journal.cash, journal.open, equity, peak, journal.hedge_budget, halted)
    plan = planner.plan(date, state, entries=entry_day)

    if record:
        for ex in plan.exits:
            pos = ex.position
            fee = 0.0 if ex.reason == "expired" else cfg.account.commission_per_contract * len(pos.legs) * pos.contracts
            journal.cash += ex.value * MULT * pos.contracts - fee
            pos.commissions += fee
            pos.exit_date, pos.exit_price, pos.exit_reason = date, ex.value, ex.reason
            journal.open.remove(pos)
            journal.closed.append(pos)
        for prop in plan.proposals:
            pos = prop.position
            pos.id = journal.next_id
            journal.next_id += 1
            journal.cash -= pos.entry_price * MULT * pos.contracts + pos.commissions
            if pos.sleeve == "hedge":
                journal.hedge_budget -= pos.entry_price * MULT * pos.contracts + pos.commissions
            elif pos.is_short_premium:
                journal.hedge_budget += cfg.hedge.premium_share * pos.credit * MULT * pos.contracts
            journal.open.append(pos)
        if entry_day:
            journal.last_entry_week = week
        new_equity = journal.cash + sum(planner.mark(date, p) * MULT * p.contracts for p in journal.open)
        journal.peak_equity = max(peak, new_equity)
        journal.selling_halted = halted
        journal.equity_log[date.strftime("%Y-%m-%d")] = round(new_equity, 2)
        journal.iv_history = {s: {d.strftime("%Y-%m-%d"): v for d, v in h.items()} for s, h in planner.iv_hist.items()}
    plan.signals["_account"] = {"equity": equity, "peak": peak, "entry_day": entry_day}
    return plan


def _money(x: float) -> str:
    return f"-${-x:,.2f}" if x < 0 else f"${x:,.2f}"


def report(plan: DayPlan, journal: Journal, recorded: bool) -> str:
    acct = plan.signals.get("_account", {})
    r = plan.regime
    lines = [f"# Options scan for {plan.date:%A %Y-%m-%d}", ""]
    lines.append("Suggestions only. Nothing here places an order.")
    lines.append("")
    lines.append(f"**Regime: {r.color.upper()}** (size multiplier {r.size:g})")
    vals = r.values
    if vals:
        parts = []
        if "vix" in vals:
            parts.append(f"VIX {vals['vix']:.1f}")
        if "vix_vix3m" in vals:
            parts.append(f"VIX/VIX3M {vals['vix_vix3m']:.2f}")
        if "spy_vs_200d" in vals:
            parts.append(f"SPY vs 200-day {vals['spy_vs_200d']:+.1%}")
        if "vix_minus_rv" in vals:
            parts.append(f"VIX minus realized {vals['vix_minus_rv']:+.1f} pts")
        lines.append(", ".join(parts))
    lines.append("Checks: " + ", ".join(f"{k} {v}" for k, v in r.checks.items()))
    lines.append("")
    if acct:
        lines.append(f"Account equity {_money(acct['equity'])}, peak {_money(acct['peak'])}")
        if not acct.get("entry_day"):
            lines.append("Not an entry day (new trades open once a week); checking exits only.")
        lines.append("")

    lines.append("## Close these")
    if not plan.exits:
        lines.append("Nothing to close.")
    for ex in plan.exits:
        p = ex.position
        lines.append(
            f"- #{p.id} {p.symbol} {p.structure} x{p.contracts}: {ex.reason}, "
            f"close near {ex.value:+.2f}/share, P&L about {_money(p.pnl(ex.value))}"
        )
    lines.append("")

    lines.append("## New trades")
    if not plan.proposals:
        lines.append("None today.")
    for prop in plan.proposals:
        p = prop.position
        legs = "; ".join(leg_summary(leg) for leg in p.legs)
        kind = "credit" if p.is_short_premium else "debit"
        lines.append(
            f"- {p.symbol} {p.structure} x{p.contracts} ({p.sleeve}): {legs}. "
            f"Limit {kind} {abs(p.entry_price):.2f}, max loss {_money(p.risk_dollars)}. {prop.why}"
        )
    lines.append("")

    if plan.skips:
        lines.append("## Skipped")
        for s in plan.skips:
            lines.append(f"- {s.symbol} ({s.sleeve}): {s.reason}")
        lines.append("")

    sigs = {k: v for k, v in plan.signals.items() if not k.startswith("_")}
    if sigs:
        lines.append("## Signals")
        lines.append("| Symbol | Price | IV | IV Rank | IV-RV | Trend | 12-1 mom |")
        lines.append("|---|---|---|---|---|---|---|")
        for sym, s in sigs.items():
            ivr = "n/a" if s["ivr"] is None else f"{s['ivr']:.0f}"
            ivrv = "n/a" if s["iv_rv"] is None else f"{s['iv_rv']:+.1f}"
            mom = "n/a" if s["mom"] is None else f"{s['mom']:+.0%}"
            lines.append(f"| {sym} | {s['price']:.2f} | {s['iv']:.1%} | {ivr} | {ivrv} | {s['trend']} | {mom} |")
        lines.append("")

    open_now = journal.open if recorded else [p for p in journal.open if all(p is not e.position for e in plan.exits)]
    if open_now:
        lines.append("## Open paper positions")
        for p in open_now:
            lines.append(f"- #{p.id} {p.symbol} {p.structure} x{p.contracts}, expires {p.expiration:%Y-%m-%d}")
        lines.append("")
    return "\n".join(lines)
