"""Paper-trading journal: a JSON file holding the paper account.

The scanner never places orders. In paper mode it records its own
suggestions here at the modeled fill price, so after a few weeks you can
compare paper results and slippage with the backtest before risking money.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .trades import Leg, Position


def _leg_to_dict(leg: Leg) -> dict:
    return {
        "expiration": leg.expiration.strftime("%Y-%m-%d"),
        "strike": leg.strike,
        "kind": leg.kind,
        "qty": leg.qty,
        "bid": leg.bid,
        "ask": leg.ask,
        "delta": leg.delta,
        "iv": leg.iv,
    }


def position_to_dict(p: Position) -> dict:
    return {
        "id": p.id,
        "symbol": p.symbol,
        "structure": p.structure,
        "sleeve": p.sleeve,
        "legs": [_leg_to_dict(leg) for leg in p.legs],
        "contracts": p.contracts,
        "entry_date": p.entry_date.strftime("%Y-%m-%d"),
        "entry_price": p.entry_price,
        "max_loss": p.max_loss,
        "exit_date": None if p.exit_date is None else p.exit_date.strftime("%Y-%m-%d"),
        "exit_price": p.exit_price,
        "exit_reason": p.exit_reason,
        "commissions": p.commissions,
        "notes": p.notes,
    }


def position_from_dict(d: dict) -> Position:
    legs = [
        Leg(pd.Timestamp(x["expiration"]), x["strike"], x["kind"], x["qty"], x["bid"], x["ask"], x["delta"], x["iv"])
        for x in d["legs"]
    ]
    return Position(
        symbol=d["symbol"],
        structure=d["structure"],
        sleeve=d["sleeve"],
        legs=legs,
        contracts=d["contracts"],
        entry_date=pd.Timestamp(d["entry_date"]),
        entry_price=d["entry_price"],
        max_loss=d["max_loss"],
        id=d["id"],
        exit_date=None if d["exit_date"] is None else pd.Timestamp(d["exit_date"]),
        exit_price=d["exit_price"],
        exit_reason=d["exit_reason"],
        commissions=d["commissions"],
        notes=d.get("notes", {}),
    )


@dataclass
class Journal:
    cash: float
    peak_equity: float
    hedge_budget: float = 0.0
    selling_halted: bool = False
    next_id: int = 1
    last_entry_week: str = ""
    open: list[Position] = field(default_factory=list)
    closed: list[Position] = field(default_factory=list)
    equity_log: dict[str, float] = field(default_factory=dict)
    iv_history: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def new(cls, capital: float) -> Journal:
        return cls(cash=capital, peak_equity=capital)

    @classmethod
    def load(cls, path: str | Path, capital: float) -> Journal:
        path = Path(path)
        if not path.exists():
            return cls.new(capital)
        d = json.loads(path.read_text())
        return cls(
            cash=d["cash"],
            peak_equity=d["peak_equity"],
            hedge_budget=d.get("hedge_budget", 0.0),
            selling_halted=d.get("selling_halted", False),
            next_id=d.get("next_id", 1),
            last_entry_week=d.get("last_entry_week", ""),
            open=[position_from_dict(x) for x in d.get("open", [])],
            closed=[position_from_dict(x) for x in d.get("closed", [])],
            equity_log=d.get("equity_log", {}),
            iv_history=d.get("iv_history", {}),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {
            "cash": self.cash,
            "peak_equity": self.peak_equity,
            "hedge_budget": self.hedge_budget,
            "selling_halted": self.selling_halted,
            "next_id": self.next_id,
            "last_entry_week": self.last_entry_week,
            "open": [position_to_dict(p) for p in self.open],
            "closed": [position_to_dict(p) for p in self.closed],
            "equity_log": self.equity_log,
            "iv_history": self.iv_history,
        }
        path.write_text(json.dumps(d, indent=2))
