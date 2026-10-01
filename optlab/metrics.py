"""Performance statistics the playbook's validation gate asks for."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def summarize(equity: pd.Series, trade_pnls: list[float] | None = None) -> dict[str, float]:
    equity = equity.dropna()
    out: dict[str, float] = {}
    if len(equity) < 2:
        return out
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    rets = equity.pct_change().dropna()
    out["final_equity"] = float(equity.iloc[-1])
    out["total_return"] = float(equity.iloc[-1] / equity.iloc[0] - 1)
    out["cagr"] = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else float("nan")
    out["volatility"] = float(rets.std() * math.sqrt(252))
    out["sharpe"] = float(rets.mean() / rets.std() * math.sqrt(252)) if rets.std() > 0 else float("nan")
    downside = rets[rets < 0]
    dstd = math.sqrt(float((downside**2).sum()) / len(rets)) if len(rets) else 0.0
    out["sortino"] = float(rets.mean() / dstd * math.sqrt(252)) if dstd > 0 else float("nan")
    out["max_drawdown"] = float((equity / equity.cummax() - 1).min())
    monthly = equity.resample("ME").last().pct_change().dropna()
    out["worst_month"] = float(monthly.min()) if len(monthly) else float("nan")
    if trade_pnls:
        pnl = np.array(trade_pnls)
        wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
        out["trades"] = float(len(pnl))
        out["win_rate"] = float(len(wins) / len(pnl))
        out["avg_win"] = float(wins.mean()) if len(wins) else 0.0
        out["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
    return out


def format_summary(name: str, stats: dict[str, float]) -> str:
    pct = {"total_return", "cagr", "volatility", "max_drawdown", "worst_month", "win_rate"}
    money = {"final_equity", "avg_win", "avg_loss"}
    lines = [name]
    for k, v in stats.items():
        if k in pct:
            lines.append(f"  {k:<14} {v:8.1%}")
        elif k in money:
            lines.append(f"  {k:<14} {v:10,.2f}")
        else:
            lines.append(f"  {k:<14} {v:8.2f}")
    return "\n".join(lines)
