"""Strategy and account settings.

Every number here comes from the Options Trading Strategy Playbook. Defaults
match a $5,000 cash account; override any of them from a TOML file
(see config/default.toml) instead of editing code.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class AccountConfig:
    starting_capital: float = 5_000.0
    account_type: str = "cash"  # "cash" or "margin"
    # Many brokers do not allow spreads in a cash account. When False, short
    # premium can only be cash-secured puts, which rarely fit a small account.
    spreads_allowed: bool = False
    commission_per_contract: float = 0.65


@dataclass
class RegimeConfig:
    term_green_below: float = 0.90  # VIX / VIX3M
    term_red_above: float = 1.00
    vix_green_low: float = 15.0
    vix_green_high: float = 30.0
    vix_thin_below: float = 13.0
    vix_red_above: float = 40.0
    trend_ma_days: int = 200
    trend_band: float = 0.02  # within 2% of the 200-day is yellow
    rv_days: int = 20
    rv_gap_green: float = 3.0  # VIX at least 3 points above realized vol
    size_green: float = 1.0
    size_yellow: float = 0.5
    size_red: float = 0.0


@dataclass
class EntryConfig:
    entry_weekday: int = 0  # Monday
    short_dte_min: int = 30
    short_dte_max: int = 45
    long_dte_min: int = 60
    long_dte_max: int = 120
    put_spread_delta: float = 0.30
    condor_delta: float = 0.16
    call_spread_delta: float = 0.25
    wing_width: float = 1.0  # dollars between short and long strike
    min_credit_to_width_spread: float = 0.33
    min_credit_to_width_condor: float = 0.25
    ivr_sell_min: float = 30.0
    ivr_buy_max: float = 25.0
    long_delta: float = 0.65
    max_spread_pct_of_mid: float = 0.10  # option momentum rule: skip wider quotes


@dataclass
class ExitConfig:
    short_profit_target: float = 0.50  # of credit received
    short_stop_multiple: float = 2.0  # loss of 2x credit
    short_exit_dte: int = 21
    long_profit_target: float = 1.00  # +100% on premium
    long_stop: float = 0.50  # -50% of premium
    long_exit_dte: int = 30


@dataclass
class RiskConfig:
    per_trade_max_loss_pct: float = 0.02
    long_premium_max_pct: float = 0.015
    portfolio_max_loss_pct: float = 0.25
    stress_spot_move: float = -0.10
    stress_vol_multiplier: float = 2.0
    stress_max_loss_pct: float = 0.15
    vega_shock_points: float = 10.0
    vega_max_loss_pct: float = 0.08
    drawdown_half_size: float = 0.10
    drawdown_stop_selling: float = 0.15
    sleeve_index: float = 0.50
    sleeve_single_stock: float = 0.20
    sleeve_directional: float = 0.20
    sleeve_hedge: float = 0.10
    # Cash-secured puts. Their max loss is owning the stock, which no small
    # account fits under the 2% and 25% max-loss caps. With "stress", a CSP is
    # sized by its loss in the stress test instead, and left out of the
    # combined max-loss cap; the book-wide stress cap still applies.
    csp_risk_basis: str = "max_loss"  # "max_loss" or "stress"
    csp_stress_max_pct: float = 0.05  # per position, stress-test loss
    csp_max_collateral_pct: float = 0.50  # per position, cash tied up
    csp_max_positions: int = 2


@dataclass
class HedgeConfig:
    enabled: bool = True
    premium_share: float = 0.10  # share of collected premium spent on hedges
    delta: float = 0.05
    dte_min: int = 60
    dte_max: int = 90
    roll_dte: int = 30
    symbol: str = "SPY"


@dataclass
class FillConfig:
    # Fill at mid minus this share of the spread when selling, plus it when buying.
    spread_share: float = 0.25


@dataclass
class UniverseConfig:
    index: list[str] = field(default_factory=lambda: ["SPY"])
    stocks: list[str] = field(default_factory=list)
    min_price: float = 20.0
    # Before a name has 20 days of IV history, call its options rich when
    # implied vol is at least this many points above 20-day realized vol.
    iv_rv_rich_fallback: float = 3.0


@dataclass
class Config:
    account: AccountConfig = field(default_factory=AccountConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    entry: EntryConfig = field(default_factory=EntryConfig)
    exits: ExitConfig = field(default_factory=ExitConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    hedge: HedgeConfig = field(default_factory=HedgeConfig)
    fills: FillConfig = field(default_factory=FillConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)


def _apply(obj: Any, values: dict[str, Any], path: str = "") -> None:
    known = {f.name: f for f in fields(obj)}
    for key, value in values.items():
        if key not in known:
            raise KeyError(f"Unknown setting '{path}{key}'")
        current = getattr(obj, key)
        if is_dataclass(current):
            _apply(current, value, f"{path}{key}.")
        else:
            setattr(obj, key, value)


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path:
        with open(path, "rb") as fh:
            _apply(cfg, tomllib.load(fh))
    return cfg
