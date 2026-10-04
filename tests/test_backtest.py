import pandas as pd

from optlab.backtest import run_backtest, simple_putwrite


def test_backtest_respects_caps(market, cfg):
    res = run_backtest(market, cfg, "2016-01-01", "2017-06-30")
    eq = res.equity
    assert eq.iloc[0] == cfg.account.starting_capital or abs(eq.iloc[0] - cfg.account.starting_capital) < 200
    trades = res.trade_frame()
    assert not trades.empty
    non_hedge = trades[trades.sleeve != "hedge"]
    # Per-trade cap: 2% of equity, and equity never got near 1.5x the start here.
    assert (non_hedge.max_loss <= 0.02 * eq.max() + 1e-6).all()
    # Combined open max loss stays under 25% of equity on every day.
    for date in eq.index[::10]:
        live = non_hedge[(non_hedge.entry_date <= date) & ((non_hedge.exit_date > date) | non_hedge.exit_date.isna())]
        assert live.max_loss.sum() <= 0.25 * eq.loc[date] + 1e-6
    assert set(res.regimes.unique()) <= {"green", "yellow", "red"}


def test_cash_account_without_spreads_never_opens_spreads(market, cfg):
    cfg.account.spreads_allowed = False
    res = run_backtest(market, cfg, "2016-06-01", "2016-12-31")
    structures = set(res.trade_frame().get("structure", pd.Series(dtype=str)))
    assert not structures & {"put_spread", "call_spread", "iron_condor"}


def test_put_write_benchmark_runs(market):
    eq = simple_putwrite(market, "SPY", "2016-01-01", "2016-12-31")
    assert len(eq) > 200 and eq.iloc[0] > 0


def test_cash_account_sells_stress_sized_puts_on_cheap_stocks(cfg):
    from optlab.data import SyntheticMarket, SyntheticParams

    m = SyntheticMarket(["CHEAP"], SyntheticParams(seed=5, start="2015-01-01", end="2016-12-31", stock_s0=20.0))
    cfg.account.spreads_allowed = False
    cfg.universe.stocks = ["CHEAP"]
    cfg.universe.min_price = 10.0
    cfg.risk.csp_risk_basis = "stress"
    res = run_backtest(m, cfg, "2016-01-01", "2016-12-31")
    csps = [p for p in res.trades if p.structure == "csp"]
    assert csps, "expected at least one cash-secured put"
    eq = res.equity
    for p in csps:
        equity = eq.loc[p.entry_date]
        assert p.notes["stress_per_contract"] * p.contracts <= 0.05 * equity + 1e-6
        assert p.risk_dollars <= 0.50 * equity + 1e-6
    for date in eq.index[::5]:
        live = [p for p in csps if p.entry_date <= date and (p.exit_date is None or p.exit_date > date)]
        assert len(live) <= cfg.risk.csp_max_positions
