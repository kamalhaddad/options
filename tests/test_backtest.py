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
