"""Command line: `optlab backtest`, `optlab scan` and `optlab paper`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .backtest import run_backtest, simple_putwrite
from .config import Config, load_config
from .data import CsvMarket, SyntheticMarket, SyntheticParams, read_cboe_index
from .data.base import MarketData
from .journal import Journal
from .metrics import format_summary, summarize
from .scanner import report, run_day


def _market(args, cfg: Config) -> MarketData:
    if args.source == "synthetic":
        params = SyntheticParams(seed=args.seed)
        if getattr(args, "start", None):
            params.start = str(pd.Timestamp(args.start) - pd.Timedelta(days=400))[:10]
        if getattr(args, "end", None):
            params.end = args.end
        return SyntheticMarket(cfg.universe.stocks, params)
    if args.source == "csv":
        if not (args.chains and args.vix and args.vix3m):
            sys.exit("--source csv needs --chains, --vix and --vix3m")
        return CsvMarket.from_files(args.chains, args.vix, args.vix3m)
    if args.source == "yahoo":
        from .data.yahoo import YahooMarket

        return YahooMarket(cfg.universe.index + cfg.universe.stocks)
    sys.exit(f"unknown source {args.source}")


def gate(strategy: dict, bench: dict) -> list[str]:
    """The playbook's validation gate, as pass/fail lines."""
    checks = [
        ("Sharpe above 0.8", strategy.get("sharpe", 0) > 0.8),
        ("Max drawdown under 25%", strategy.get("max_drawdown", -1) > -0.25),
        ("Sharpe beats benchmark", strategy.get("sharpe", 0) > bench.get("sharpe", 0)),
        ("Drawdown smaller than benchmark", strategy.get("max_drawdown", -1) > bench.get("max_drawdown", -1)),
    ]
    return [f"  [{'PASS' if ok else 'FAIL'}] {name}" for name, ok in checks]


def cmd_backtest(args) -> None:
    cfg = load_config(args.config)
    market = _market(args, cfg)
    result = run_backtest(market, cfg, args.start, args.end)
    stats = result.stats()
    eq = result.equity
    if args.put_index:
        put = read_cboe_index(args.put_index).loc[eq.index[0] : eq.index[-1]]
        bench_name = "Cboe PutWrite index (PUT)"
    else:
        put = simple_putwrite(market, cfg.universe.index[0], eq.index[0], eq.index[-1])
        bench_name = "Simple put-write benchmark (1-month ATM, cash-secured, same data)"
    put = put / put.iloc[0] * eq.iloc[0]  # same starting capital as the strategy
    bench = summarize(put)
    print(format_summary("Strategy", stats))
    print(format_summary(bench_name, bench))
    print("Validation gate:")
    print("\n".join(gate(stats, bench)))
    if args.source == "synthetic":
        print("\nSynthetic data tests the machinery only; it says nothing about real profitability.")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        result.trade_frame().to_csv(out / "trades.csv", index=False)
        pd.DataFrame({"strategy": eq, "benchmark": put.reindex(eq.index).ffill()}).to_csv(
            out / "equity.csv", index_label="date"
        )
        pd.DataFrame(result.skips, columns=["date", "symbol", "sleeve", "reason"]).to_csv(
            out / "skips.csv", index=False
        )
        print(f"Wrote trades.csv, equity.csv and skips.csv to {out}")


def cmd_scan(args, record: bool) -> None:
    cfg = load_config(args.config)
    market = _market(args, cfg)
    journal = Journal.load(args.journal, cfg.account.starting_capital)
    plan = run_day(market, cfg, journal, record=record)
    text = report(plan, journal, recorded=record)
    if record:
        journal.save(args.journal)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="optlab", description="Options playbook backtester and daily scanner")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, default_source):
        sp.add_argument(
            "--config", nargs="*", help="TOML files overriding defaults, later ones win (see config/default.toml)"
        )
        sp.add_argument("--source", default=default_source, choices=["synthetic", "csv", "yahoo"])
        sp.add_argument("--seed", type=int, default=1, help="synthetic data seed")
        sp.add_argument("--chains", nargs="*", help="option chain CSV/Parquet files (csv source)")
        sp.add_argument("--vix", help="Cboe VIX_History.csv (csv source)")
        sp.add_argument("--vix3m", help="Cboe VIX3M_History.csv (csv source)")
        sp.add_argument("--out", help="output directory (backtest) or report file (scan)")

    bt = sub.add_parser("backtest", help="run the playbook over history")
    common(bt, "synthetic")
    bt.add_argument("--start")
    bt.add_argument("--end")
    bt.add_argument("--put-index", help="Cboe PUT_History.csv to use as the benchmark")

    for name, helptext in (
        ("scan", "report today's signals and suggestions"),
        ("paper", "scan and record in the paper journal"),
    ):
        sp = sub.add_parser(name, help=helptext)
        common(sp, "yahoo")
        sp.add_argument("--journal", default="journal/paper.json")

    args = p.parse_args(argv)
    if args.cmd == "backtest":
        cmd_backtest(args)
    else:
        cmd_scan(args, record=args.cmd == "paper")


if __name__ == "__main__":
    main()
