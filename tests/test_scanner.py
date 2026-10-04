from optlab.cli import main
from optlab.journal import Journal
from optlab.scanner import report, run_day


def test_paper_day_records_and_round_trips(market, cfg, tmp_path):
    journal = Journal.new(cfg.account.starting_capital)
    plan = run_day(market, cfg, journal, record=True)
    text = report(plan, journal, recorded=True)
    assert "Suggestions only" in text and "Regime" in text
    assert len(journal.open) == len(plan.proposals)
    path = tmp_path / "paper.json"
    journal.save(path)
    again = Journal.load(path, 0)
    assert again.cash == journal.cash and len(again.open) == len(journal.open)
    # Same day again: not a new entry week, so nothing new opens.
    plan2 = run_day(market, cfg, again, record=True)
    assert not plan2.proposals


def test_cli_backtest_smoke(tmp_path, capsys):
    main(["backtest", "--source", "synthetic", "--start", "2016-01-01", "--end", "2016-06-30", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert "Validation gate" in out
    assert (tmp_path / "trades.csv").exists()


def test_config_files_layer_in_order():
    from optlab.config import load_config

    cfg = load_config(["config/default.toml", "config/dolthub.toml"])
    assert cfg.entry.short_dte_max == 60  # dolthub.toml wins
    assert cfg.risk.csp_risk_basis == "stress"  # kept from default.toml
